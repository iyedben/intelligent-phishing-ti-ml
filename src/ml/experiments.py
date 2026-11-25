"""Reproducible URL-only phishing classification experiment components."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import ipaddress
from pathlib import Path
import platform
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, extract_url_features


RANDOM_STATE = 2026
POSITIVE_CLASS = 1  # transformed label: phishing
PHISHING_SOURCE_LABEL = 0
LEGITIMATE_SOURCE_LABEL = 1
MODEL_FEATURE_NAMES: tuple[str, ...] = tuple(
    feature.name for feature in FEATURE_SCHEMA if feature.python_type is int
)
DOMAIN_GROUPING_METHOD = "hostname_fallback_no_offline_psl"
DOMAIN_GROUPING_POLICY = (
    "case-folded parsed hostname; IP literals use canonical IP text; "
    "no Public Suffix List package or local suffix data is available"
)
URL_NORMALIZATION_POLICY = "case-fold URL text and remove trailing slash characters"


@dataclass(frozen=True, slots=True)
class ConstructedDataset:
    """Extractor-built features, phishing-positive target, and URL groups."""

    X: pd.DataFrame
    y: pd.Series
    urls: pd.Series
    source_labels: pd.Series
    quality: dict[str, Any]


def build_dataset(csv_path: str | Path) -> ConstructedDataset:
    """Build model inputs only by applying the canonical extractor to each URL.

    Source label 0 is mapped to positive class 1 (phishing); source label 1 is
    mapped to 0 (legitimate). The raw CSV is read-only and never rewritten.
    ``Domain``, ``FILENAME``, ``Title``, and all precomputed feature columns are
    excluded from the model matrix.
    """
    frame = pd.read_csv(csv_path)
    required = {"URL", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame["URL"].isna().any():
        raise ValueError("Dataset contains missing URLs; no rows were silently dropped")
    if not frame["label"].isin([PHISHING_SOURCE_LABEL, LEGITIMATE_SOURCE_LABEL]).all():
        raise ValueError("Expected source labels 0=phishing and 1=legitimate")

    extracted = pd.DataFrame.from_records(
        [extract_url_features(url) for url in frame["URL"].astype(str)],
        columns=tuple(feature.name for feature in FEATURE_SCHEMA),
        index=frame.index,
    )
    # Domain is retained by the extractor for compatibility/audit, but is a
    # high-cardinality identifier and is not a numerical model input.
    X = extracted.loc[:, list(MODEL_FEATURE_NAMES)].copy()
    source_labels = frame["label"].astype("int8").copy()
    y = source_labels.map({PHISHING_SOURCE_LABEL: 1, LEGITIMATE_SOURCE_LABEL: 0}).astype("int8")
    urls = frame["URL"].astype("string").copy()

    if X.isna().any().any():
        raise ValueError("Extractor produced missing model features; no rows were silently dropped")
    numeric = X.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Extractor produced infinite/non-finite model features")

    duplicate_url_rows = int(urls.duplicated().sum())
    duplicate_url_groups = int(urls[urls.duplicated(keep=False)].nunique())
    url_label_counts = pd.DataFrame({"URL": urls, "label": y}).groupby("URL", sort=False)["label"].nunique()
    conflicting_url_groups = int((url_label_counts > 1).sum())
    duplicate_feature_rows = int(X.duplicated().sum())
    vector_label_counts = pd.concat([X, y.rename("_target")], axis=1).groupby(
        list(MODEL_FEATURE_NAMES), sort=False, dropna=False
    )["_target"].nunique()
    conflicting_feature_vectors = int((vector_label_counts > 1).sum())

    quality: dict[str, Any] = {
        "row_count": int(len(frame)),
        "feature_count": int(X.shape[1]),
        "missing_feature_values": int(X.isna().sum().sum()),
        "infinite_feature_values": int((~np.isfinite(numeric)).sum()),
        "duplicate_url_occurrences": duplicate_url_rows,
        "distinct_repeated_urls": duplicate_url_groups,
        "conflicting_duplicate_url_groups": conflicting_url_groups,
        "duplicate_feature_vector_occurrences": duplicate_feature_rows,
        "conflicting_duplicate_feature_vector_groups": conflicting_feature_vectors,
        "fully_duplicated_source_rows": int(frame.duplicated().sum()),
        "source_label_counts": {int(k): int(v) for k, v in source_labels.value_counts().sort_index().items()},
        "phishing_positive_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "rows_excluded": 0,
    }
    return ConstructedDataset(X=X, y=y, urls=urls, source_labels=source_labels, quality=quality)


def group_aware_three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    urls: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Create reproducible approximate 70/15/15 partitions with URL groups intact.

    Unique URLs are shuffled independently within each class and assigned to
    20 group folds. Folds 0–13 form training, 14–16 validation, and 17–19 test.
    This explicit allocator keeps every duplicate URL together and avoids the
    high cost of stratified splitters when nearly every row has a unique group.
    """
    fold_assignment = _assign_stratified_group_folds(y, urls, n_splits=20, random_state=random_state)

    partitions = {
        "train": np.flatnonzero(fold_assignment < 14),
        "validation": np.flatnonzero((fold_assignment >= 14) & (fold_assignment < 17)),
        "test": np.flatnonzero(fold_assignment >= 17),
    }
    if sum(len(index) for index in partitions.values()) != len(y):
        raise RuntimeError("Split did not assign every row exactly once")
    url_sets = {name: set(urls.iloc[index].astype(str)) for name, index in partitions.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = url_sets[left] & url_sets[right]
        if overlap:
            raise RuntimeError(f"URL group leakage between {left} and {right}: {len(overlap)} URLs")
    return partitions


def normalized_url_key(url: str) -> str:
    """Return the documented audit key for equivalent URL spelling checks."""
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty string")
    return url.casefold().rstrip("/")


def domain_group_key(url: str) -> str:
    """Return the explicit offline hostname-grouping key for one URL.

    A Public Suffix List implementation is not installed and no local suffix
    data is vendored, so this Phase 11A benchmark uses a documented hostname
    fallback. This keeps subdomains of the same hostname together but does not
    establish registrable-domain (eTLD+1) equivalence across sibling hosts.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty string")
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        raise ValueError("url must not contain whitespace or control characters")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"malformed URL: {exc}") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc or hostname is None:
        raise ValueError("url must be an absolute HTTP(S) URL with a hostname")

    normalized_host = hostname.rstrip(".").casefold()
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return f"hostname:{normalized_host}"
    return f"ip:{address.compressed.casefold()}"


def domain_grouped_three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    urls: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Create a deterministic hostname-grouped 70/15/15 benchmark split.

    Whole hostname groups are assigned to one of three partitions. Groups are
    ordered deterministically by descending size with seeded tie noise, then
    assigned to the partition with the lowest normalized row/class load. This
    preserves mixed-label groups and balances rows and the phishing target
    without fitting or using any model.
    """
    if not (len(X) == len(y) == len(urls)):
        raise ValueError("X, y, and urls must have equal lengths")
    if len(urls) < 3:
        raise ValueError("at least three rows are required for three partitions")

    group_keys = pd.Series([domain_group_key(url) for url in urls.astype(str)], index=urls.index)
    group_codes, unique_groups = pd.factorize(group_keys, sort=False)
    group_count = len(unique_groups)
    if group_count < 3:
        raise ValueError("at least three hostname groups are required for three partitions")

    y_values = np.asarray(y, dtype=np.int8)
    if not np.isin(y_values, [0, 1]).all():
        raise ValueError("y must contain only model targets 0 and 1")
    group_rows = np.bincount(group_codes, minlength=group_count).astype(np.float64)
    group_positive = np.bincount(group_codes, weights=y_values, minlength=group_count)
    target_rows = np.asarray([0.70, 0.15, 0.15]) * len(y_values)
    target_positive = np.asarray([0.70, 0.15, 0.15]) * int(y_values.sum())
    positive_scale = np.where(target_positive > 0, target_positive, 1.0)

    rng = np.random.default_rng(random_state)
    order = np.lexsort((rng.random(group_count), -group_rows))
    partition_for_group = np.full(group_count, -1, dtype=np.int8)
    current_rows = np.zeros(3, dtype=np.float64)
    current_positive = np.zeros(3, dtype=np.float64)
    for group_code in order:
        scores = current_rows / target_rows + current_positive / positive_scale
        partition = int(np.argmin(scores))
        partition_for_group[group_code] = partition
        current_rows[partition] += group_rows[group_code]
        current_positive[partition] += group_positive[group_code]

    row_partition = partition_for_group[group_codes]
    partitions = {
        "train": np.flatnonzero(row_partition == 0),
        "validation": np.flatnonzero(row_partition == 1),
        "test": np.flatnonzero(row_partition == 2),
    }
    if any(len(indices) == 0 for indices in partitions.values()):
        raise RuntimeError("hostname-grouped split produced an empty partition")

    partition_groups = {
        name: set(group_keys.iloc[indices].astype(str)) for name, indices in partitions.items()
    }
    partition_urls = {
        name: set(urls.iloc[indices].astype(str)) for name, indices in partitions.items()
    }
    partition_normalized_urls = {
        name: {normalized_url_key(url) for url in urls.iloc[indices].astype(str)}
        for name, indices in partitions.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if partition_groups[left] & partition_groups[right]:
            raise RuntimeError(f"hostname group leakage: {left}/{right}")
        if partition_urls[left] & partition_urls[right]:
            raise RuntimeError(f"exact URL leakage: {left}/{right}")
        if partition_normalized_urls[left] & partition_normalized_urls[right]:
            raise RuntimeError(f"normalized URL leakage: {left}/{right}")
    return partitions


def _assign_stratified_group_folds(y: pd.Series, urls: pd.Series, *, n_splits: int,
                                   random_state: int) -> np.ndarray:
    """Assign each exact URL group to one fold, stratified by its single label."""
    y_values = np.asarray(y, dtype=np.int8)
    group_codes, unique_urls = pd.factorize(urls.astype(str), sort=False)
    group_labels = pd.DataFrame({"code": group_codes, "label": y_values}).groupby("code", sort=False)["label"]
    label_counts = group_labels.nunique()
    if (label_counts > 1).any():
        raise ValueError("A duplicate URL has conflicting labels; resolve by investigation, not row dropping")
    one_label = group_labels.first().reindex(range(len(unique_urls))).to_numpy(dtype=np.int8)
    if min(np.bincount(one_label)) < n_splits:
        raise ValueError("Not enough distinct URL groups per class for requested group folds")

    rng = np.random.default_rng(random_state)
    group_fold = np.full(len(unique_urls), -1, dtype=np.int16)
    for class_label in (0, 1):
        class_groups = np.flatnonzero(one_label == class_label)
        shuffled = rng.permutation(class_groups)
        for fold_id, group_indices in enumerate(np.array_split(shuffled, n_splits)):
            group_fold[group_indices] = fold_id
    if (group_fold < 0).any():
        raise RuntimeError("Group fold allocator left URL groups unassigned")
    return group_fold[group_codes]


def _assign_balanced_hostname_folds(
    y: pd.Series,
    urls: pd.Series,
    *,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Assign whole hostname groups to balanced deterministic CV folds."""
    if len(y) != len(urls):
        raise ValueError("y and urls must have equal lengths")
    if n_splits < 2:
        raise ValueError("at least two folds are required")
    y_values = np.asarray(y, dtype=np.int8)
    if not np.isin(y_values, [0, 1]).all():
        raise ValueError("y must contain only model targets 0 and 1")

    group_keys = pd.Series([domain_group_key(url) for url in urls.astype(str)], index=urls.index)
    group_codes, unique_groups = pd.factorize(group_keys, sort=False)
    group_count = len(unique_groups)
    if group_count < n_splits:
        raise ValueError("not enough hostname groups for requested CV folds")

    group_rows = np.bincount(group_codes, minlength=group_count).astype(np.float64)
    group_positive = np.bincount(group_codes, weights=y_values, minlength=group_count)
    target_rows = np.full(n_splits, len(y_values) / n_splits, dtype=np.float64)
    target_positive = np.full(n_splits, int(y_values.sum()) / n_splits, dtype=np.float64)
    positive_scale = np.where(target_positive > 0, target_positive, 1.0)

    rng = np.random.default_rng(random_state)
    order = np.lexsort((rng.random(group_count), -group_rows))
    fold_for_group = np.full(group_count, -1, dtype=np.int16)
    current_rows = np.zeros(n_splits, dtype=np.float64)
    current_positive = np.zeros(n_splits, dtype=np.float64)
    for group_code in order:
        scores = current_rows / target_rows + current_positive / positive_scale
        fold_id = int(np.argmin(scores))
        fold_for_group[group_code] = fold_id
        current_rows[fold_id] += group_rows[group_code]
        current_positive[fold_id] += group_positive[group_code]

    fold_assignment = fold_for_group[group_codes]
    if (fold_assignment < 0).any():
        raise RuntimeError("hostname CV allocator left groups unassigned")
    return fold_assignment


def hostname_grouped_cv_indices(
    y: pd.Series,
    urls: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic hostname-isolated CV indices for training rows only."""
    fold_assignment = _assign_balanced_hostname_folds(
        y, urls, n_splits=n_splits, random_state=random_state
    )
    group_keys = pd.Series([domain_group_key(url) for url in urls.astype(str)], index=urls.index)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        validation_indices = np.flatnonzero(fold_assignment == fold_id)
        training_indices = np.flatnonzero(fold_assignment != fold_id)
        training_groups = set(group_keys.iloc[training_indices].astype(str))
        validation_groups = set(group_keys.iloc[validation_indices].astype(str))
        if training_groups & validation_groups:
            raise RuntimeError(f"hostname leakage across CV fold {fold_id}")
        if not len(validation_indices) or not len(training_indices):
            raise RuntimeError(f"hostname CV fold {fold_id} is empty")
        splits.append((training_indices, validation_indices))
    return splits


def phishing_probability(estimator: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return probability for model class 1 using the estimator's class ordering."""
    classes = np.asarray(getattr(estimator, "classes_", []))
    matching = np.flatnonzero(classes == POSITIVE_CLASS)
    if len(matching) != 1:
        raise ValueError("estimator must expose exactly one phishing-positive class 1")
    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba output does not match estimator classes_")
    return probabilities[:, int(matching[0])]


def make_baseline_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Return fixed baseline estimators; learned scaling is inside LR's pipeline."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2_000, random_state=random_state)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight=None,
            n_jobs=1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
            verbosity=0,
        ),
    }


def calculate_metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute metrics with positive class 1=phishing and phishing-first matrix."""
    truth = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    fnr = float(fn / (fn + tp)) if fn + tp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_phishing": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
        "recall_phishing": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "f1_phishing": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "tn_legitimate_correct": int(tn),
        "fp_legitimate_flagged_phishing": int(fp),
        "fn_phishing_missed": int(fn),
        "tp_phishing_correct": int(tp),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def validation_results(models: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                       X_validation: pd.DataFrame, y_validation: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit each baseline on training only and score validation predictions."""
    fitted: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        fitted[name] = estimator
        probabilities = phishing_probability(estimator, X_validation)
        rows.append({"model": name, **calculate_metrics(y_validation, probabilities, threshold=0.5)})
    return fitted, pd.DataFrame(rows).set_index("model")


def threshold_results(y_true: pd.Series, probabilities: np.ndarray,
                      thresholds: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)) -> pd.DataFrame:
    """Evaluate phishing-focused operating points without changing model fit."""
    return pd.DataFrame([calculate_metrics(y_true, probabilities, threshold=t) for t in thresholds])


def group_cross_validation(models: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                           urls_train: pd.Series, *, random_state: int = RANDOM_STATE,
                           n_splits: int = 5) -> pd.DataFrame:
    """Cross-validate on training rows only, keeping duplicate URLs in one fold."""
    from sklearn.metrics import make_scorer

    def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(fn / (fn + tp)) if fn + tp else 0.0

    scoring = {
        "accuracy": "accuracy",
        "precision_phishing": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_phishing": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1_phishing": make_scorer(f1_score, pos_label=1, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "false_negative_rate": make_scorer(false_negative_rate),
    }
    fold_assignment = _assign_stratified_group_folds(
        y_train, urls_train, n_splits=n_splits, random_state=random_state
    )
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        valid_indices = np.flatnonzero(fold_assignment == fold_id)
        train_indices = np.flatnonzero(fold_assignment != fold_id)
        train_urls = set(urls_train.iloc[train_indices].astype(str))
        valid_urls = set(urls_train.iloc[valid_indices].astype(str))
        if train_urls & valid_urls:
            raise RuntimeError(f"Duplicate URL leakage across CV fold {fold_id}")
        cv_splits.append((train_indices, valid_indices))
    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        result = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def feature_importance_tables(models: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Return estimator feature importance and standardized LR coefficients."""
    outputs: dict[str, pd.DataFrame] = {}
    for name, estimator in models.items():
        if name == "Logistic Regression":
            coefficients = estimator.named_steps["classifier"].coef_[0]
            table = pd.DataFrame({"feature": feature_names, "coefficient_standardized": coefficients})
            outputs[name] = table.assign(abs_value=table.coefficient_standardized.abs()).sort_values("abs_value", ascending=False)
        elif hasattr(estimator, "feature_importances_"):
            table = pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
            outputs[name] = table.sort_values("importance", ascending=False)
    return outputs


def dataset_sha256(path: str | Path) -> str:
    """Return SHA-256 of the unchanged source CSV."""
    digest = sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Return core runtime versions for experiment provenance."""
    import xgboost

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def save_model_artifact(path: str | Path, estimator: Any, metadata: dict[str, Any]) -> Path:
    """Save a candidate model bundle without overwriting an existing artifact."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifact: {artifact_path}")
    bundle = {
        "artifact_status": "research_candidate_not_production",
        "estimator": estimator,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": MODEL_FEATURE_NAMES,
        "positive_class": {"value": 1, "meaning": "phishing"},
        "metadata": metadata,
    }
    joblib.dump(bundle, artifact_path)
    return artifact_path
