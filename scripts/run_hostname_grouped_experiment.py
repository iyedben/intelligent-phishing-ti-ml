"""Run the Phase 11B hostname-grouped model comparison.

This script is deliberately separate from the historical URL-grouped experiment.
It fits all candidates on the hostname-grouped training partition, uses only
training rows for grouped CV, selects the operating threshold on validation
rows, and scores the frozen winner once on the held-out test partition.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.ml.experiments import (
    DOMAIN_GROUPING_METHOD,
    DOMAIN_GROUPING_POLICY,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_NAMES,
    POSITIVE_CLASS,
    RANDOM_STATE,
    build_dataset,
    calculate_metrics,
    dataset_sha256,
    domain_group_key,
    domain_grouped_three_way_split,
    hostname_grouped_cv_indices,
    make_baseline_models,
    phishing_probability,
    runtime_versions,
    save_model_artifact,
    threshold_results,
)


DEFAULT_DATASET = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
DEFAULT_REPORT = Path("reports/hostname_grouped_model_comparison.json")
DEFAULT_MODEL_DIR = Path("models")
CV_FOLDS = 3
THRESHOLD_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
GROUPING_METADATA = {
    "method": DOMAIN_GROUPING_METHOD,
    "description": DOMAIN_GROUPING_POLICY,
    "claim_boundary": "hostname isolation only; not registrable-domain/eTLD+1 isolation",
}


def select_validation_threshold(
    y_true: Any,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLD_GRID,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose a validation-only operating point with a documented tie-break."""
    candidates = threshold_results(y_true, probabilities, thresholds=thresholds)
    ordered = candidates.sort_values(
        ["f1_phishing", "recall_phishing", "precision_phishing", "false_positive_rate", "threshold"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    selected = ordered.iloc[0]
    return float(selected["threshold"]), candidates.to_dict(orient="records")


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a model from validation/CV evidence only, never test metrics."""
    if not records:
        raise ValueError("at least one candidate record is required")
    return min(
        records,
        key=lambda record: (
            -float(record["validation_metrics"]["pr_auc"]),
            -float(record["cv_metrics"]["pr_auc_mean"]),
            -float(record["validation_metrics"]["recall_phishing"]),
            -float(record["validation_metrics"]["precision_phishing"]),
            float(record["validation_metrics"]["false_positive_rate"]),
            str(record["model"]),
        ),
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Keep estimator configuration readable without serializing live objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "get_params"):
        class_name = f"{value.__class__.__module__}.{value.__class__.__name__}"
        return {"class": class_name, "params": _json_safe(value.get_params(deep=False))}
    return repr(value)


def _estimator_configuration(estimator: Any) -> dict[str, Any]:
    return _json_safe(estimator.get_params(deep=False))


def _partition_metadata(dataset: Any, partitions: dict[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, indices in partitions.items():
        urls = dataset.urls.iloc[indices].astype(str)
        output[name] = {
            "rows": int(len(indices)),
            "hostname_groups": int(len({
                domain_group_key(url)
                for url in urls
            })),
            "phishing_rows": int(dataset.y.iloc[indices].sum()),
            "legitimate_rows": int(len(indices) - dataset.y.iloc[indices].sum()),
        }
    return output


def _leakage_metadata(dataset: Any, partitions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Measure split overlap after the split is built; no rows are removed."""
    domain_sets = {
        name: {domain_group_key(str(dataset.urls.iloc[index])) for index in indices}
        for name, indices in partitions.items()
    }
    url_sets = {
        name: set(dataset.urls.iloc[indices].astype(str)) for name, indices in partitions.items()
    }
    normalized_sets = {
        name: {url.casefold().rstrip("/") for url in values}
        for name, values in url_sets.items()
    }
    vector_sets = {
        name: {
            tuple(row)
            for row in dataset.X.iloc[indices].to_numpy(dtype=np.int64)
        }
        for name, indices in partitions.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    return {
        "domain_group_overlap": {
            f"{left}_vs_{right}": int(len(domain_sets[left] & domain_sets[right]))
            for left, right in pairs
        },
        "exact_url_overlap": {
            f"{left}_vs_{right}": int(len(url_sets[left] & url_sets[right]))
            for left, right in pairs
        },
        "normalized_url_overlap": {
            f"{left}_vs_{right}": int(len(normalized_sets[left] & normalized_sets[right]))
            for left, right in pairs
        },
        "feature_vector_overlap": {
            f"{left}_vs_{right}": int(len(vector_sets[left] & vector_sets[right]))
            for left, right in pairs
        },
    }


def _cross_validate_candidate(
    name: str,
    estimator: Any,
    X_train: Any,
    y_train: Any,
    urls_train: Any,
    *,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    splits = hostname_grouped_cv_indices(
        y_train, urls_train, n_splits=CV_FOLDS, random_state=random_state
    )
    fold_metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for train_indices, validation_indices in splits:
        fold_estimator = deepcopy(estimator)
        fold_estimator.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
        probabilities = phishing_probability(fold_estimator, X_train.iloc[validation_indices])
        fold_metrics.append(calculate_metrics(y_train.iloc[validation_indices], probabilities, threshold=0.5))
    elapsed = time.perf_counter() - started
    metric_names = (
        "pr_auc",
        "recall_phishing",
        "precision_phishing",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
    )
    summary: dict[str, Any] = {
        "model": name,
        "folds": fold_metrics,
        "cv_folds": CV_FOLDS,
        "runtime_seconds": elapsed,
    }
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in fold_metrics], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=1))
    return summary, elapsed


def run_experiment(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    report_path: Path = DEFAULT_REPORT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Execute the controlled hostname-grouped comparison once."""
    dataset = build_dataset(dataset_path)
    partitions = domain_grouped_three_way_split(
        dataset.X, dataset.y, dataset.urls, random_state=random_state
    )
    train_indices = partitions["train"]
    validation_indices = partitions["validation"]
    # Test rows are intentionally not materialized or scored until selection is frozen.
    X_train = dataset.X.iloc[train_indices]
    y_train = dataset.y.iloc[train_indices]
    urls_train = dataset.urls.iloc[train_indices]
    X_validation = dataset.X.iloc[validation_indices]
    y_validation = dataset.y.iloc[validation_indices]

    models = make_baseline_models(random_state=random_state)
    cv_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    candidate_records: list[dict[str, Any]] = []
    for name, estimator in models.items():
        cv_summary, _ = _cross_validate_candidate(
            name, estimator, X_train, y_train, urls_train, random_state=random_state
        )
        cv_results[name] = cv_summary
        started = time.perf_counter()
        fitted = deepcopy(estimator)
        fitted.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        fitted_models[name] = fitted
        validation_probabilities = phishing_probability(fitted, X_validation)
        threshold, threshold_table = select_validation_threshold(y_validation, validation_probabilities)
        validation_metrics = calculate_metrics(y_validation, validation_probabilities, threshold=threshold)
        candidate_records.append(
            {
                "model": name,
                "configuration": _estimator_configuration(fitted),
                "cv_metrics": {
                    key: value for key, value in cv_summary.items()
                    if key.endswith("_mean") or key.endswith("_std")
                },
                "cv_runtime_seconds": cv_summary["runtime_seconds"],
                "fit_runtime_seconds": fit_seconds,
                "validation_metrics": validation_metrics,
                "validation_threshold_candidates": threshold_table,
                "selected_threshold": threshold,
            }
        )

    selected = select_candidate(candidate_records)
    selected_name = str(selected["model"])
    selected_threshold = float(selected["selected_threshold"])
    # This is the first access to test rows or test labels in the protocol.
    X_test = dataset.X.iloc[partitions["test"]]
    y_test = dataset.y.iloc[partitions["test"]]
    test_probabilities = phishing_probability(fitted_models[selected_name], X_test)
    final_test_metrics = calculate_metrics(y_test, test_probabilities, threshold=selected_threshold)

    partition_stats = _partition_metadata(dataset, partitions)
    leakage_stats = _leakage_metadata(dataset, partitions)
    base_metadata: dict[str, Any] = {
        "experiment_name": "hostname-grouped-model-comparison",
        "research_status": "research_only_not_production",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256(dataset_path),
            "rows": int(len(dataset.y)),
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "target_mapping": {"1": "phishing", "0": "legitimate"},
        "grouping_policy": deepcopy(GROUPING_METADATA),
        "split": {
            "random_state": random_state,
            "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "partitions": partition_stats,
            "leakage_verification": leakage_stats,
            "cv": {
                "strategy": "deterministic greedy hostname-group allocation",
                "n_splits": CV_FOLDS,
                "scope": "training partition only",
            },
        },
        "candidates": candidate_records,
        "selection": {
            "selected_model": selected_name,
            "selected_threshold": selected_threshold,
            "rule": "validation PR-AUC, then grouped-CV PR-AUC, recall, precision, lower FPR, model-name tie-break",
            "threshold_rule": "validation F1, then recall, precision, lower FPR, lower threshold",
            "rationale": "Test data was not used for candidate or threshold selection.",
        },
        "final_test_metrics": final_test_metrics,
        "runtime_versions": runtime_versions(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_records: list[dict[str, Any]] = []
    for record in candidate_records:
        slug = str(record["model"]).lower().replace(" ", "_").replace("/", "_")
        artifact_path = model_dir / f"hostname_grouped_{slug}_seed{random_state}_candidate.joblib"
        model_metadata = deepcopy(base_metadata)
        model_metadata["candidate"] = deepcopy(record)
        model_metadata["selected"] = record["model"] == selected_name
        model_metadata["final_test_metrics"] = final_test_metrics if model_metadata["selected"] else None
        saved_path = save_model_artifact(artifact_path, fitted_models[str(record["model"])], model_metadata)
        artifact_record = {
            "model": record["model"],
            "path": str(saved_path),
            "sha256": _file_sha256(saved_path),
            "selected": bool(model_metadata["selected"]),
        }
        metadata_path = report_path.parent / f"{saved_path.stem}_metadata.json"
        metadata_path.write_text(json.dumps({**model_metadata, "artifact": artifact_record}, indent=2) + "\n", encoding="utf-8")
        artifact_record["metadata_path"] = str(metadata_path)
        artifact_records.append(artifact_record)

    report = {
        **base_metadata,
        "artifacts": artifact_records,
        "historical_artifact": {
            "path": "models/url_only_random_forest_seed2026_candidate.joblib",
            "role": "historical URL-grouped reference; not a fair domain-grouped training baseline",
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    report = run_experiment(
        dataset_path=args.dataset,
        report_path=args.report,
        model_dir=args.model_dir,
    )
    print(json.dumps({
        "selected_model": report["selection"]["selected_model"],
        "selected_threshold": report["selection"]["selected_threshold"],
        "final_test_metrics": report["final_test_metrics"],
        "artifacts": report["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
