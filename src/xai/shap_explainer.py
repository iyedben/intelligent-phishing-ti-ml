"""SHAP explanations for the saved Random Forest URL classifier.

This module only parses the supplied URL and evaluates a previously fitted
model. It never performs network requests or fits/tunes a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

from src.features import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, extract_url_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "url_only_random_forest_seed2026_candidate.joblib"
PHISHING_CLASS = 1
MODEL_OUTPUT = "raw"
FEATURE_PERTURBATION = "tree_path_dependent"
ADDITIVITY_ATOL = 1e-6


def model_feature_names() -> tuple[str, ...]:
    """Return canonical numeric model features in schema order."""
    return tuple(spec.name for spec in FEATURE_SCHEMA if spec.python_type is int)


def _python_scalar(value: Any) -> int | float | str | bool | None:
    """Convert NumPy scalars into JSON-compatible Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError(f"Feature value is not a scalar: {type(value).__name__}")


def _class_output_index(classes: np.ndarray, label: int) -> int:
    matches = np.flatnonzero(np.asarray(classes) == label)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one model output for class {label}; classes={classes!r}")
    return int(matches[0])


def _class_slice(values: Any, *, output_index: int, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize supported SHAP binary-classifier shapes to (rows, features)."""
    if isinstance(values, list):
        if output_index >= len(values):
            raise ValueError("SHAP returned fewer class outputs than the model has")
        result = np.asarray(values[output_index], dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 3:
            if result.shape == (n_rows, n_features, 2):
                result = result[:, :, output_index]
            elif result.shape == (2, n_rows, n_features):
                result = result[output_index, :, :]
            else:
                raise ValueError(f"Unexpected multi-output SHAP shape: {result.shape}")
    if result.ndim == 1 and n_rows == 1 and result.shape[0] == n_features:
        result = result.reshape(1, n_features)
    if result.shape != (n_rows, n_features):
        raise ValueError(
            f"Expected phishing SHAP shape {(n_rows, n_features)}, received {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("SHAP returned non-finite contributions")
    return result


def _base_value_for_class(base_values: Any, *, output_index: int, n_rows: int) -> np.ndarray:
    """Normalize SHAP base values to one scalar per row for class output."""
    bases = np.asarray(base_values, dtype=np.float64)
    if bases.ndim == 0:
        result = np.full(n_rows, float(bases), dtype=np.float64)
    elif bases.ndim == 1:
        if len(bases) == n_rows and n_rows != 2:
            result = bases
        elif len(bases) == 2:
            result = np.full(n_rows, float(bases[output_index]), dtype=np.float64)
        elif len(bases) == n_rows:
            result = bases
        else:
            raise ValueError(f"Unexpected SHAP base-value shape: {bases.shape}")
    elif bases.ndim == 2 and bases.shape[0] == n_rows and bases.shape[1] == 2:
        result = bases[:, output_index]
    else:
        raise ValueError(f"Unexpected SHAP base-value shape: {bases.shape}")
    if not np.isfinite(result).all():
        raise ValueError("SHAP returned non-finite base values")
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class URLShapExplainer:
    """Loaded model and exact TreeExplainer for URL-only explanations."""

    estimator: Any
    explainer: shap.TreeExplainer
    feature_names: tuple[str, ...]
    feature_schema_version: str
    artifact_path: Path
    shap_version: str
    _phishing_output_index: int

    @classmethod
    def from_artifact(cls, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> "URLShapExplainer":
        """Load the existing joblib candidate and initialize Tree SHAP; never fit."""
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"Saved Random Forest artifact not found: {path}")
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "estimator" not in bundle:
            raise ValueError("Expected a Phase 5 joblib artifact bundle with an estimator")
        estimator = bundle["estimator"]
        if estimator.__class__.__name__ != "RandomForestClassifier":
            raise TypeError(f"Expected RandomForestClassifier, received {type(estimator).__name__}")

        canonical_names = model_feature_names()
        artifact_names = tuple(bundle.get("feature_names", ()))
        if artifact_names and artifact_names != canonical_names:
            raise ValueError("Artifact feature order does not match the canonical numeric schema")
        names = artifact_names or canonical_names
        schema_version = bundle.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        if schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Artifact schema {schema_version!r} differs from current schema {FEATURE_SCHEMA_VERSION!r}"
            )
        if getattr(estimator, "n_features_in_", None) != len(names):
            raise ValueError("Model input feature count does not match the canonical schema")
        if hasattr(estimator, "feature_names_in_") and tuple(estimator.feature_names_in_) != names:
            raise ValueError("Model's fitted feature names/order do not match the artifact/schema")
        output_index = _class_output_index(np.asarray(estimator.classes_), PHISHING_CLASS)

        tree_explainer = shap.TreeExplainer(
            estimator,
            model_output=MODEL_OUTPUT,
            feature_perturbation=FEATURE_PERTURBATION,
            feature_names=list(names),
        )
        return cls(
            estimator=estimator,
            explainer=tree_explainer,
            feature_names=names,
            feature_schema_version=str(schema_version),
            artifact_path=path,
            shap_version=shap.__version__,
            _phishing_output_index=output_index,
        )

    def feature_frame(self, url: str) -> tuple[dict[str, int | str], pd.DataFrame]:
        """Extract canonical features and form the model's ordered numeric row."""
        extracted = extract_url_features(url)
        row = {name: int(extracted[name]) for name in self.feature_names}
        return extracted, pd.DataFrame([row], columns=self.feature_names, dtype=np.int64)

    def explain_frame(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return phishing probabilities, class-1 SHAP values, and base values."""
        if tuple(X.columns) != self.feature_names:
            raise ValueError("Input columns/order must match the canonical model feature schema")
        if X.empty:
            raise ValueError("Cannot explain an empty feature frame")
        probabilities = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        if probabilities.shape != (len(X), len(self.estimator.classes_)):
            raise ValueError(f"Unexpected probability shape: {probabilities.shape}")
        phishing_probabilities = probabilities[:, self._phishing_output_index]
        if not np.isfinite(phishing_probabilities).all() or np.any((phishing_probabilities < 0) | (phishing_probabilities > 1)):
            raise ValueError("Model returned invalid phishing probabilities")

        explanation = self.explainer(X)
        contributions = _class_slice(
            explanation.values,
            output_index=self._phishing_output_index,
            n_rows=len(X),
            n_features=len(self.feature_names),
        )
        base_values = _base_value_for_class(
            explanation.base_values,
            output_index=self._phishing_output_index,
            n_rows=len(X),
        )
        reconstructed = base_values + contributions.sum(axis=1)
        if not np.allclose(reconstructed, phishing_probabilities, rtol=0.0, atol=ADDITIVITY_ATOL):
            max_delta = float(np.max(np.abs(reconstructed - phishing_probabilities)))
            raise ValueError(
                "TreeExplainer output does not add to the model's phishing probability; "
                f"maximum absolute difference={max_delta:.3g}"
            )
        return phishing_probabilities, contributions, base_values

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank features by mean absolute class-1 SHAP value on supplied rows."""
        _, values, _ = self.explain_frame(X)
        means = np.mean(np.abs(values), axis=0)
        table = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": means})
        table = table.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        return table

    def explain_url(self, url: str) -> dict[str, Any]:
        """Return a JSON-compatible local explanation for one URL string."""
        extracted, X = self.feature_frame(url)
        probabilities, shap_values, base_values = self.explain_frame(X)
        probability = float(probabilities[0])
        prediction = "phishing" if probability >= 0.5 else "legitimate"
        contributions: list[dict[str, Any]] = []
        sorted_indices = np.argsort(-np.abs(shap_values[0]), kind="stable")
        for index in sorted_indices:
            contribution = float(shap_values[0, index])
            direction = "phishing" if contribution > 0 else "legitimate" if contribution < 0 else "neutral"
            contributions.append(
                {
                    "feature": self.feature_names[int(index)],
                    "value": _python_scalar(X.iloc[0, int(index)]),
                    "shap_value": contribution,
                    "direction": direction,
                    "statement": humanize_contribution(self.feature_names[int(index)], contribution),
                }
            )

        base_value = float(base_values[0])
        shap_sum = float(shap_values[0].sum())
        reconstructed = base_value + shap_sum
        return {
            "url": url,
            "prediction": prediction,
            "phishing_probability": probability,
            "threshold": 0.5,
            "positive_class": PHISHING_CLASS,
            "features": {name: _python_scalar(extracted[name]) for name in self.feature_names},
            "explanation": {
                "method": "SHAP TreeExplainer",
                "model_output": MODEL_OUTPUT,
                "feature_perturbation": FEATURE_PERTURBATION,
                "explained_class": PHISHING_CLASS,
                "shap_version": self.shap_version,
                "base_value": base_value,
                "shap_sum": shap_sum,
                "reconstructed_output": reconstructed,
                "model_phishing_probability": probability,
                "additivity_absolute_error": abs(reconstructed - probability),
                "contributions": contributions,
            },
        }


def humanize_contribution(feature: str, shap_value: float) -> str:
    """Describe a model contribution without implying real-world causation."""
    if shap_value > 0:
        return f"{feature} contributed toward phishing in this model."
    if shap_value < 0:
        return f"{feature} contributed toward legitimate in this model."
    return f"{feature} made no directional contribution in this explanation."


def explain_url(url: str) -> dict[str, Any]:
    """Load the saved candidate and explain one URL without contacting it."""
    return URLShapExplainer.from_artifact().explain_url(url)
