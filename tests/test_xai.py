"""Tests against the saved Phase 5 model; no training is performed here."""

from __future__ import annotations

import json
import hashlib
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import joblib

from src.features import FEATURE_SCHEMA_VERSION
from src.xai import URLShapExplainer
from src.xai.shap_explainer import (
    DEFAULT_ARTIFACT_PATH,
    _class_output_index,
    model_feature_names,
)


SELECTED_ARTIFACT_PATH = Path("models/hostname_grouped_phase11b/hostname_grouped_random_forest_seed2026_candidate.joblib")
SELECTED_ARTIFACT_SHA256 = "ceed3f58cedca8493a3a591a7ce2f3ce674a6f6e5fb8b70d42d35637f41793f4"
HISTORICAL_ARTIFACT_PATH = Path("models/url_only_random_forest_seed2026_candidate.joblib")
HISTORICAL_ARTIFACT_SHA256 = "ba205afaa56fb532bd4ceea566eb441d6494c39e0a811e976732e76550b7debc"


@pytest.fixture(scope="module")
def xai() -> URLShapExplainer:
    return URLShapExplainer.from_artifact()


def test_saved_model_loads_with_schema_and_explainer(xai: URLShapExplainer) -> None:
    assert DEFAULT_ARTIFACT_PATH.is_file()
    assert xai.estimator.__class__.__name__ == "RandomForestClassifier"
    assert xai.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert xai.feature_names == model_feature_names()
    assert xai.estimator.n_features_in_ == len(xai.feature_names)
    assert xai.explainer.__class__.__name__ == "TreeExplainer"
    assert xai.shap_version


def test_selected_hostname_grouped_artifact_is_the_application_candidate() -> None:
    assert DEFAULT_ARTIFACT_PATH == SELECTED_ARTIFACT_PATH.resolve()
    assert DEFAULT_ARTIFACT_PATH.is_file()
    digest = hashlib.sha256(DEFAULT_ARTIFACT_PATH.read_bytes()).hexdigest()
    assert digest == SELECTED_ARTIFACT_SHA256

    bundle = joblib.load(DEFAULT_ARTIFACT_PATH)
    metadata = bundle["metadata"]
    assert metadata["experiment_name"] == "hostname-grouped-model-comparison"
    assert metadata["grouping_policy"]["method"] == "hostname_fallback_no_offline_psl"
    assert metadata["selection"]["selected_model"] == "Random Forest"
    assert metadata["selection"]["selected_threshold"] == 0.5
    assert tuple(bundle["feature_names"]) == model_feature_names()
    assert bundle["positive_class"] == {"value": 1, "meaning": "phishing"}


@pytest.mark.skipif(not HISTORICAL_ARTIFACT_PATH.is_file(), reason="historical model artifact is unavailable")
def test_historical_url_grouped_artifact_remains_unchanged() -> None:
    digest = hashlib.sha256(HISTORICAL_ARTIFACT_PATH.read_bytes()).hexdigest()
    assert digest == HISTORICAL_ARTIFACT_SHA256


@pytest.mark.parametrize("url", [
    "https://www.ihub-tansa.jaxa.jp",
    "https://classroom.google.com",
])
def test_selected_artifact_explains_requested_urls_as_class_one(xai: URLShapExplainer, url: str) -> None:
    result = xai.explain_url(url)
    assert result["url"] == url
    assert 0.0 <= result["phishing_probability"] <= 1.0
    assert result["threshold"] == 0.5
    assert result["positive_class"] == 1
    assert result["explanation"]["explained_class"] == 1
    assert result["explanation"]["model_phishing_probability"] == result["phishing_probability"]


def test_shap_shape_and_probability_additivity(xai: URLShapExplainer) -> None:
    _, X = xai.feature_frame("https://login.example.org/user?id=15")
    probabilities, contributions, base_values = xai.explain_frame(X)
    assert contributions.shape == (1, len(xai.feature_names))
    assert base_values.shape == (1,)
    assert np.isfinite(contributions).all()
    assert np.isclose(base_values[0] + contributions[0].sum(), probabilities[0], atol=1e-6)


def test_local_explanation_is_deterministic_and_json_serializable(xai: URLShapExplainer) -> None:
    url = "https://login.example.org/user?id=15"
    first = xai.explain_url(url)
    second = xai.explain_url(url)
    assert first == second
    encoded = json.dumps(first, allow_nan=False)
    assert json.loads(encoded)["url"] == url


def test_probability_prediction_semantics(xai: URLShapExplainer) -> None:
    result = xai.explain_url("http://203.0.113.10/account/secure")
    probability = result["phishing_probability"]
    assert 0.0 <= probability <= 1.0
    expected = "phishing" if probability >= 0.5 else "legitimate"
    assert result["prediction"] == expected
    assert result["positive_class"] == 1
    assert result["explanation"]["explained_class"] == 1
    assert result["explanation"]["model_phishing_probability"] == probability


def test_phishing_probability_uses_label_mapping_with_reversed_classes() -> None:
    class ReversedEstimator:
        classes_ = np.array([1, 0])

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.tile(np.array([[0.8, 0.2]]), (len(X), 1))

    class FakeShapExplainer:
        def __call__(self, X: pd.DataFrame) -> SimpleNamespace:
            values = np.zeros((len(X), len(model_feature_names()), 2), dtype=np.float64)
            values[:, :, 0] = 0.0875
            return SimpleNamespace(
                values=values,
                base_values=np.tile(np.array([[0.1, 0.9]]), (len(X), 1)),
            )

    estimator = ReversedEstimator()
    explainer = URLShapExplainer(
        estimator=estimator,
        explainer=FakeShapExplainer(),  # type: ignore[arg-type]
        feature_names=model_feature_names(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        artifact_path=Path("reversed-classes-test"),
        shap_version="test-double",
        _phishing_output_index=_class_output_index(estimator.classes_, 1),
    )
    X = pd.DataFrame([[30, 22, 0, 0, 0, 0, 0, 1]], columns=explainer.feature_names)

    probabilities, _, _ = explainer.explain_frame(X)

    assert _class_output_index(estimator.classes_, 1) == 0
    assert probabilities.tolist() == [0.8]


@pytest.mark.parametrize(
    ("probability", "expected_prediction"),
    [(0.49, "legitimate"), (0.50, "phishing"), (0.51, "phishing")],
)
def test_explain_url_uses_inclusive_half_threshold(
    xai: URLShapExplainer,
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
    expected_prediction: str,
) -> None:
    def fake_explain_frame(self: URLShapExplainer, X: pd.DataFrame):
        return (
            np.array([probability]),
            np.zeros((1, len(self.feature_names))),
            np.array([probability]),
        )

    monkeypatch.setattr(URLShapExplainer, "explain_frame", fake_explain_frame)

    result = xai.explain_url("https://example.org/")

    assert result["prediction"] == expected_prediction
    assert result["threshold"] == 0.5


def test_no_network_activity_during_url_explanation(
    xai: URLShapExplainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("URL explanation attempted network activity")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "gethostbyname", denied)
    result = xai.explain_url("https://example.org/path?q=2")
    assert result["url"] == "https://example.org/path?q=2"


def test_feature_order_mismatch_is_rejected(xai: URLShapExplainer) -> None:
    _, X = xai.feature_frame("https://example.org/")
    with pytest.raises(ValueError, match="columns/order"):
        xai.explain_frame(X.loc[:, list(reversed(X.columns))])
