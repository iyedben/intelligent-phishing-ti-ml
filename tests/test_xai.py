"""Tests against the saved Phase 5 model; no training is performed here."""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from src.features import FEATURE_SCHEMA_VERSION
from src.xai import URLShapExplainer
from src.xai.shap_explainer import DEFAULT_ARTIFACT_PATH, model_feature_names


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
