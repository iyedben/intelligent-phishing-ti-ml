"""Frontend serving and API regression tests with no external services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.schemas.analyze import (
    AnalyzeResponse,
    ExplanationResponse,
    MLResultResponse,
)


class StubAnalysisService:
    def analyze(self, url: str) -> AnalyzeResponse:
        return AnalyzeResponse(
            url=url,
            ml=MLResultResponse(
                prediction="legitimate",
                phishing_probability=0.12,
                threshold=0.5,
                positive_class=1,
            ),
            explanation=ExplanationResponse(
                method="SHAP TreeExplainer",
                explained_class=1,
                model_output="raw",
                feature_perturbation="tree_path_dependent",
                shap_version="test",
                base_value=0.1,
                shap_sum=0.02,
                reconstructed_output=0.12,
                model_phishing_probability=0.12,
                additivity_absolute_error=0.0,
                top_contributions=[],
            ),
            threat_intelligence=[],
            correlation={
                "status": "ml_only_no_ti_providers",
                "assessment": "Only the model result is available.",
                "evidence_level": "none",
                "disagreement": False,
                "rules_applied": [],
            },
        )


def test_root_serves_dashboard_and_assets() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/")
        css = client.get("/static/css/styles.css")
        js = client.get("/static/js/app.js")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Intelligent Phishing Detection" in page.text
    assert 'id="analysis-form"' in page.text
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert "prefers-reduced-motion" in css.text
    assert js.status_code == 200
    assert 'fetch("/api/v1/analyze"' in js.text
    assert "innerHTML" not in js.text


def test_frontend_coexists_with_health_docs_and_analysis_without_network(monkeypatch) -> None:
    def deny_external(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("in-process frontend integration attempted an external request")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_external)
    with TestClient(create_app(analysis_service=StubAnalysisService())) as client:
        health = client.get("/health")
        docs = client.get("/docs")
        analysis = client.post("/api/v1/analyze", json={"url": "https://example.com"})
        invalid = client.post("/api/v1/analyze", json={"url": "not-a-url"})

    assert health.status_code == 200 and health.json() == {"status": "ok"}
    assert docs.status_code == 200 and "Swagger UI" in docs.text
    assert analysis.status_code == 200
    assert analysis.json()["ml"]["phishing_probability"] == 0.12
    assert analysis.json()["explanation"]["explained_class"] == 1
    assert analysis.json()["correlation"]["status"] == "ml_only_no_ti_providers"
    assert invalid.status_code == 422


def test_openapi_keeps_analysis_api_documented() -> None:
    application = create_app()
    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/api/v1/analyze" in schema["paths"]
    assert "/" not in schema["paths"]
    assert schema["components"]["schemas"]["AnalyzeRequest"]["properties"]["url"]["maxLength"] == 4096


def test_frontend_error_loading_and_empty_result_paths_are_present() -> None:
    script = (Path(__file__).resolve().parents[1] / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    page = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'response.status === 422' in script
    assert 'response.status >= 500' in script
    assert 'Unable to reach the analysis API.' in script
    assert 'No provider results were returned.' in script
    assert 'No feature contributions were returned.' in page
    assert 'analyzeButton.disabled = true' in script
    assert 'loading.hidden = false' in script
    assert 'maxlength="4096"' in page
    assert 'fetch(submittedUrl' not in script
