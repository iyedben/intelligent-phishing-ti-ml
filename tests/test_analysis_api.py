"""API/service orchestration tests using only local deterministic doubles."""

from __future__ import annotations

from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from src.api import create_app
from src.services.analysis_service import AnalysisService
from src.threat_intel.correlation import CorrelationEngine
from src.threat_intel.models import EvidenceLevel, LookupStatus, ThreatIntelResult


URL = "https://example.com"


class FakeExplainer:
    def __init__(self, *, prediction: str = "phishing", probability: float = 0.95) -> None:
        self.prediction = prediction
        self.probability = probability
        self.urls: list[str] = []

    def explain_url(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        contributions = [
            {
                "feature": "URLLength", "value": 19, "shap_value": 0.31,
                "direction": "phishing", "statement": "URLLength contributed toward phishing in this model.",
            },
            {
                "feature": "IsHTTPS", "value": 1, "shap_value": -0.05,
                "direction": "legitimate", "statement": "IsHTTPS contributed toward legitimate in this model.",
            },
        ]
        return {
            "url": url,
            "prediction": self.prediction,
            "phishing_probability": self.probability,
            "threshold": 0.5,
            "positive_class": 1,
            "features": {"URLLength": 19, "IsHTTPS": 1},
            "explanation": {
                "method": "SHAP TreeExplainer", "model_output": "raw",
                "feature_perturbation": "tree_path_dependent", "explained_class": 1,
                "shap_version": "test-double", "base_value": 0.4,
                "shap_sum": self.probability - 0.4,
                "reconstructed_output": self.probability,
                "model_phishing_probability": self.probability,
                "additivity_absolute_error": 0.0,
                "contributions": contributions,
            },
        }


class FakeProvider:
    def __init__(self, result: ThreatIntelResult, *, raises: bool = False) -> None:
        self.name = result.source
        self.result = result
        self.raises = raises
        self.urls: list[str] = []

    def check_url(self, url: str) -> ThreatIntelResult:
        self.urls.append(url)
        if self.raises:
            raise RuntimeError("private internal text and fake-secret")
        return self.result.model_copy(deep=True)


def result(
    source: str,
    status: LookupStatus,
    *,
    match: bool | None,
    verdict: str,
    evidence: EvidenceLevel = EvidenceLevel.NONE,
    error: str | None = None,
) -> ThreatIntelResult:
    return ThreatIntelResult(
        source=source,
        queried_url=URL,
        lookup_status=status,
        match=match,
        verdict=verdict,
        evidence_level=evidence,
        error=error,
        details={"fixture": "mocked"},
    )


class RecordingCorrelationEngine:
    def __init__(self) -> None:
        self.delegate = CorrelationEngine()
        self.calls: list[tuple[str, object, list[ThreatIntelResult]]] = []

    def assess(self, url: str, ml: object, providers: list[ThreatIntelResult]):
        self.calls.append((url, ml, providers))
        return self.delegate.assess(url, ml, providers)


def service_for(
    providers: list[FakeProvider],
    *,
    explainer: FakeExplainer | None = None,
    correlation_engine: Any | None = None,
) -> tuple[AnalysisService, FakeExplainer, RecordingCorrelationEngine]:
    fake_explainer = explainer or FakeExplainer()
    correlation = correlation_engine or RecordingCorrelationEngine()
    return (
        AnalysisService(
            explainer=fake_explainer,
            providers=providers,
            correlation_engine=correlation,
        ),
        fake_explainer,
        correlation,
    )


def test_valid_analysis_endpoint_preserves_ml_shap_ti_and_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = [
        FakeProvider(result("phishtank", LookupStatus.MATCH, match=True, verdict="phishing", evidence=EvidenceLevel.CONFIRMED)),
        FakeProvider(result("urlhaus", LookupStatus.NO_MATCH, match=False, verdict="no_urlhaus_record")),
        FakeProvider(result("openphish", LookupStatus.NO_MATCH, match=False, verdict="not_listed_in_community_feed")),
    ]
    service, explainer, correlation = service_for(providers)

    def deny_external(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("analysis attempted an unconfigured direct network request")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_external)
    with TestClient(create_app(analysis_service=service)) as client:
        response = client.post("/api/v1/analyze", json={"url": URL})

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == URL
    assert body["ml"] == {
        "prediction": "phishing", "phishing_probability": 0.95,
        "threshold": 0.5, "positive_class": 1,
    }
    assert body["explanation"]["method"] == "SHAP TreeExplainer"
    assert body["explanation"]["explained_class"] == 1
    assert body["explanation"]["top_contributions"][0]["direction"] == "phishing"
    assert body["explanation"]["top_contributions"][1]["direction"] == "legitimate"
    assert [item["source"] for item in body["threat_intelligence"]] == ["phishtank", "urlhaus", "openphish"]
    assert body["threat_intelligence"][1]["verdict"] == "no_urlhaus_record"
    assert body["correlation"]["status"] == "correlated_phishing_evidence"
    assert correlation.calls[0][0] == URL
    assert explainer.urls == [URL]
    assert all(provider.urls == [URL] for provider in providers)


@pytest.mark.parametrize("url", ["", "   ", "example.com", "ftp://example.com/file", "https:///no-host", "https://bad host/"])
def test_invalid_url_returns_422_before_service_runs(url: str) -> None:
    providers = [FakeProvider(result("urlhaus", LookupStatus.NO_MATCH, match=False, verdict="no_urlhaus_record"))]
    service, explainer, _ = service_for(providers)
    with TestClient(create_app(analysis_service=service)) as client:
        response = client.post("/api/v1/analyze", json={"url": url})
    assert response.status_code == 422
    assert explainer.urls == []
    assert providers[0].urls == []


def test_url_is_not_rewritten_and_request_schema_is_documented() -> None:
    exact_url = "HTTPS://example.com/Path?q=a%2Fb#frag"
    provider = FakeProvider(result("urlhaus", LookupStatus.NO_MATCH, match=False, verdict="no_urlhaus_record"))
    service, explainer, _ = service_for([provider])
    application = create_app(analysis_service=service)
    with TestClient(application) as client:
        response = client.post("/api/v1/analyze", json={"url": exact_url})
    assert response.status_code == 200
    assert response.json()["url"] == exact_url
    assert explainer.urls == [exact_url]
    openapi = application.openapi()
    operation = openapi["paths"]["/api/v1/analyze"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/AnalyzeRequest"
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/AnalyzeResponse"
    assert "/health" in openapi["paths"]


def test_unavailable_provider_and_no_matches_remain_distinct() -> None:
    providers = [
        FakeProvider(result("urlhaus", LookupStatus.NO_MATCH, match=False, verdict="no_urlhaus_record")),
        FakeProvider(result("phishtank", LookupStatus.UNAVAILABLE, match=None, verdict="unknown", error="http_error_403")),
        FakeProvider(result("openphish", LookupStatus.NO_MATCH, match=False, verdict="not_listed_in_community_feed")),
    ]
    service, _, _ = service_for(providers)
    body = service.analyze(URL).model_dump(mode="json")
    by_source = {item["source"]: item for item in body["threat_intelligence"]}
    assert by_source["urlhaus"]["lookup_status"] == "no_match"
    assert by_source["openphish"]["lookup_status"] == "no_match"
    assert by_source["phishtank"]["lookup_status"] == "unavailable"
    assert by_source["phishtank"]["confidence"] is None
    assert body["correlation"]["status"] == "ml_only_partial_ti"


def test_urlhaus_malware_match_is_not_converted_to_phishing() -> None:
    malware = result(
        "urlhaus", LookupStatus.MATCH, match=True,
        verdict="malware_distribution", evidence=EvidenceLevel.CONFIRMED,
    )
    service, _, _ = service_for(
        [FakeProvider(malware)], explainer=FakeExplainer(prediction="legitimate", probability=0.2)
    )
    body = service.analyze(URL).model_dump(mode="json")
    assert body["threat_intelligence"][0]["verdict"] == "malware_distribution"
    assert body["correlation"]["status"] == "malware_intelligence_ml_not_phishing"
    assert body["correlation"]["disagreement"] is False


def test_provider_unexpected_failure_is_partial_and_sanitized() -> None:
    provider = FakeProvider(
        result("urlhaus", LookupStatus.UNAVAILABLE, match=None, verdict="unknown"), raises=True
    )
    service, _, _ = service_for([provider])
    body = service.analyze(URL).model_dump(mode="json")
    assert body["threat_intelligence"][0]["error"] == "provider_internal_error"
    assert "fake-secret" not in str(body)
    assert body["correlation"]["status"] == "ml_only_ti_unavailable"


def test_internal_failure_returns_safe_http_500() -> None:
    class BrokenExplainer:
        def explain_url(self, url: str) -> dict[str, Any]:
            raise RuntimeError("SECRET-KEY C:\\private\\model.joblib")

    service, _, _ = service_for([], explainer=BrokenExplainer())
    with TestClient(create_app(analysis_service=service)) as client:
        response = client.post("/api/v1/analyze", json={"url": URL})
    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis could not be completed."}
    assert "SECRET-KEY" not in response.text
    assert "private" not in response.text


def test_default_service_factory_is_lazy_and_reused() -> None:
    created: list[object] = []
    service, _, _ = service_for([
        FakeProvider(result("urlhaus", LookupStatus.NO_MATCH, match=False, verdict="no_urlhaus_record"))
    ])

    def factory() -> AnalysisService:
        created.append(service)
        return service

    application = create_app(service_factory=factory)
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert created == []
        assert client.post("/api/v1/analyze", json={"url": URL}).status_code == 200
        assert client.post("/api/v1/analyze", json={"url": URL}).status_code == 200
    assert created == [service]


def test_service_initialization_failure_returns_safe_http_500() -> None:
    def broken_factory() -> AnalysisService:
        raise RuntimeError("secret credential / private filesystem path")

    with TestClient(create_app(service_factory=broken_factory)) as client:
        response = client.post("/api/v1/analyze", json={"url": URL})
    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis could not be completed."}
    assert "secret" not in response.text
    assert "private" not in response.text
