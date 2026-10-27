"""Real local ML/XAI integration and explicit URL network-boundary tests."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import pytest
import requests
from fastapi.testclient import TestClient

from src.api import create_app
from src.services.analysis_service import AnalysisService
from src.threat_intel.correlation import CorrelationEngine
from src.threat_intel.models import EvidenceLevel, LookupStatus, ThreatIntelResult
from src.xai import URLShapExplainer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "url_only_random_forest_seed2026_candidate.joblib"
PHISHING_URL = "http://a-a-a-a-a-a-a-a-a-a-a-a-a-a.com/1234567890?x=1&y=2&z=3"
LEGITIMATE_URL = "https://example.com"
FAKE_CREDENTIAL = "phase9-test-only-fake-credential"


@pytest.fixture(scope="module")
def explainer() -> URLShapExplainer:
    if not ARTIFACT.is_file():
        pytest.skip("Local model artifact is unavailable; it is intentionally not Git-tracked")
    return URLShapExplainer.from_artifact(ARTIFACT)


class StubProvider:
    def __init__(self, source: str, status: LookupStatus, *, verdict: str = "unknown", evidence: EvidenceLevel = EvidenceLevel.NONE) -> None:
        self.name = source
        self.status = status
        self.verdict = verdict
        self.evidence = evidence
        self.calls: list[str] = []

    def check_url(self, url: str) -> ThreatIntelResult:
        self.calls.append(url)
        matched = self.status == LookupStatus.MATCH
        return ThreatIntelResult(
            source=self.name,
            queried_url=url,
            lookup_status=self.status,
            match=True if matched else False if self.status == LookupStatus.NO_MATCH else None,
            verdict=self.verdict if matched else "not_listed" if self.status == LookupStatus.NO_MATCH else "unknown",
            evidence_level=self.evidence if matched else EvidenceLevel.NONE,
            confidence=None,
            details={"fixture": "mocked"},
            error="request_error" if self.status == LookupStatus.UNAVAILABLE else None,
        )


def providers(
    *,
    phishtank: tuple[LookupStatus, str, EvidenceLevel] = (LookupStatus.NO_MATCH, "unknown", EvidenceLevel.NONE),
    urlhaus: tuple[LookupStatus, str, EvidenceLevel] = (LookupStatus.NO_MATCH, "unknown", EvidenceLevel.NONE),
    openphish: tuple[LookupStatus, str, EvidenceLevel] = (LookupStatus.NO_MATCH, "unknown", EvidenceLevel.NONE),
) -> list[StubProvider]:
    phishtank_status, phishtank_verdict, phishtank_evidence = phishtank
    urlhaus_status, urlhaus_verdict, urlhaus_evidence = urlhaus
    openphish_status, openphish_verdict, openphish_evidence = openphish
    return [
        StubProvider("phishtank", phishtank_status, verdict=phishtank_verdict, evidence=phishtank_evidence),
        StubProvider("urlhaus", urlhaus_status, verdict=urlhaus_verdict, evidence=urlhaus_evidence),
        StubProvider("openphish", openphish_status, verdict=openphish_verdict, evidence=openphish_evidence),
    ]


def service_for(model: URLShapExplainer, ti: list[StubProvider]) -> AnalysisService:
    return AnalysisService(explainer=model, providers=ti, correlation_engine=CorrelationEngine())


@pytest.mark.parametrize(
    ("url", "expected_prediction", "ti", "expected_correlation"),
    [
        (
            PHISHING_URL,
            "phishing",
            providers(phishtank=(LookupStatus.MATCH, "phishing", EvidenceLevel.CONFIRMED)),
            "correlated_phishing_evidence",
        ),
        (
            LEGITIMATE_URL,
            "legitimate",
            providers(),
            "ml_only_no_ti_match",
        ),
        (
            LEGITIMATE_URL,
            "legitimate",
            providers(phishtank=(LookupStatus.MATCH, "phishing", EvidenceLevel.CONFIRMED)),
            "ml_ti_disagreement",
        ),
        (
            LEGITIMATE_URL,
            "legitimate",
            providers(urlhaus=(LookupStatus.MATCH, "malware_distribution", EvidenceLevel.LISTED)),
            "malware_intelligence_ml_not_phishing",
        ),
        (
            PHISHING_URL,
            "phishing",
            providers(
                phishtank=(LookupStatus.UNAVAILABLE, "unknown", EvidenceLevel.NONE),
                urlhaus=(LookupStatus.UNAVAILABLE, "unknown", EvidenceLevel.NONE),
                openphish=(LookupStatus.UNAVAILABLE, "unknown", EvidenceLevel.NONE),
            ),
            "ml_only_ti_unavailable",
        ),
    ],
)
def test_real_model_shap_ti_correlation_api_path(
    explainer: URLShapExplainer,
    url: str,
    expected_prediction: str,
    ti: list[StubProvider],
    expected_correlation: str,
) -> None:
    with TestClient(create_app(analysis_service=service_for(explainer, ti))) as client:
        response = client.post("/api/v1/analyze", json={"url": url})

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == url
    assert body["ml"]["prediction"] == expected_prediction
    assert body["ml"]["positive_class"] == 1
    assert body["explanation"]["explained_class"] == 1
    assert body["explanation"]["method"] == "SHAP TreeExplainer"
    assert body["explanation"]["additivity_absolute_error"] < 1e-6
    assert len(body["threat_intelligence"]) == 3
    assert body["correlation"]["status"] == expected_correlation
    assert FAKE_CREDENTIAL not in response.text
    if expected_correlation == "malware_intelligence_ml_not_phishing":
        assert body["threat_intelligence"][1]["verdict"] == "malware_distribution"
        assert body["correlation"]["disagreement"] is False
    if expected_correlation == "ml_ti_disagreement":
        assert body["correlation"]["disagreement"] is True


@pytest.mark.parametrize(
    "url",
    ["", "ftp://example.com/", "example.com", "https:///missing-host", "https://bad host/", "https://[invalid"],
)
def test_invalid_input_is_rejected_before_inference(explainer: URLShapExplainer, url: str) -> None:
    ti = providers()
    with TestClient(create_app(analysis_service=service_for(explainer, ti))) as client:
        response = client.post("/api/v1/analyze", json={"url": url})
    assert response.status_code == 422
    assert not any(provider.calls for provider in ti)


def test_http_and_https_are_supported_and_exact_url_is_preserved(explainer: URLShapExplainer) -> None:
    for url in (LEGITIMATE_URL, "http://example.com/path"):
        ti = providers()
        with TestClient(create_app(analysis_service=service_for(explainer, ti))) as client:
            response = client.post("/api/v1/analyze", json={"url": url})
        assert response.status_code == 200
        assert response.json()["url"] == url
        assert all(provider.calls == [url] for provider in ti)


def test_url_length_boundary_is_enforced(explainer: URLShapExplainer) -> None:
    maximum_url = "https://example.com/" + "a" * (4096 - len("https://example.com/"))
    ti = providers()
    app = create_app(analysis_service=service_for(explainer, ti))
    with TestClient(app) as client:
        accepted = client.post("/api/v1/analyze", json={"url": maximum_url})
        rejected = client.post("/api/v1/analyze", json={"url": maximum_url + "a"})
    assert accepted.status_code == 200
    assert accepted.json()["url"] == maximum_url
    assert rejected.status_code == 422


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
    ],
)
def test_private_and_metadata_urls_are_only_parsed_as_strings(
    explainer: URLShapExplainer,
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("The application attempted a network or DNS operation")

    ti = providers()
    with TestClient(create_app(analysis_service=service_for(explainer, ti))) as client:
        # Starlette creates its in-process Windows socket pair at startup. Arm
        # the guards after that harness transport exists and before the API call.
        monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
        monkeypatch.setattr(urllib_request, "urlopen", forbidden)
        monkeypatch.setattr(urllib_request.OpenerDirector, "open", forbidden)
        monkeypatch.setattr(socket, "getaddrinfo", forbidden)
        monkeypatch.setattr(socket, "gethostbyname", forbidden)
        monkeypatch.setattr(socket, "gethostbyname_ex", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        monkeypatch.setattr(socket.socket, "connect", forbidden)
        monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
        response = client.post("/api/v1/analyze", json={"url": url})

    assert response.status_code == 200
    assert response.json()["url"] == url
    assert all(provider.calls == [url] for provider in ti)


def test_internal_exception_is_sanitized_by_api(explainer: URLShapExplainer) -> None:
    class BrokenProvider(StubProvider):
        def check_url(self, url: str) -> ThreatIntelResult:
            raise RuntimeError(f"{FAKE_CREDENTIAL} C:\\private\\trace")

    ti = [BrokenProvider("urlhaus", LookupStatus.UNAVAILABLE)]
    with TestClient(create_app(analysis_service=service_for(explainer, ti))) as client:
        response = client.post("/api/v1/analyze", json={"url": LEGITIMATE_URL})
    assert response.status_code == 200
    assert response.json()["threat_intelligence"][0]["error"] == "provider_internal_error"
    assert FAKE_CREDENTIAL not in response.text


def test_unexpected_analysis_exception_returns_safe_500() -> None:
    class BrokenExplainer:
        def explain_url(self, url: str) -> dict[str, Any]:
            raise RuntimeError(f"{FAKE_CREDENTIAL} C:\\private\\artifact")

    service = AnalysisService(explainer=BrokenExplainer(), providers=[], correlation_engine=CorrelationEngine())
    with TestClient(create_app(analysis_service=service)) as client:
        response = client.post("/api/v1/analyze", json={"url": LEGITIMATE_URL})
    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis could not be completed."}
    assert FAKE_CREDENTIAL not in response.text
    assert "private" not in response.text
