"""Evidence-rule and provenance tests; all provider results are local fixtures."""

from src.threat_intel.correlation import CorrelationEngine
from src.threat_intel.models import (
    EvidenceLevel,
    LookupStatus,
    MLPrediction,
    ThreatIntelResult,
)
import pytest
from unittest.mock import Mock

from src.threat_intel.phishtank import PhishTankProvider
from src.threat_intel.urlhaus import URLhausProvider


URL = "https://example.test/account"


def phishtank(confirmed: bool = True) -> ThreatIntelResult:
    return ThreatIntelResult(
        source="phishtank",
        queried_url=URL,
        lookup_status=LookupStatus.MATCH,
        match=True,
        verdict="phishing" if confirmed else "unverified_phishing_submission",
        evidence_level=EvidenceLevel.CONFIRMED if confirmed else EvidenceLevel.LISTED,
        details={"phish_id": 123},
    )


def urlhaus_match() -> ThreatIntelResult:
    return ThreatIntelResult(
        source="urlhaus", queried_url=URL, lookup_status=LookupStatus.MATCH,
        match=True, verdict="malware_distribution", evidence_level=EvidenceLevel.LISTED,
        details={"id": "456", "threat": "malware_download"},
    )


def unavailable(source: str) -> ThreatIntelResult:
    return ThreatIntelResult(
        source=source, queried_url=URL, lookup_status=LookupStatus.UNAVAILABLE,
        match=None, verdict="unknown", error="timeout",
    )


def ml(prediction: str, probability: float) -> MLPrediction:
    return MLPrediction(prediction=prediction, phishing_probability=probability)


def test_agreement_keeps_each_provider_and_ml_evidence_separate() -> None:
    results = [phishtank(), ThreatIntelResult(
        source="urlhaus", queried_url=URL, lookup_status=LookupStatus.NO_MATCH,
        match=False, verdict="no_urlhaus_record",
    )]
    assessment = CorrelationEngine().assess(URL, ml("phishing", 0.95), results)
    assert assessment.correlation.status == "correlated_phishing_evidence"
    assert assessment.correlation.evidence_level == EvidenceLevel.CONFIRMED
    assert assessment.ml.phishing_probability == 0.95
    assert [item.source for item in assessment.threat_intelligence] == ["phishtank", "urlhaus"]
    assert assessment.threat_intelligence[1].verdict == "no_urlhaus_record"


def test_mocked_provider_integration_confirms_correlation_without_merging_sources() -> None:
    pt_session = Mock()
    pt_response = Mock(status_code=200)
    pt_response.raise_for_status.return_value = None
    pt_response.json.return_value = {"results": {
        "in_database": True, "verified": "y", "valid": "y", "phish_id": "mock-7"
    }}
    pt_session.post.return_value = pt_response

    uh_session = Mock()
    uh_response = Mock(status_code=200)
    uh_response.raise_for_status.return_value = None
    uh_response.json.return_value = {"query_status": "no_results"}
    uh_session.post.return_value = uh_response

    provider_results = [
        PhishTankProvider(app_key="test-key", session=pt_session).check_url(URL),
        URLhausProvider(auth_key="test-key", session=uh_session).check_url(URL),
    ]
    assessment = CorrelationEngine().assess(URL, ml("phishing", 0.95), provider_results)
    assert assessment.correlation.status == "correlated_phishing_evidence"
    assert assessment.correlation.evidence_level == EvidenceLevel.CONFIRMED
    assert assessment.threat_intelligence[0].source == "phishtank"
    assert assessment.threat_intelligence[0].verdict == "phishing"
    assert assessment.threat_intelligence[1].source == "urlhaus"
    assert assessment.threat_intelligence[1].verdict == "no_urlhaus_record"


def test_mocked_provider_integration_surfaces_disagreement() -> None:
    session = Mock()
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": {
        "in_database": True, "verified": True, "valid": True, "phish_id": "mock-8"
    }}
    session.post.return_value = response
    result = PhishTankProvider(app_key="test-key", session=session).check_url(URL)
    assessment = CorrelationEngine().assess(URL, ml("legitimate", 0.20), [result])
    assert assessment.correlation.status == "ml_ti_disagreement"
    assert assessment.correlation.disagreement is True
    assert assessment.ml.prediction == "legitimate"
    assert assessment.threat_intelligence[0].verdict == "phishing"


def test_ml_phishtank_disagreement_is_explicit() -> None:
    assessment = CorrelationEngine().assess(URL, ml("legitimate", 0.20), [phishtank()])
    assert assessment.correlation.status == "ml_ti_disagreement"
    assert assessment.correlation.disagreement is True
    assert assessment.ml.prediction == "legitimate"
    assert assessment.threat_intelligence[0].verdict == "phishing"


def test_urlhaus_hit_is_malware_evidence_not_phishing_confirmation() -> None:
    assessment = CorrelationEngine().assess(URL, ml("legitimate", 0.20), [urlhaus_match()])
    assert assessment.correlation.status == "malware_intelligence_ml_not_phishing"
    assert assessment.correlation.disagreement is False
    assert assessment.threat_intelligence[0].verdict == "malware_distribution"
    assert "phishing" not in assessment.threat_intelligence[0].verdict


def test_ml_only_no_match_and_all_providers_unavailable() -> None:
    no_match = ThreatIntelResult(
        source="urlhaus", queried_url=URL, lookup_status=LookupStatus.NO_MATCH,
        match=False, verdict="no_urlhaus_record",
    )
    assert CorrelationEngine().assess(URL, ml("phishing", 0.91), [no_match]).correlation.status == "ml_only_no_ti_match"
    unavailable_results = [unavailable("phishtank"), unavailable("urlhaus")]
    result = CorrelationEngine().assess(URL, ml("phishing", 0.91), unavailable_results)
    assert result.correlation.status == "ml_only_ti_unavailable"
    assert len(result.threat_intelligence) == 2


def test_ti_only_and_openphish_listing_disagreement() -> None:
    listed = ThreatIntelResult(
        source="openphish", queried_url=URL, lookup_status=LookupStatus.MATCH,
        match=True, verdict="phishing", evidence_level=EvidenceLevel.LISTED,
    )
    disagreement = CorrelationEngine().assess(URL, ml("legitimate", 0.20), [listed])
    assert disagreement.correlation.disagreement is True
    ti_only = CorrelationEngine().assess(URL, None, [urlhaus_match()])
    assert ti_only.correlation.status == "ti_only_malware_distribution"


def test_shap_metadata_is_preserved_with_ml_evidence() -> None:
    explanation = {"method": "SHAP TreeExplainer", "explained_class": 1}
    assessment = CorrelationEngine().assess(
        URL,
        MLPrediction(prediction="phishing", phishing_probability=0.8, explanation_metadata=explanation),
        [],
    )
    assert assessment.ml.explanation_metadata == explanation
    assert assessment.threat_intelligence == []


def test_ml_prediction_probability_and_threshold_must_agree() -> None:
    with pytest.raises(ValueError, match="must agree"):
        MLPrediction(prediction="phishing", phishing_probability=0.20, threshold=0.5)
