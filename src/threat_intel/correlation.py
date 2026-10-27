"""Deterministic evidence rules that keep ML and provider verdicts distinct."""

from __future__ import annotations

from .models import (
    CorrelationAssessment,
    EvidenceLevel,
    LookupStatus,
    MLPrediction,
    ThreatAssessment,
    ThreatIntelResult,
)


class CorrelationEngine:
    """Correlate independent evidence without averaging or collapsing verdicts."""

    def assess(
        self,
        url: str,
        ml: MLPrediction | None,
        provider_results: list[ThreatIntelResult] | tuple[ThreatIntelResult, ...],
    ) -> ThreatAssessment:
        results = list(provider_results)
        confirmed_phishing = [
            result for result in results
            if result.match is True
            and result.verdict == "phishing"
            and result.evidence_level == EvidenceLevel.CONFIRMED
        ]
        listed_phishing = [
            result for result in results
            if result.match is True
            and result.verdict == "phishing"
            and result.evidence_level != EvidenceLevel.CONFIRMED
        ]
        malware_matches = [
            result for result in results
            if result.match is True and result.verdict == "malware_distribution"
        ]
        ambiguous_records = [
            result for result in results
            if result.match is True and result not in confirmed_phishing + listed_phishing + malware_matches
        ]
        unavailable = [
            result for result in results
            if result.lookup_status in {LookupStatus.UNAVAILABLE, LookupStatus.ERROR}
        ]
        available = [
            result for result in results
            if result.lookup_status not in {LookupStatus.UNAVAILABLE, LookupStatus.ERROR, LookupStatus.INVALID_INPUT}
        ]
        ml_phishing = ml is not None and ml.prediction == "phishing"
        rules: list[str] = []

        if confirmed_phishing and ml is not None and ml_phishing:
            status = "correlated_phishing_evidence"
            assessment = (
                "The ML model predicts phishing and a provider reports a verified, currently valid phishing record. "
                "These remain separate evidence sources."
            )
            level = EvidenceLevel.CONFIRMED
            rules.append("confirmed_phishtank_and_ml_phishing")
        elif confirmed_phishing and ml is not None:
            status = "ml_ti_disagreement"
            assessment = (
                "A provider reports a verified, currently valid phishing record while the ML model predicts legitimate. "
                "The contradictory evidence is preserved."
            )
            level = EvidenceLevel.CONFIRMED
            rules.append("confirmed_phishtank_conflicts_with_ml_prediction")
        elif listed_phishing and ml is not None and ml_phishing:
            status = "correlated_phishing_listings"
            assessment = (
                "The ML model predicts phishing and a phishing source lists the URL; the listing is not represented "
                "as a verified PhishTank confirmation."
            )
            level = EvidenceLevel.LISTED
            rules.append("phishing_listing_and_ml_phishing")
        elif listed_phishing and ml is not None:
            status = "ml_ti_disagreement"
            assessment = (
                "A phishing source lists the URL while the ML model predicts legitimate. "
                "The listing's evidence level and the model result are both retained."
            )
            level = EvidenceLevel.LISTED
            rules.append("phishing_listing_conflicts_with_ml_prediction")
        elif confirmed_phishing:
            status = "ti_only_confirmed_phishing"
            assessment = "A provider reports a verified, currently valid phishing record; no ML prediction was supplied."
            level = EvidenceLevel.CONFIRMED
            rules.append("confirmed_phishtank_record_without_ml")
        elif listed_phishing:
            status = "ti_only_phishing_listing"
            assessment = "A phishing source lists the URL; no ML prediction was supplied."
            level = EvidenceLevel.LISTED
            rules.append("phishing_listing_without_ml")
        elif malware_matches and ml is not None and ml_phishing:
            status = "ml_phishing_and_malware_intelligence"
            assessment = (
                "The ML model predicts phishing and URLhaus reports malware-distribution intelligence. "
                "URLhaus evidence is not treated as a phishing verdict."
            )
            level = _highest_level(malware_matches)
            rules.append("ml_phishing_plus_urlhaus_malware_distribution")
        elif malware_matches and ml is not None:
            status = "malware_intelligence_ml_not_phishing"
            assessment = (
                "URLhaus reports malware-distribution intelligence despite a non-phishing ML prediction. "
                "These signals concern different verdicts and are not collapsed into a disagreement or phishing label."
            )
            level = _highest_level(malware_matches)
            rules.append("urlhaus_malware_distribution_with_non_phishing_ml")
        elif malware_matches:
            status = "ti_only_malware_distribution"
            assessment = "URLhaus reports malware-distribution intelligence; no ML prediction was supplied."
            level = _highest_level(malware_matches)
            rules.append("urlhaus_malware_distribution_without_ml")
        elif ambiguous_records:
            status = "unconfirmed_ti_record"
            assessment = "A provider has a record, but its returned status does not support a confirmed phishing or malware verdict."
            level = EvidenceLevel.LISTED
            rules.append("provider_record_not_equivalent_to_confirmed_verdict")
        elif ml is not None and not results:
            status = "ml_only_no_ti_providers"
            assessment = "Only the ML result is available; no threat-intelligence providers were configured."
            level = EvidenceLevel.NONE
            rules.append("ml_result_without_configured_ti_providers")
        elif ml is not None and unavailable and not available:
            status = "ml_only_ti_unavailable"
            assessment = "Only the ML result is available; all configured threat-intelligence lookups failed or were unavailable."
            level = EvidenceLevel.NONE
            rules.append("all_ti_providers_unavailable")
        elif ml is not None and unavailable:
            status = "ml_only_partial_ti"
            assessment = "No configured provider returned a match; at least one lookup was unavailable, so TI coverage is partial."
            level = EvidenceLevel.NONE
            rules.append("partial_ti_with_no_returned_match")
        elif ml is not None:
            status = "ml_only_no_ti_match"
            assessment = "The ML result is available and no responding threat-intelligence provider matched the URL."
            level = EvidenceLevel.NONE
            rules.append("ml_result_without_provider_match")
        elif unavailable and not available:
            status = "ti_unavailable"
            assessment = "No ML result is available and all configured threat-intelligence lookups failed or were unavailable."
            level = EvidenceLevel.NONE
            rules.append("all_ti_providers_unavailable_without_ml")
        elif available:
            status = "ti_no_match"
            assessment = "No ML result is available and responding threat-intelligence providers returned no match."
            level = EvidenceLevel.NONE
            rules.append("provider_lookups_without_match_or_ml")
        else:
            status = "no_evidence"
            assessment = "No ML result or threat-intelligence provider result was supplied."
            level = EvidenceLevel.NONE
            rules.append("no_evidence_available")

        return ThreatAssessment(
            url=url,
            ml=ml,
            threat_intelligence=results,
            correlation=CorrelationAssessment(
                status=status,
                assessment=assessment,
                evidence_level=level,
                disagreement=status == "ml_ti_disagreement",
                rules_applied=rules,
            ),
        )


def _highest_level(results: list[ThreatIntelResult]) -> EvidenceLevel:
    """Select the strongest provider-declared qualitative evidence level."""
    if any(result.evidence_level == EvidenceLevel.CONFIRMED for result in results):
        return EvidenceLevel.CONFIRMED
    if any(result.evidence_level == EvidenceLevel.VERIFIED for result in results):
        return EvidenceLevel.VERIFIED
    if any(result.evidence_level == EvidenceLevel.LISTED for result in results):
        return EvidenceLevel.LISTED
    return EvidenceLevel.NONE
