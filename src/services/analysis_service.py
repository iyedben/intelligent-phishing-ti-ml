"""Orchestrate the existing URL inference, SHAP, TI, and correlation layers."""

from __future__ import annotations

from typing import Any, Sequence

from src.api.schemas.analyze import (
    AnalyzeResponse,
    ExplanationResponse,
    MLResultResponse,
    SHAPContributionResponse,
)
from src.threat_intel import (
    CorrelationEngine,
    MemoryTTLCache,
    MLPrediction,
    OpenPhishProvider,
    PhishTankProvider,
    ThreatIntelProvider,
    ThreatIntelResult,
    URLhausProvider,
)
from src.threat_intel.models import LookupStatus
from src.threat_intel.normalizer import normalize_lookup_url
from src.xai import URLShapExplainer


class AnalysisService:
    """Use injected, reusable components to analyze one URL string.

    The existing ``URLShapExplainer.explain_url`` is the canonical inference
    entry point: it invokes the feature extractor, saved model, and Tree SHAP
    once and returns the established output semantics. Provider lookups and
    correlation are then delegated to their Phase 7 interfaces.
    """

    def __init__(
        self,
        *,
        explainer: Any,
        providers: Sequence[ThreatIntelProvider],
        correlation_engine: Any,
    ) -> None:
        self._explainer = explainer
        self._providers = tuple(providers)
        self._correlation_engine = correlation_engine

    def analyze(self, url: str) -> AnalyzeResponse:
        """Validate the exact URL, run existing components, and build response."""
        submitted_url = normalize_lookup_url(url)
        local_result = self._explainer.explain_url(submitted_url)

        explanation_data = local_result["explanation"]
        top_contributions = [
            SHAPContributionResponse.model_validate(item)
            for item in explanation_data["contributions"][:5]
        ]
        explanation = ExplanationResponse(
            method=explanation_data["method"],
            explained_class=explanation_data["explained_class"],
            model_output=explanation_data["model_output"],
            feature_perturbation=explanation_data["feature_perturbation"],
            shap_version=explanation_data["shap_version"],
            base_value=explanation_data["base_value"],
            shap_sum=explanation_data["shap_sum"],
            reconstructed_output=explanation_data["reconstructed_output"],
            model_phishing_probability=explanation_data["model_phishing_probability"],
            additivity_absolute_error=explanation_data["additivity_absolute_error"],
            top_contributions=top_contributions,
        )
        ml = MLPrediction(
            prediction=local_result["prediction"],
            phishing_probability=local_result["phishing_probability"],
            threshold=local_result["threshold"],
            explanation_metadata={
                "method": explanation.method,
                "explained_class": explanation.explained_class,
                "top_contributions": [item.model_dump(mode="json") for item in top_contributions],
            },
        )
        ml_response = MLResultResponse(
            prediction=local_result["prediction"],
            phishing_probability=local_result["phishing_probability"],
            threshold=local_result["threshold"],
            positive_class=local_result["positive_class"],
        )

        provider_results: list[ThreatIntelResult] = []
        for provider in self._providers:
            try:
                provider_results.append(provider.check_url(submitted_url))
            except Exception:
                # Phase 7 adapters normally normalize failures themselves. This
                # final guard preserves partial analysis for unexpected adapter
                # failures without returning exception text or credentials.
                provider_results.append(
                    ThreatIntelResult(
                        source=provider.name,
                        queried_url=submitted_url,
                        lookup_status=LookupStatus.ERROR,
                        match=None,
                        verdict="unknown",
                        error="provider_internal_error",
                    )
                )

        correlated = self._correlation_engine.assess(submitted_url, ml, provider_results)
        return AnalyzeResponse(
            url=submitted_url,
            ml=ml_response,
            explanation=explanation,
            threat_intelligence=correlated.threat_intelligence,
            correlation=correlated.correlation,
        )


def build_default_analysis_service() -> AnalysisService:
    """Load the existing model once and construct long-lived provider clients."""
    explainer = URLShapExplainer.from_artifact()
    cache = MemoryTTLCache()
    providers: tuple[ThreatIntelProvider, ...] = (
        PhishTankProvider(cache=cache),
        URLhausProvider(cache=cache),
        OpenPhishProvider(),
    )
    return AnalysisService(
        explainer=explainer,
        providers=providers,
        correlation_engine=CorrelationEngine(),
    )
