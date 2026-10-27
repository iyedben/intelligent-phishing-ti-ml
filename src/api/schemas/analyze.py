"""Request and response models for URL analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.threat_intel.models import CorrelationAssessment, ThreatIntelResult
from src.threat_intel.normalizer import normalize_lookup_url


class AnalyzeRequest(BaseModel):
    """An exact HTTP(S) URL string to analyze; it is never normalized here."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="Absolute HTTP(S) URL to analyze")

    @field_validator("url")
    @classmethod
    def validate_supported_url(cls, value: str) -> str:
        try:
            return normalize_lookup_url(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Enter a valid absolute HTTP(S) URL without whitespace.") from exc


class MLResultResponse(BaseModel):
    """Model output; probability is P(phishing), with class 1 positive."""

    model_config = ConfigDict(extra="forbid")

    prediction: Literal["phishing", "legitimate"]
    phishing_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    positive_class: Literal[1]


class SHAPContributionResponse(BaseModel):
    """One feature's existing Phase 6 SHAP contribution."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: int
    shap_value: float
    direction: Literal["phishing", "legitimate", "neutral"]
    statement: str


class ExplanationResponse(BaseModel):
    """Selected metadata and the strongest per-feature contributions."""

    model_config = ConfigDict(extra="forbid")

    method: str
    explained_class: Literal[1]
    model_output: str
    feature_perturbation: str
    shap_version: str
    base_value: float
    shap_sum: float
    reconstructed_output: float
    model_phishing_probability: float
    additivity_absolute_error: float
    top_contributions: list[SHAPContributionResponse]


class AnalyzeResponse(BaseModel):
    """Combined response retaining the ML, TI, and correlation sections."""

    model_config = ConfigDict(extra="forbid")

    url: str
    ml: MLResultResponse
    explanation: ExplanationResponse
    threat_intelligence: list[ThreatIntelResult]
    correlation: CorrelationAssessment
