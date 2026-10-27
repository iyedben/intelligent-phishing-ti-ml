"""Validated, provenance-preserving threat-intelligence data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LookupStatus(StrEnum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    INVALID_INPUT = "invalid_input"


class EvidenceLevel(StrEnum):
    NONE = "none"
    LISTED = "listed"
    VERIFIED = "verified"
    CONFIRMED = "confirmed"


class ThreatIntelResult(BaseModel):
    """One provider's result; its verdict is never merged with model output."""

    model_config = ConfigDict(extra="forbid")

    source: str
    queried_url: str
    lookup_status: LookupStatus
    match: bool | None
    verdict: str
    evidence_level: EvidenceLevel = EvidenceLevel.NONE
    confidence: str | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MLPrediction(BaseModel):
    """Phishing model evidence; probability is specifically P(phishing)."""

    model_config = ConfigDict(extra="forbid")

    prediction: Literal["phishing", "legitimate"]
    phishing_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    explanation_metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def prediction_matches_threshold(self) -> "MLPrediction":
        expected = "phishing" if self.phishing_probability >= self.threshold else "legitimate"
        if self.prediction != expected:
            raise ValueError("prediction must agree with phishing_probability and threshold")
        return self


class CorrelationAssessment(BaseModel):
    """Rule-based summary that retains separate source evidence."""

    model_config = ConfigDict(extra="forbid")

    status: str
    assessment: str
    evidence_level: EvidenceLevel
    disagreement: bool = False
    rules_applied: list[str] = Field(default_factory=list)


class ThreatAssessment(BaseModel):
    """Combined response with independent ML, TI, and correlation sections."""

    model_config = ConfigDict(extra="forbid")

    url: str
    ml: MLPrediction | None
    threat_intelligence: list[ThreatIntelResult] = Field(default_factory=list)
    correlation: CorrelationAssessment


def now_utc() -> datetime:
    """Return timezone-aware UTC for provider timestamps."""
    return datetime.now(timezone.utc)
