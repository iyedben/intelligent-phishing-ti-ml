"""Threat-intelligence lookups and evidence correlation."""

from .correlation import CorrelationEngine
from .base import MemoryTTLCache, ThreatIntelProvider
from .models import (
    CorrelationAssessment,
    EvidenceLevel,
    LookupStatus,
    MLPrediction,
    ThreatAssessment,
    ThreatIntelResult,
)
from .normalizer import normalize_lookup_url
from .phishtank import PhishTankProvider
from .urlhaus import URLhausProvider
from .openphish import OpenPhishProvider

__all__ = [
    "CorrelationAssessment",
    "CorrelationEngine",
    "EvidenceLevel",
    "LookupStatus",
    "MemoryTTLCache",
    "MLPrediction",
    "OpenPhishProvider",
    "PhishTankProvider",
    "ThreatAssessment",
    "ThreatIntelResult",
    "ThreatIntelProvider",
    "URLhausProvider",
    "normalize_lookup_url",
]
