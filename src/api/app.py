"""Factory for the API; model and provider services are initialized lazily."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import FastAPI

from src.api.routes.analyze import router as analyze_router
from src.api.routes.health import router as health_router

if TYPE_CHECKING:
    from src.services.analysis_service import AnalysisService


def _default_analysis_service_factory():
    """Import and initialize ML/TI components only on the first analysis call."""
    from src.services.analysis_service import build_default_analysis_service

    return build_default_analysis_service()


def create_app(
    analysis_service: AnalysisService | None = None,
    *,
    service_factory: Callable[[], AnalysisService] | None = None,
) -> FastAPI:
    """Create a testable application without importing/initializing ML or TI."""
    application = FastAPI(
        title="IPDTI-ML API",
        version="0.1.0",
        description=(
            "URL-only phishing analysis with model explanations and correlated "
            "threat-intelligence evidence. Submitted destinations are not fetched."
        ),
    )
    application.include_router(health_router)
    application.include_router(analyze_router, prefix="/api/v1")
    application.state.analysis_service = analysis_service
    application.state.analysis_service_factory = service_factory or _default_analysis_service_factory
    application.state.analysis_service_lock = Lock()
    return application
