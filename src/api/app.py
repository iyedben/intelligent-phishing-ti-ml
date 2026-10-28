"""Factory for the API; model and provider services are initialized lazily."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware import DEFAULT_MAX_REQUEST_BODY_BYTES, RequestBodyLimitMiddleware
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
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
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
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=max_request_body_bytes,
    )

    @application.exception_handler(RequestValidationError)
    async def sanitized_validation_error_handler(_, __) -> JSONResponse:
        """Return the standard FastAPI 422 shape without untrusted values."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "loc": ["body"],
                        "msg": "Request validation failed.",
                        "type": "value_error",
                    }
                ]
            },
        )

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    application.mount(
        "/static",
        StaticFiles(directory=frontend_dir, check_dir=True),
        name="static",
    )

    @application.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        """Serve the local dashboard without intercepting API or docs routes."""
        return FileResponse(frontend_dir / "index.html", media_type="text/html")

    application.include_router(health_router)
    application.include_router(analyze_router, prefix="/api/v1")
    application.state.analysis_service = analysis_service
    application.state.analysis_service_factory = service_factory or _default_analysis_service_factory
    application.state.analysis_service_lock = Lock()
    return application
