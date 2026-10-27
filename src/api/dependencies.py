"""Per-application lazy service dependency."""

from __future__ import annotations

from threading import Lock
from typing import Any

from fastapi import HTTPException, Request

def get_analysis_service(request: Request) -> Any:
    """Return the app's shared service, creating it once on first analysis."""
    application = request.app
    service = application.state.analysis_service
    if service is not None:
        return service

    with application.state.analysis_service_lock:
        service = application.state.analysis_service
        if service is None:
            try:
                service = application.state.analysis_service_factory()
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail="Analysis could not be completed.",
                ) from None
            application.state.analysis_service = service
    return service
