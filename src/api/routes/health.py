"""Unversioned process health route."""

from fastapi import APIRouter

from src.api.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def get_health() -> HealthResponse:
    """Report that the API process can serve requests."""
    return HealthResponse(status="ok")
