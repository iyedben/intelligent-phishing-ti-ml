"""Factory for the API foundation; no ML or TI services are initialized."""

from fastapi import FastAPI

from src.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create the FastAPI application with its dependency-free health route."""
    application = FastAPI(
        title="IPDTI-ML API",
        version="0.1.0",
        description=(
            "Foundational API for the Intelligent Phishing Detection and Threat "
            "Intelligence project. Phase 8A currently exposes only service health."
        ),
    )
    application.include_router(health_router)
    return application
