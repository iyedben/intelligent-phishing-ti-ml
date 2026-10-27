"""Foundational FastAPI behavior; tests use in-process HTTPX only."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import requests

from src.api import create_app
from src.api.schemas import HealthResponse


def test_application_factory_creates_fastapi_without_external_services() -> None:
    application = create_app()
    assert isinstance(application, FastAPI)
    assert application.title == "IPDTI-ML API"
    assert application.version == "0.1.0"
    assert not hasattr(application.state, "model")
    assert not hasattr(application.state, "threat_intel_providers")


def test_health_endpoint_response_and_schema(monkeypatch) -> None:
    def deny_request(*args, **kwargs):
        raise AssertionError("Health endpoint attempted an external requests call")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_request)
    application = create_app()
    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert HealthResponse.model_validate(response.json()).status == "ok"
    openapi_schema = application.openapi()
    response_schema = openapi_schema["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"] == "#/components/schemas/HealthResponse"
