"""Focused offline tests for bounded API and provider resource use."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.middleware import RequestBodyLimitMiddleware
from src.threat_intel.base import MemoryTTLCache
from src.threat_intel.models import LookupStatus, ThreatIntelResult
from src.threat_intel.openphish import OpenPhishProvider
from src.threat_intel.phishtank import PhishTankProvider
from src.threat_intel.urlhaus import URLhausProvider


SYNTHETIC_MARKER = "AUDIT-SYNTHETIC-INPUT-MARKER"


def make_response(body: bytes, *, content_length: str | None = None) -> Mock:
    result = Mock(status_code=200)
    result.headers = {} if content_length is None else {"Content-Length": content_length}
    result.iter_content.return_value = iter([body[:4], b"", body[4:]])
    return result


def cache_result(url: str) -> ThreatIntelResult:
    return ThreatIntelResult(
        source="urlhaus",
        queried_url=url,
        lookup_status=LookupStatus.NO_MATCH,
        match=False,
        verdict="no_urlhaus_record",
    )


def test_request_body_over_limit_returns_413_before_service_call() -> None:
    service = Mock()
    app = create_app(analysis_service=service, max_request_body_bytes=128)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            content=b"{" + b" " * 512 + b"}",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the configured size limit."}
    service.analyze.assert_not_called()


def test_validation_error_does_not_reflect_submitted_url_or_userinfo() -> None:
    submitted = f"https://user:{SYNTHETIC_MARKER}@example.invalid/a b"
    with TestClient(create_app(analysis_service=object())) as client:
        response = client.post("/api/v1/analyze", json={"url": submitted})
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {"loc": ["body"], "msg": "Request validation failed.", "type": "value_error"}
        ]
    }
    assert SYNTHETIC_MARKER not in response.text
    assert submitted not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {"url": "https://example.invalid/" + "x" * 4096 + SYNTHETIC_MARKER},
        {"url": "https://example.invalid/", "unexpected": SYNTHETIC_MARKER},
        {"url": 7, "extra": SYNTHETIC_MARKER},
    ],
)
def test_all_request_validation_errors_are_sanitized(payload: dict[str, object]) -> None:
    with TestClient(create_app(analysis_service=object())) as client:
        response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
    assert SYNTHETIC_MARKER not in response.text
    assert isinstance(response.json()["detail"], list)
    assert all(set(error) == {"loc", "msg", "type"} for error in response.json()["detail"])


def test_sanitized_422_matches_generated_openapi_schema() -> None:
    app = create_app(analysis_service=object())
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        response = client.post("/api/v1/analyze", json={"url": "not-a-url"})

    documented = schema["paths"]["/api/v1/analyze"]["post"]["responses"]["422"]
    error_ref = documented["content"]["application/json"]["schema"]["$ref"]
    assert error_ref == "#/components/schemas/HTTPValidationError"
    error_schema = schema["components"]["schemas"]["HTTPValidationError"]
    validation_ref = error_schema["properties"]["detail"]["items"]["$ref"]
    assert validation_ref == "#/components/schemas/ValidationError"

    body = response.json()
    assert response.status_code == 422
    assert set(body) == set(error_schema["properties"])
    assert len(body["detail"]) == 1
    error = body["detail"][0]
    validation_schema = schema["components"]["schemas"]["ValidationError"]
    assert set(error) == set(validation_schema["required"])
    assert error["loc"] == ["body"]


def test_body_limit_checks_chunked_body_even_with_misleading_length() -> None:
    sent: list[dict[str, object]] = []
    incoming = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"67890", "more_body": True},
            {"type": "http.request", "body": b"x", "more_body": False},
        ]
    )

    async def downstream(scope, receive, send):
        raise AssertionError("oversized content must not reach the application")

    async def receive():
        return next(incoming)

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
    scope = {"type": "http", "headers": [(b"content-length", b"1")]}
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413
    assert b"".join(item.get("body", b"") for item in sent) == (
        b'{"detail":"Request body exceeds the configured size limit."}'
    )


def test_cache_lru_eviction_and_capacity() -> None:
    cache = MemoryTTLCache(max_entries=2)
    cache.set("provider:a", cache_result("https://a.invalid/"), 300)
    cache.set("provider:b", cache_result("https://b.invalid/"), 300)
    assert cache.get("provider:a") is not None  # a becomes most recently used
    cache.set("provider:c", cache_result("https://c.invalid/"), 300)
    assert len(cache) == 2
    assert cache.get("provider:a") is not None
    assert cache.get("provider:b") is None
    assert cache.get("provider:c") is not None


def test_cache_sweeps_expired_entries_on_unrelated_set(monkeypatch: pytest.MonkeyPatch) -> None:
    current = [100.0]
    monkeypatch.setattr("src.threat_intel.base.monotonic", lambda: current[0])
    cache = MemoryTTLCache(max_entries=3)
    cache.set("old", cache_result("https://old.invalid/"), 2)
    current[0] = 103.0
    cache.set("new", cache_result("https://new.invalid/"), 30)
    assert len(cache) == 1
    assert cache.get("old") is None
    assert cache.get("new") is not None


def test_cache_is_safe_under_concurrent_get_set() -> None:
    cache = MemoryTTLCache(max_entries=32)

    def operation(index: int) -> None:
        key = f"urlhaus:{index % 64}"
        cache.set(key, cache_result(f"https://{index % 64}.invalid/"), 60)
        cache.get(key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(operation, range(400)))
    assert len(cache) <= 32


@pytest.mark.parametrize("provider_name", ["phishtank", "urlhaus", "openphish"])
@pytest.mark.parametrize("header_mode", ["oversized", "missing", "understated"])
def test_provider_rejects_oversized_stream_and_closes_response(
    provider_name: str,
    header_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"x" * 12
    header = {"oversized": "99", "missing": None, "understated": "1"}[header_mode]
    mocked_response = make_response(body, content_length=header)
    session = Mock()
    if provider_name == "phishtank":
        monkeypatch.setattr(PhishTankProvider, "max_response_bytes", 8)
        session.post.return_value = mocked_response
        provider = PhishTankProvider(app_key="test-only", session=session)
        result = provider.check_url("https://example.invalid/")
        call = session.post.call_args
    elif provider_name == "urlhaus":
        monkeypatch.setattr(URLhausProvider, "max_response_bytes", 8)
        session.post.return_value = mocked_response
        provider = URLhausProvider(auth_key="test-only", session=session)
        result = provider.check_url("https://example.invalid/")
        call = session.post.call_args
    else:
        monkeypatch.setattr(OpenPhishProvider, "max_feed_bytes", 8)
        session.get.return_value = mocked_response
        provider = OpenPhishProvider(session=session)
        result = provider.check_url("https://example.invalid/")
        call = session.get.call_args

    assert call.kwargs["stream"] is True
    assert call.kwargs["allow_redirects"] is False
    assert result.lookup_status == LookupStatus.ERROR
    assert result.error == ("oversized_feed" if provider_name == "openphish" else "response_too_large")
    mocked_response.close.assert_called_once()
    if header_mode == "oversized":
        mocked_response.iter_content.assert_not_called()
    else:
        mocked_response.iter_content.assert_called_once()


def test_provider_timeout_semantics_remain_distinct() -> None:
    session = Mock()
    session.post.side_effect = __import__("requests").Timeout()
    result = URLhausProvider(auth_key="test-only", session=session).check_url("https://example.invalid/")
    assert result.lookup_status == LookupStatus.UNAVAILABLE
    assert result.error == "timeout"

