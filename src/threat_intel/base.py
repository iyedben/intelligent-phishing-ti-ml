"""Provider interface, exact-input validation, and optional TTL cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha256
from threading import RLock
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from .models import LookupStatus, ThreatIntelResult


class ThreatIntelProvider(ABC):
    """A provider performs an intelligence lookup and never visits the IOC URL."""

    name: str
    cache_ttl_seconds: float = 300.0

    @abstractmethod
    def check_url(self, url: str) -> ThreatIntelResult:
        """Look up the exact submitted URL and return normalized evidence."""


class MemoryTTLCache:
    """Optional process-local cache keyed by provider and exact URL bytes."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[float, ThreatIntelResult]] = {}
        self._lock = RLock()

    @staticmethod
    def key(provider: str, url: str) -> str:
        digest = sha256(url.encode("utf-8")).hexdigest()
        return f"{provider}:{digest}"

    def get(self, key: str) -> ThreatIntelResult | None:
        with self._lock:
            stored = self._items.get(key)
            if stored is None:
                return None
            expiry, value = stored
            if monotonic() >= expiry:
                del self._items[key]
                return None
            return value.model_copy(deep=True)

    def set(self, key: str, value: ThreatIntelResult, ttl_seconds: float) -> None:
        with self._lock:
            self._items[key] = (monotonic() + ttl_seconds, value.model_copy(deep=True))


def validate_lookup_url(url: str) -> str | None:
    """Return a safe error code if URL is invalid; otherwise return None.

    The original string is preserved. Leading/trailing or embedded whitespace
    is rejected instead of silently trimmed, and fragments remain part of the
    exact value submitted to the provider.
    """
    if not isinstance(url, str):
        return "url_must_be_string"
    if not url or url != url.strip():
        return "url_empty_or_surrounding_whitespace"
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        return "url_contains_whitespace_or_control"
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return "malformed_url"
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc or not hostname:
        return "absolute_http_url_required"
    return None


def invalid_url_result(source: str, url: Any, reason: str) -> ThreatIntelResult:
    return ThreatIntelResult(
        source=source,
        queried_url=url if isinstance(url, str) else "",
        lookup_status=LookupStatus.INVALID_INPUT,
        match=None,
        verdict="unknown",
        error=reason,
    )


def cached_result(cache: MemoryTTLCache | None, provider: str, url: str) -> ThreatIntelResult | None:
    return cache.get(cache.key(provider, url)) if cache is not None else None


def cache_success(
    cache: MemoryTTLCache | None,
    provider: str,
    url: str,
    result: ThreatIntelResult,
    ttl_seconds: float,
) -> None:
    if cache is not None and result.lookup_status in {LookupStatus.MATCH, LookupStatus.NO_MATCH}:
        cache.set(cache.key(provider, url), result, ttl_seconds)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_elapsed_ms(started: float, ended: float) -> float:
    return round(max(0.0, (ended - started) * 1000), 3)


def json_safe_provider_details(values: dict[str, Any]) -> dict[str, Any]:
    """Keep provider metadata JSON-safe and exclude arbitrary objects."""
    def clean(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [clean(item) for item in value[:100]]
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in list(value.items())[:100]}
        return str(value)[:500]

    return {str(key): clean(value) for key, value in values.items()}
