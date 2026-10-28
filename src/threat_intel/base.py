"""Provider interface, exact-input validation, and optional TTL cache."""

from __future__ import annotations

from collections import OrderedDict
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha256
from threading import RLock
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from .models import LookupStatus, ThreatIntelResult


class ResponseTooLargeError(ValueError):
    """Raised when a streamed provider response exceeds its byte budget."""


def read_limited_response(response: Any, max_bytes: int, *, chunk_size: int = 64 * 1024) -> bytes:
    """Read a streamed HTTP response while enforcing a decoded-byte limit.

    A declared Content-Length larger than the limit is rejected early, but the
    cumulative streamed byte count remains authoritative when the header is
    absent, malformed, understated, or describes a compressed representation.
    The caller owns and must close the response on every path.
    """
    if max_bytes <= 0 or chunk_size <= 0:
        raise ValueError("response byte limits must be positive")
    header = getattr(response, "headers", {}).get("Content-Length")
    try:
        declared_length = int(header) if header is not None else None
    except (TypeError, ValueError):
        declared_length = None
    if declared_length is not None and declared_length > max_bytes:
        raise ResponseTooLargeError("provider response exceeds configured byte limit")

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            raise TypeError("streamed response chunks must be bytes")
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError("provider response exceeds configured byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def close_response(response: Any | None) -> None:
    """Close a streamed response while preventing close errors masking results."""
    if response is None:
        return
    try:
        response.close()
    except Exception:
        pass


class ThreatIntelProvider(ABC):
    """A provider performs an intelligence lookup and never visits the IOC URL."""

    name: str
    cache_ttl_seconds: float = 300.0

    @abstractmethod
    def check_url(self, url: str) -> ThreatIntelResult:
        """Look up the exact submitted URL and return normalized evidence."""


class MemoryTTLCache:
    """Bounded process-local TTL cache keyed by provider and exact URL bytes.

    Successful reads refresh LRU order. Expired entries are swept on every
    get/set, and the least-recently-used item is evicted at capacity.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, ThreatIntelResult]] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def key(provider: str, url: str) -> str:
        digest = sha256(url.encode("utf-8")).hexdigest()
        return f"{provider}:{digest}"

    def get(self, key: str) -> ThreatIntelResult | None:
        with self._lock:
            self._remove_expired(monotonic())
            stored = self._items.get(key)
            if stored is None:
                return None
            expiry, value = stored
            if monotonic() >= expiry:  # Guard a clock advance after the sweep.
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value.model_copy(deep=True)

    def set(self, key: str, value: ThreatIntelResult, ttl_seconds: float) -> None:
        with self._lock:
            now = monotonic()
            self._remove_expired(now)
            if ttl_seconds <= 0:
                self._items.pop(key, None)
                return
            self._items[key] = (now + ttl_seconds, value.model_copy(deep=True))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            self._remove_expired(monotonic())
            return len(self._items)

    def _remove_expired(self, now: float) -> None:
        expired = [key for key, (expiry, _) in self._items.items() if now >= expiry]
        for key in expired:
            del self._items[key]


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
