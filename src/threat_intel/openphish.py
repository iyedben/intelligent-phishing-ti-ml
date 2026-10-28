"""Optional local exact-match client for OpenPhish's free Community Feed."""

from __future__ import annotations

from time import monotonic
from typing import Any

import requests

from .base import (
    ResponseTooLargeError,
    ThreatIntelProvider,
    close_response,
    invalid_url_result,
    read_limited_response,
    safe_elapsed_ms,
    utc_now,
    validate_lookup_url,
)
from .models import EvidenceLevel, LookupStatus, ThreatIntelResult


class OpenPhishProvider(ThreatIntelProvider):
    """Fetch and locally match the official 12-hour text feed (no premium API)."""

    name = "openphish"
    feed_url = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
    timeout_seconds = 8.0
    feed_ttl_seconds = 12 * 60 * 60
    # The community feed is plain URL-per-line text; 16 MiB is ample for this
    # format while keeping the maximum decoded response allocation bounded.
    max_feed_bytes = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        session: Any | None = None,
        timeout_seconds: float = timeout_seconds,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self._feed_urls: frozenset[str] | None = None
        self._feed_fetched_monotonic: float | None = None
        self._feed_fetched_at = None

    def check_url(self, url: str) -> ThreatIntelResult:
        invalid = validate_lookup_url(url)
        if invalid:
            return invalid_url_result(self.name, url, invalid)
        started = monotonic()
        response = None
        try:
            if self._feed_is_fresh():
                feed_urls = self._feed_urls or frozenset()
                fetched_at = self._feed_fetched_at
                cache_hit = True
            else:
                response = self.session.get(
                    self.feed_url,
                    headers={"User-Agent": "IPDTI-ML/1.0 (OpenPhish Community Feed reader)", "Accept": "text/plain"},
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
                if not 200 <= response.status_code < 300:
                    return self._failure(
                        url, f"http_error_{response.status_code}", LookupStatus.UNAVAILABLE, started
                    )
                response.raise_for_status()
                feed_text = read_limited_response(response, self.max_feed_bytes).decode("utf-8-sig")
                feed_urls = frozenset(
                    line.strip()
                    for line in feed_text.splitlines()
                    if line.strip().startswith(("http://", "https://"))
                )
                if not feed_urls:
                    return self._failure(url, "empty_or_invalid_feed", LookupStatus.ERROR, started)
                fetched_at = utc_now()
                self._feed_urls = feed_urls
                self._feed_fetched_monotonic = monotonic()
                self._feed_fetched_at = fetched_at
                cache_hit = False
        except ResponseTooLargeError:
            return self._failure(url, "oversized_feed", LookupStatus.ERROR, started)
        except requests.Timeout:
            return self._failure(url, "timeout", LookupStatus.UNAVAILABLE, started)
        except requests.HTTPError:
            code = getattr(locals().get("response"), "status_code", None)
            error = f"http_error_{code or 'unknown'}"
            return self._failure(url, error, LookupStatus.UNAVAILABLE, started)
        except requests.RequestException:
            return self._failure(url, "request_error", LookupStatus.UNAVAILABLE, started)
        except (ValueError, TypeError):
            return self._failure(url, "invalid_feed_response", LookupStatus.ERROR, started)
        finally:
            close_response(response)

        matched = url in feed_urls
        return ThreatIntelResult(
            source=self.name,
            queried_url=url,
            lookup_status=LookupStatus.MATCH if matched else LookupStatus.NO_MATCH,
            match=matched,
            verdict="phishing" if matched else "not_listed_in_community_feed",
            evidence_level=EvidenceLevel.LISTED if matched else EvidenceLevel.NONE,
            checked_at=utc_now(),
            response_time_ms=safe_elapsed_ms(started, monotonic()),
            details={
                "feed_url": self.feed_url,
                "feed_fetched_at": fetched_at.isoformat() if fetched_at else None,
                "feed_update_interval_hours": 12,
                "cache_hit": cache_hit,
            },
        )

    def _feed_is_fresh(self) -> bool:
        return (
            self._feed_urls is not None
            and self._feed_fetched_monotonic is not None
            and monotonic() - self._feed_fetched_monotonic < self.feed_ttl_seconds
        )

    def _failure(
        self, url: str, error: str, status: LookupStatus, started: float
    ) -> ThreatIntelResult:
        return ThreatIntelResult(
            source=self.name,
            queried_url=url,
            lookup_status=status,
            match=None,
            verdict="unknown",
            checked_at=utc_now(),
            response_time_ms=safe_elapsed_ms(started, monotonic()),
            error=error,
        )
