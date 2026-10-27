"""Client for URLhaus's authenticated Community API URL lookup."""

from __future__ import annotations

from time import monotonic
from typing import Any

import requests

from .base import (
    MemoryTTLCache,
    ThreatIntelProvider,
    cache_success,
    cached_result,
    invalid_url_result,
    json_safe_provider_details,
    safe_elapsed_ms,
    utc_now,
    validate_lookup_url,
)
from .config import get_setting
from .models import EvidenceLevel, LookupStatus, ThreatIntelResult


class URLhausProvider(ThreatIntelProvider):
    """URLhaus exact-URL lookup; a hit means malware-distribution intelligence."""

    name = "urlhaus"
    endpoint = "https://urlhaus-api.abuse.ch/v1/url/"
    timeout_seconds = 8.0
    cache_ttl_seconds = 300.0
    user_agent = "IPDTI-ML/1.0 (URLhaus URL lookup)"

    def __init__(
        self,
        auth_key: str | None = None,
        *,
        session: Any | None = None,
        timeout_seconds: float = timeout_seconds,
        cache: MemoryTTLCache | None = None,
    ) -> None:
        self.auth_key = auth_key if auth_key is not None else get_setting("URLHAUS_AUTH_KEY")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.cache = cache

    def check_url(self, url: str) -> ThreatIntelResult:
        invalid = validate_lookup_url(url)
        if invalid:
            return invalid_url_result(self.name, url, invalid)
        if not self.auth_key:
            return self._failure(url, "missing_api_key", LookupStatus.UNAVAILABLE, utc_now(), None)
        cached = cached_result(self.cache, self.name, url)
        if cached:
            cached.details["cache_hit"] = True
            cached.response_time_ms = 0.0
            return cached

        started = monotonic()
        checked_at = utc_now()
        try:
            response = self.session.post(
                self.endpoint,
                data={"url": url},
                headers={"Auth-Key": self.auth_key, "User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
            if not 200 <= response.status_code < 300:
                code = response.status_code
                error = f"http_{code}_rate_limited" if code == 509 else f"http_error_{code}"
                return self._failure(url, error, LookupStatus.UNAVAILABLE, checked_at, started)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout:
            return self._failure(url, "timeout", LookupStatus.UNAVAILABLE, checked_at, started)
        except requests.HTTPError:
            code = getattr(locals().get("response"), "status_code", None)
            error = f"http_{code}_rate_limited" if code == 509 else f"http_error_{code or 'unknown'}"
            return self._failure(url, error, LookupStatus.UNAVAILABLE, checked_at, started)
        except requests.RequestException:
            return self._failure(url, "request_error", LookupStatus.UNAVAILABLE, checked_at, started)
        except (ValueError, TypeError):
            return self._failure(url, "invalid_json", LookupStatus.ERROR, checked_at, started)

        if not isinstance(payload, dict) or not isinstance(payload.get("query_status"), str):
            return self._failure(url, "invalid_response", LookupStatus.ERROR, checked_at, started)
        query_status = payload["query_status"]
        elapsed = safe_elapsed_ms(started, monotonic())
        if query_status == "no_results":
            result = ThreatIntelResult(
                source=self.name,
                queried_url=url,
                lookup_status=LookupStatus.NO_MATCH,
                match=False,
                verdict="no_urlhaus_record",
                checked_at=checked_at,
                response_time_ms=elapsed,
                details={"query_status": query_status},
            )
        elif (
            query_status == "ok"
            and payload.get("id") not in (None, "")
            and isinstance(payload.get("id"), (str, int))
            and payload.get("url_status") in {"online", "offline", "unknown"}
            and payload.get("threat") == "malware_download"
        ):
            details = _select_details(payload)
            status = payload["url_status"]
            result = ThreatIntelResult(
                source=self.name,
                queried_url=url,
                lookup_status=LookupStatus.MATCH,
                match=True,
                verdict="malware_distribution",
                evidence_level=EvidenceLevel.LISTED,
                checked_at=checked_at,
                response_time_ms=elapsed,
                details=details,
            )
            if status == "online":
                result.evidence_level = EvidenceLevel.CONFIRMED
        elif query_status in {"invalid_url", "http_get_expected", "http_post_expected"}:
            return self._failure(url, f"provider_query_status_{query_status}", LookupStatus.ERROR, checked_at, started)
        else:
            return self._failure(url, "invalid_response", LookupStatus.ERROR, checked_at, started)
        cache_success(self.cache, self.name, url, result, self.cache_ttl_seconds)
        return result

    def _failure(
        self,
        url: str,
        error: str,
        status: LookupStatus,
        checked_at: Any,
        started: float | None,
    ) -> ThreatIntelResult:
        elapsed = 0.0 if started is None else safe_elapsed_ms(started, monotonic())
        return ThreatIntelResult(
            source=self.name,
            queried_url=url,
            lookup_status=status,
            match=None,
            verdict="unknown",
            checked_at=checked_at,
            response_time_ms=elapsed,
            error=error,
        )


def _select_details(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "query_status", "id", "urlhaus_reference", "url_status", "host", "date_added",
        "last_online", "threat", "reporter", "larted", "takedown_time_seconds", "tags", "blacklists",
    )
    details = {key: payload[key] for key in fields if key in payload}
    payloads = payload.get("payloads")
    if isinstance(payloads, list):
        payload_fields = (
            "firstseen", "filename", "file_type", "response_size", "response_md5", "response_sha256",
            "urlhaus_download", "signature", "imphash", "ssdeep", "tlsh", "magika",
        )
        details["payloads"] = [
            {key: payload[key] for key in payload_fields if key in payload}
            for payload in payloads[:20]
            if isinstance(payload, dict)
        ]
    return json_safe_provider_details(details)
