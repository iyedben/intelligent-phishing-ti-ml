"""Client for the official PhishTank single-URL check API."""

from __future__ import annotations

import os
from time import monotonic
from typing import Any

import requests

from .base import (
    MemoryTTLCache,
    ThreatIntelProvider,
    cache_success,
    cached_result,
    invalid_url_result,
    safe_elapsed_ms,
    utc_now,
    validate_lookup_url,
)
from .models import EvidenceLevel, LookupStatus, ThreatIntelResult


class PhishTankProvider(ThreatIntelProvider):
    """PhishTank API client; unauthenticated checks are allowed at lower rate."""

    name = "phishtank"
    endpoint = "https://checkurl.phishtank.com/checkurl/"
    timeout_seconds = 8.0
    cache_ttl_seconds = 300.0
    user_agent = "IPDTI-ML/1.0 (+https://github.com/; URL threat-intelligence lookup)"

    def __init__(
        self,
        app_key: str | None = None,
        *,
        session: Any | None = None,
        timeout_seconds: float = timeout_seconds,
        cache: MemoryTTLCache | None = None,
    ) -> None:
        self.app_key = app_key if app_key is not None else os.getenv("PHISHTANK_APP_KEY", "")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.cache = cache

    def check_url(self, url: str) -> ThreatIntelResult:
        invalid = validate_lookup_url(url)
        if invalid:
            return invalid_url_result(self.name, url, invalid)
        cached = cached_result(self.cache, self.name, url)
        if cached:
            cached.details["cache_hit"] = True
            cached.response_time_ms = 0.0
            return cached

        started = monotonic()
        checked_at = utc_now()
        form: dict[str, str] = {"url": url, "format": "json"}
        if self.app_key:
            form["app_key"] = self.app_key
        try:
            response = self.session.post(
                self.endpoint,
                data=form,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
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
        except requests.HTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            code = code or getattr(locals().get("response"), "status_code", None)
            error = f"http_{code}_rate_limited" if code == 509 else f"http_error_{code or 'unknown'}"
            return self._failure(url, error, LookupStatus.UNAVAILABLE, checked_at, started)
        except requests.RequestException:
            return self._failure(url, "request_error", LookupStatus.UNAVAILABLE, checked_at, started)
        except (ValueError, TypeError):
            return self._failure(url, "invalid_json", LookupStatus.ERROR, checked_at, started)

        record = _extract_record(payload)
        if record is None:
            return self._failure(url, "invalid_response", LookupStatus.ERROR, checked_at, started)
        in_database = _parse_bool(record.get("in_database"))
        if in_database is None:
            return self._failure(url, "invalid_response", LookupStatus.ERROR, checked_at, started)
        elapsed = safe_elapsed_ms(started, monotonic())
        details = {
            key: record.get(key)
            for key in (
                "phish_id",
                "phish_detail_page",
                "verified",
                "verified_at",
                "verification_time",
                "valid",
                "submitted_at",
                "submission_time",
                "target",
            )
            if key in record
        }
        if not in_database:
            result = ThreatIntelResult(
                source=self.name,
                queried_url=url,
                lookup_status=LookupStatus.NO_MATCH,
                match=False,
                verdict="no_phishtank_record",
                checked_at=checked_at,
                response_time_ms=elapsed,
                details=details,
            )
        else:
            verified = _parse_bool(record.get("verified"))
            valid = _parse_bool(record.get("valid"))
            confirmed = verified is True and valid is True
            if confirmed:
                verdict = "phishing"
                evidence = EvidenceLevel.CONFIRMED
            elif verified is False:
                verdict = "unverified_phishing_submission"
                evidence = EvidenceLevel.LISTED
            elif valid is False:
                verdict = "not_currently_validated_as_phishing"
                evidence = EvidenceLevel.LISTED
            else:
                verdict = "phishing_record_status_unknown"
                evidence = EvidenceLevel.LISTED
            result = ThreatIntelResult(
                source=self.name,
                queried_url=url,
                lookup_status=LookupStatus.MATCH,
                match=True,
                verdict=verdict,
                evidence_level=evidence,
                checked_at=checked_at,
                response_time_ms=elapsed,
                details=details,
            )
        cache_success(self.cache, self.name, url, result, self.cache_ttl_seconds)
        return result

    def _failure(
        self,
        url: str,
        error: str,
        status: LookupStatus,
        checked_at: Any,
        started: float,
    ) -> ThreatIntelResult:
        return ThreatIntelResult(
            source=self.name,
            queried_url=url,
            lookup_status=status,
            match=None,
            verdict="unknown",
            checked_at=checked_at,
            response_time_ms=safe_elapsed_ms(started, monotonic()),
            error=error,
        )


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.casefold().strip()
        if normalized in {"y", "yes", "true", "1"}:
            return True
        if normalized in {"n", "no", "false", "0"}:
            return False
    return None


def _extract_record(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if isinstance(results, dict):
        if "in_database" in results:
            return results
        # Accommodate the API's XML-style indexed wrapper if a gateway emits it.
        indexed = [item for item in results.values() if isinstance(item, dict) and "in_database" in item]
        return indexed[0] if len(indexed) == 1 else None
    if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    return None
