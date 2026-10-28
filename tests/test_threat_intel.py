"""Mocked provider tests; this module never calls live TI services."""

from __future__ import annotations

import json
import logging
from unittest.mock import Mock

import pytest
import requests

from src.threat_intel.models import LookupStatus
from src.threat_intel.config import get_setting
from src.threat_intel.openphish import OpenPhishProvider
from src.threat_intel.phishtank import PhishTankProvider
from src.threat_intel.urlhaus import URLhausProvider


def response(payload: object, *, status_code: int = 200, text: str | None = None) -> Mock:
    result = Mock()
    result.status_code = status_code
    result.json.return_value = payload
    result.text = text
    body = text.encode("utf-8") if text is not None else json.dumps(payload).encode("utf-8")
    result.headers = {"Content-Length": str(len(body))}
    result.iter_content.return_value = iter([body])
    if status_code >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    else:
        result.raise_for_status.return_value = None
    return result


def test_phishtank_verified_positive_is_confirmed_phishing() -> None:
    session = Mock()
    session.post.return_value = response({"results": {
        "url": "https://submitted.example/path?q=a&b=c#frag",
        "in_database": True,
        "verified": "y",
        "valid": "y",
        "phish_id": 73,
        "phish_detail_page": "https://phishtank.example/73",
        "submission_time": "2026-01-01T00:00:00Z",
        "verification_time": "2026-01-01T00:01:00Z",
        "target": "Example",
    }})
    provider = PhishTankProvider(app_key="sample-secret", session=session)
    url = "https://submitted.example/path?q=a&b=c#frag"
    result = provider.check_url(url)
    assert result.lookup_status == LookupStatus.MATCH
    assert result.match is True
    assert result.verdict == "phishing"
    assert result.evidence_level == "confirmed"
    assert result.details["phish_id"] == 73
    assert session.post.call_args.args[0] == provider.endpoint
    assert session.post.call_args.kwargs["data"]["url"] == url
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert session.post.call_args.kwargs["stream"] is True
    session.post.return_value.close.assert_called_once()
    assert session.post.call_args.kwargs["timeout"] == 8.0
    assert session.post.call_args.kwargs["data"]["app_key"] == "sample-secret"


def test_phishtank_database_membership_is_not_confirmation() -> None:
    session = Mock()
    session.post.return_value = response({"results": {
        "in_database": True, "verified": "n", "valid": "n", "phish_id": "14"
    }})
    result = PhishTankProvider(app_key="", session=session).check_url("https://example.test/")
    assert result.match is True
    assert result.verdict == "unverified_phishing_submission"
    assert result.evidence_level == "listed"


def test_phishtank_negative_and_missing_key_attempt() -> None:
    session = Mock()
    session.post.return_value = response({"results": {"in_database": False}})
    result = PhishTankProvider(app_key="", session=session).check_url("https://example.test/")
    assert result.lookup_status == LookupStatus.NO_MATCH
    assert result.match is False
    assert "app_key" not in session.post.call_args.kwargs["data"]


def test_urlhaus_positive_preserves_malware_verdict_and_payload_metadata() -> None:
    session = Mock()
    session.post.return_value = response({
        "query_status": "ok", "id": "987", "url_status": "online",
        "threat": "malware_download", "tags": ["test-tag"],
        "date_added": "2026-02-03 00:00:00 UTC", "last_online": None,
        "reporter": "researcher", "payloads": [{"filename": "sample.bin", "file_type": "bin"}],
    })
    result = URLhausProvider(auth_key="secret-value", session=session).check_url("https://malware.example/p")
    assert result.match is True
    assert result.verdict == "malware_distribution"
    assert result.evidence_level == "confirmed"
    assert result.details["tags"] == ["test-tag"]
    assert result.details["payloads"] == [{"filename": "sample.bin", "file_type": "bin"}]
    call = session.post.call_args
    assert call.args[0] == "https://urlhaus-api.abuse.ch/v1/url/"
    assert call.kwargs["data"] == {"url": "https://malware.example/p"}
    assert call.kwargs["headers"]["Auth-Key"] == "secret-value"
    assert call.kwargs["allow_redirects"] is False


def test_urlhaus_negative_and_missing_key() -> None:
    session = Mock()
    missing = URLhausProvider(auth_key="", session=session).check_url("https://example.test/")
    assert missing.lookup_status == LookupStatus.UNAVAILABLE
    assert missing.error == "missing_api_key"
    session.post.return_value = response({"query_status": "no_results"})
    result = URLhausProvider(auth_key="available", session=session).check_url("https://example.test/")
    assert result.lookup_status == LookupStatus.NO_MATCH
    assert result.match is False
    assert result.verdict == "no_urlhaus_record"


@pytest.mark.parametrize("provider_name", ["phishtank", "urlhaus"])
def test_authenticated_providers_isolate_timeout(provider_name: str) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("secret should not be echoed")
    provider = (
        PhishTankProvider(app_key="key", session=session)
        if provider_name == "phishtank"
        else URLhausProvider(auth_key="key", session=session)
    )
    result = provider.check_url("https://example.test/")
    assert result.lookup_status == LookupStatus.UNAVAILABLE
    assert result.error == "timeout"
    assert "secret" not in result.model_dump_json()


def test_http_error_and_invalid_responses_are_normalized() -> None:
    session = Mock()
    session.post.return_value = response({}, status_code=503)
    result = PhishTankProvider(app_key="key", session=session).check_url("https://example.test/")
    assert result.lookup_status == LookupStatus.UNAVAILABLE
    assert result.error == "http_error_503"

    session.post.return_value = response({"not_results": True})
    invalid_phishtank = PhishTankProvider(app_key="key", session=session).check_url("https://example.test/")
    assert invalid_phishtank.lookup_status == LookupStatus.ERROR
    assert invalid_phishtank.error == "invalid_response"

    session.post.return_value = response({"query_status": "unexpected"})
    invalid_urlhaus = URLhausProvider(auth_key="key", session=session).check_url("https://example.test/")
    assert invalid_urlhaus.lookup_status == LookupStatus.ERROR
    assert invalid_urlhaus.error == "invalid_response"


def test_openphish_free_feed_exact_url_match_and_feed_cache() -> None:
    session = Mock()
    session.get.return_value = response({}, text="https://listed.example/login\nhttps://another.example/")
    provider = OpenPhishProvider(session=session)
    matched = provider.check_url("https://listed.example/login")
    assert matched.match is True
    assert matched.verdict == "phishing"
    assert matched.evidence_level == "listed"
    assert provider.check_url("https://LISTED.example/login").match is False
    assert session.get.call_count == 1
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert session.get.call_args.kwargs["stream"] is True
    session.get.return_value.close.assert_called_once()


def test_openphish_feed_failure_is_unavailable() -> None:
    session = Mock()
    session.get.side_effect = requests.Timeout("hidden detail")
    result = OpenPhishProvider(session=session).check_url("https://example.test/")
    assert result.lookup_status == LookupStatus.UNAVAILABLE
    assert result.error == "timeout"


def test_credentials_are_not_returned_or_logged(caplog: pytest.LogCaptureFixture) -> None:
    session = Mock()
    session.post.return_value = response({"results": {"in_database": False}})
    secret = "do-not-disclose-this-key"
    with caplog.at_level(logging.DEBUG):
        result = PhishTankProvider(app_key=secret, session=session).check_url("https://example.test/")
    serialized = json.dumps(result.model_dump(mode="json"))
    assert secret not in serialized
    assert secret not in caplog.text


def test_invalid_url_is_rejected_without_request() -> None:
    session = Mock()
    result = PhishTankProvider(session=session).check_url(" https://example.test/path ")
    assert result.lookup_status == LookupStatus.INVALID_INPUT
    assert result.error == "url_empty_or_surrounding_whitespace"
    session.post.assert_not_called()


def test_dotenv_config_loads_values_and_preserves_explicit_environment(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "URLHAUS_AUTH_KEY=temporary-test-key\nPHISHTANK_APP_KEY=temporary-phishtank-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("URLHAUS_AUTH_KEY", raising=False)
    monkeypatch.setattr("src.threat_intel.config.DOTENV_PATH", dotenv_path)

    # An explicit provider constructor still takes precedence over dotenv.
    session = Mock()
    provider_from_env = URLhausProvider(session=session)
    assert provider_from_env.auth_key == "temporary-test-key"
    assert get_setting("PHISHTANK_APP_KEY") == "temporary-phishtank-key"


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("URLHAUS_AUTH_KEY=dotenv-test-value\n", encoding="utf-8")
    monkeypatch.setattr("src.threat_intel.config.DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("URLHAUS_AUTH_KEY", "shell-test-value")
    provider = URLhausProvider(session=Mock())
    assert provider.auth_key == "shell-test-value"
