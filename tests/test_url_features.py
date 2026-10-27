"""Behavior tests for the URL-only extractor using explicit synthetic URLs."""

import socket

import pytest

from src.features import FEATURE_NAMES, extract_url_features


def test_normal_https_url_and_stable_order() -> None:
    url = "https://example.com"
    result = extract_url_features(url)

    assert tuple(result) == FEATURE_NAMES
    assert result == {
        "URLLength": len(url),
        "Domain": "example.com",
        "DomainLength": len("example.com"),
        "IsDomainIP": 0,
        "NoOfDegitsInURL": 0,
        "NoOfEqualsInURL": 0,
        "NoOfQMarkInURL": 0,
        "NoOfAmpersandInURL": 0,
        "IsHTTPS": 1,
    }


def test_http_path_query_fragment_and_literal_character_counts() -> None:
    url = "http://sub-domain.example.co.uk:8080/path-1?x=2&y=3#frag?z=4"
    result = extract_url_features(url)

    assert result["Domain"] == "sub-domain.example.co.uk:8080"
    assert result["DomainLength"] == len("sub-domain.example.co.uk:8080")
    assert result["URLLength"] == len(url)
    assert result["NoOfDegitsInURL"] == 8
    assert result["NoOfEqualsInURL"] == 3
    assert result["NoOfQMarkInURL"] == 2
    assert result["NoOfAmpersandInURL"] == 1
    assert result["IsHTTPS"] == 0


def test_encoded_characters_are_counted_as_written() -> None:
    url = "https://example.com/a%2Fb?next=%3F"
    result = extract_url_features(url)

    assert result["NoOfDegitsInURL"] == 2
    assert result["NoOfQMarkInURL"] == 1


def test_ip_literal_host_is_detected_without_dns() -> None:
    result = extract_url_features("http://192.0.2.1:8080/a")

    assert result["Domain"] == "192.0.2.1:8080"
    assert result["IsDomainIP"] == 1


def test_ipv6_literal_host_is_detected() -> None:
    result = extract_url_features("https://[2001:db8::1]:8443/path")

    assert result["Domain"] == "[2001:db8::1]:8443"
    assert result["IsDomainIP"] == 1


def test_unicode_idn_is_preserved_without_network_or_idna_lookup() -> None:
    url = "https://例え.テスト/道?q=猫"
    result = extract_url_features(url)

    assert result["Domain"] == "例え.テスト"
    assert result["DomainLength"] == len("例え.テスト")
    assert result["URLLength"] == len(url)


def test_extraction_is_deterministic() -> None:
    url = "https://example.com/path?item=12&next=3"

    assert extract_url_features(url) == extract_url_features(url)


def test_extraction_does_not_open_a_network_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("feature extraction attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_connection)
    monkeypatch.setattr(socket, "create_connection", deny_connection)

    assert extract_url_features("https://example.com")["IsHTTPS"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "",
        "example.com/path",
        "ftp://example.com/file",
        "https:///missing-host",
        "https://example.com:invalid/path",
        "https://[2001:db8::1/path",
        "https://exa mple.com/path",
    ],
)
def test_malformed_or_non_web_urls_raise_value_error(url: str) -> None:
    with pytest.raises(ValueError):
        extract_url_features(url)


def test_non_string_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        extract_url_features(None)  # type: ignore[arg-type]
