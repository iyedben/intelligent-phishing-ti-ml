"""Offline URL parsing and deterministic feature extraction."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from .schema import FEATURE_NAMES


def extract_url_features(url: str) -> dict[str, int | str]:
    """Return the approved URL-only features in canonical schema order.

    The input must be an absolute HTTP or HTTPS URL. Parsing and character
    counting operate only on the supplied string. This function does not
    resolve DNS, access the filesystem, make HTTP requests, decode URL escapes,
    or contact external services.

    Raises:
        TypeError: if ``url`` is not a string.
        ValueError: if it is empty, malformed, contains whitespace/control
            characters, lacks a valid HTTP(S) authority, or has an invalid port.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a str")
    if not url:
        raise ValueError("url must not be empty")
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        raise ValueError("url must not contain whitespace or control characters")

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        # Accessing .port validates port syntax and range without connecting.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"malformed URL: {exc}") from exc

    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.netloc or hostname is None:
        raise ValueError("url must be an absolute HTTP(S) URL with a hostname")

    try:
        ipaddress.ip_address(hostname)
        is_ip_literal = 1
    except ValueError:
        is_ip_literal = 0

    values: dict[str, int | str] = {
        "URLLength": len(url),
        "Domain": parsed.netloc,
        "DomainLength": len(parsed.netloc),
        "IsDomainIP": is_ip_literal,
        "NoOfDegitsInURL": sum(character in "0123456789" for character in url),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "IsHTTPS": int(scheme == "https"),
    }
    # Construct through the canonical schema so return order cannot drift.
    return {name: values[name] for name in FEATURE_NAMES}
