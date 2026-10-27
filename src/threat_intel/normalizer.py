"""Shared, conservative URL input handling for TI providers."""

from .base import validate_lookup_url


def normalize_lookup_url(url: str) -> str:
    """Validate but do not rewrite a submitted URL.

    Query parameters, capitalization, trailing slash, percent escapes, and
    fragments are kept exactly as supplied. Provider clients encode the string
    only in their form/request boundary. Invalid input raises ``ValueError``.
    """
    error = validate_lookup_url(url)
    if error:
        raise ValueError(error)
    return url
