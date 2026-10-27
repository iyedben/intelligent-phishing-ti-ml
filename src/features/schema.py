"""Canonical schema shared by feature extraction, validation, and future consumers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Description of one stable extractor output field."""

    name: str
    python_type: type
    description: str
    calculation: str
    phiusiil_column: str | None
    derived: bool
    caveat: str = ""


FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="URLLength",
        python_type=int,
        description="Character count of the exact input URL string.",
        calculation="len(url), counting Python Unicode code points; no trimming or decoding.",
        phiusiil_column="URLLength",
        derived=True,
        caveat="Deterministic definition, but EDA validation found frequent one-character differences from the stored UCI column.",
    ),
    FeatureSpec(
        name="Domain",
        python_type=str,
        description="Raw URL authority/netloc substring.",
        calculation="urllib.parse.urlsplit(url).netloc, preserving spelling, userinfo, and port.",
        phiusiil_column="Domain",
        derived=True,
        caveat="High-cardinality identifier; may include userinfo or port. Do not use as a predictor without an explicit decision.",
    ),
    FeatureSpec(
        name="DomainLength",
        python_type=int,
        description="Character count of the raw URL authority/netloc.",
        calculation="len(urllib.parse.urlsplit(url).netloc).",
        phiusiil_column="DomainLength",
        derived=True,
        caveat="Counts userinfo and port when present, matching the authority representation rather than only the DNS hostname.",
    ),
    FeatureSpec(
        name="IsDomainIP",
        python_type=int,
        description="Whether the parsed hostname itself is a valid IPv4 or IPv6 literal.",
        calculation="ipaddress.ip_address(urlsplit(url).hostname) succeeds -> 1, otherwise 0.",
        phiusiil_column="IsDomainIP",
        derived=True,
        caveat="Does not resolve DNS. UCI appears to flag some hostnames containing IPv4-looking labels, which this stricter literal check does not.",
    ),
    FeatureSpec(
        name="NoOfDegitsInURL",
        python_type=int,
        description="Count of ASCII digits in the exact input URL string (UCI source spelling retained).",
        calculation="Count each character in '0123456789' across url, including path, query, and fragment.",
        phiusiil_column="NoOfDegitsInURL",
        derived=True,
        caveat="The UCI values differ on a small subset; exact original counting scope is undocumented.",
    ),
    FeatureSpec(
        name="NoOfEqualsInURL",
        python_type=int,
        description="Count of literal equals signs in the exact input URL string.",
        calculation="url.count('='), including path, query, and fragment.",
        phiusiil_column="NoOfEqualsInURL",
        derived=True,
        caveat="UCI mismatches observed mainly at trailing delimiters; source handling is undocumented.",
    ),
    FeatureSpec(
        name="NoOfQMarkInURL",
        python_type=int,
        description="Count of literal question marks in the exact input URL string.",
        calculation="url.count('?'), including nested URL text and fragments.",
        phiusiil_column="NoOfQMarkInURL",
        derived=True,
        caveat="A trailing standalone question mark is counted here; UCI omits it in a small observed edge case.",
    ),
    FeatureSpec(
        name="NoOfAmpersandInURL",
        python_type=int,
        description="Count of literal ampersand characters in the exact input URL string.",
        calculation="url.count('&'), including query, path, and fragment text.",
        phiusiil_column="NoOfAmpersandInURL",
        derived=True,
        caveat="Substantial UCI mismatches suggest different ampersand handling, but the exact cause is unresolved; no compatibility claim is made.",
    ),
    FeatureSpec(
        name="IsHTTPS",
        python_type=int,
        description="Whether the URL's parsed scheme is HTTPS.",
        calculation="urlsplit(url).scheme.casefold() == 'https' -> 1, otherwise 0.",
        phiusiil_column="IsHTTPS",
        derived=True,
        caveat="Scheme-only; does not test TLS. Some UCI positives occur on HTTP URLs containing nested HTTPS text.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(feature.name for feature in FEATURE_SCHEMA)
