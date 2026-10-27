# Threat Intelligence Layer

## Scope and evidence separation

The package `src/threat_intel/` provides provider lookups and structured results for the later application layer. It does not open submitted URLs. Provider requests go only to fixed provider endpoints; a submitted URL is sent as an encoded API form value or compared locally with a feed. Redirects are disabled, request timeouts are explicit, and every provider returns an unavailable/error result rather than propagating ordinary request failures.

An ML probability, a PhishTank phishing record, an OpenPhish feed listing, and a URLhaus malware-distribution listing are different evidence types. Each `ThreatIntelResult` preserves source, original queried URL, status, match, verdict, evidence level, timestamp, latency, provider details, and a sanitized error code. There is no generic `is_malicious` field and no provider probability is invented. `confidence` remains unset because the selected providers do not return a comparable confidence score.

## Providers

| Provider | Purpose and semantics | Access / authentication | Freshness and limits |
|---|---|---|---|
| PhishTank | Community phishing URL lookup. Only `in_database=true`, `verified=true`, and `valid=true` maps to `verdict="phishing"` with `evidence_level="confirmed"`. A database record alone is only a listed submission, not confirmation. | Official `POST https://checkurl.phishtank.com/checkurl/`; optional `PHISHTANK_APP_KEY`. Unauthenticated one-off checks are supported with stricter limits. A descriptive User-Agent is sent. | 8-second timeout, no automatic retries, 5-minute optional in-memory result cache. Honor provider rate-limit headers and HTTP 509 responses. |
| URLhaus | Malware URL intelligence. A match maps to `verdict="malware_distribution"`, never to a phishing verdict by itself. Online status is retained as provider metadata and gets a stronger evidence level than an offline listing. | Official Community API `POST https://urlhaus-api.abuse.ch/v1/url/`; `URLHAUS_AUTH_KEY` is required. A free Auth-Key is available through abuse.ch registration. | 8-second timeout, no automatic retries, 5-minute optional in-memory result cache. The API is under fair-use terms; commercial/for-profit usage may require a paid enhanced API. |
| OpenPhish Community Feed | Optional local exact-match phishing listing. The feed contains URLs only; the code makes no unsupported claims about target, verification, or confidence. | Official text feed linked from OpenPhish's feed page; no API key. The Community Feed is free, but limited compared with paid feeds; users must comply with its Terms of Use. | Feed is advertised with a 12-hour update interval. One downloaded snapshot is cached in memory for 12 hours per provider instance. |

PhishTank's official [API instructions](https://phishtank.org/api_info.php) describe the `url`, `format`, and optional `app_key` form parameters, descriptive User-Agent, and stricter limits without a key. URLhaus's [Community API documentation](https://urlhaus.abuse.ch/api/) and [URL lookup API](https://urlhaus-api.abuse.ch/#url-lookup) document the required Auth-Key and URL query. URLhaus states that its collection targets URLs directly distributing malware and distinguishes phishing from malware in its [submission policy](https://urlhaus.abuse.ch/api/#policy). OpenPhish's [feed page](https://www.openphish.com/phishing_feeds.html) lists Community as a limited, free text feed updated every 12 hours; its [linked feed](https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt) is used without scraping or premium endpoints.

## URL handling

`normalize_lookup_url()` validates an absolute HTTP(S) URL and returns the same string unchanged. It rejects empty input, leading/trailing or embedded whitespace, control characters, malformed ports, and non-HTTP(S) URLs rather than silently trimming or rewriting them. Scheme/host case, path, query, percent escapes, trailing slash, and fragment are preserved. Requests encodes the submitted value only at the HTTP form boundary; the original URL is retained in the result. Provider matching rules remain provider-defined: for example, OpenPhish compares the exact submitted URL to exact feed lines, so a missing slash, changed case, or fragment may affect a match.

The network requests use fixed HTTPS provider endpoints and `allow_redirects=False`. The URL argument is never passed as a request destination, fetched, DNS-resolved, or opened. Querying a provider discloses the submitted URL to that provider; this is an expected privacy consideration for a future service interface.

## Normalized result statuses

- `match`: the provider has a corresponding record/listing. Inspect `verdict` and `evidence_level` to understand what the record means.
- `no_match`: a valid provider response indicates no matching record.
- `unavailable`: provider credentials are absent where required, a timeout/network failure occurred, or the provider returned an HTTP availability/rate-limit error.
- `error`: the response was malformed or rejected the lookup.
- `invalid_input`: local validation rejected the URL before a request.

`match=null` means no reliable match/no-match answer was obtained. Error fields use fixed safe codes (for example `timeout`, `missing_api_key`, or `http_error_503`); exception text, response headers, and credentials are not returned or logged. Provider-specific `details` are allow-listed and JSON-safe.

## Caching and reliability

`MemoryTTLCache` is optional, process-local, and keyed by provider plus a SHA-256 digest of the exact URL; it does not store API credentials in the key. Successful match/no-match results may be cached for five minutes. Errors and unavailable responses are not cached. OpenPhish's provider object keeps only its 12-hour feed snapshot in memory. Cache state is lost on process restart; there is no persistent cache or database. A future service should keep provider instances alive to use the cache and should consider rate limiting and bounded concurrency. There are no retries or background bulk downloads.

## Credentials and security

At provider initialization, the configuration helper loads the repository-root `.env` with `python-dotenv` and `override=False`. Explicit process environment values therefore take precedence. `.env.example` contains placeholders only. For example, in PowerShell:

```text
PHISHTANK_APP_KEY=
URLHAUS_AUTH_KEY=
```

`.env.example` contains placeholders only. `.env` and `.env.*` are ignored by Git (with `.env.example` explicitly retained). `python-dotenv` is a direct runtime dependency because local development now loads those settings without custom shell setup; `requirements-dev.txt` inherits runtime dependencies through `-r requirements.txt`. PhishTank's key is optional; URLhaus requires its Auth-Key. Credentials are never embedded in returned details, error messages, or logs. Requests use finite 8-second timeouts. Response parsing is schema-aware and does not execute provider content. Payload metadata is projected to a small set of fields and is not downloaded.

## Failure behavior and current verification

Provider requests are isolated: PhishTank can be unavailable while URLhaus/OpenPhish still contribute evidence. Missing URLhaus credentials return `unavailable/missing_api_key`; PhishTank still attempts an unauthenticated check if invoked without its optional key. No provider failure invalidates an already-computed ML result. The normal automated test suite uses mocked HTTP responses and makes no live provider requests. In prior local manual verification, URLhaus loaded the configured key and returned a successful no-match for `https://example.com`; OpenPhish also returned a feed result, while PhishTank returned HTTP 403 because registration/API access was unavailable. Later requests from restricted execution environments returned normalized `request_error` statuses for the providers. These outcomes are time- and network-environment-dependent; no live provider request was made during Phase 9 validation. A no-match remains limited to the provider's current data and is not proof of safety.

The component does not yet provide distributed throttling, durable caching, feed-signature verification, or provider-specific freshness guarantees beyond documented/feed-derived timestamps. A no-match result is time-bound and must not be interpreted as proof of safety.
