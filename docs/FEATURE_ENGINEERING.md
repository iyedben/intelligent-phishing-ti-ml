# Deterministic URL Feature Engineering Policy

## Purpose and deployment boundary

The production input is a URL string. `src.features.extract_url_features(url)` performs only local parsing and literal string operations. It has no network or filesystem code, uses no external API, and does not resolve DNS, request pages, download resources, execute JavaScript, or render content. Training, evaluation, and future inference must call this same extractor and use the canonical ordering in `src/features/schema.py`.

The extractor accepts an absolute `http://` or `https://` URL with a hostname. It raises `TypeError` for non-string input and `ValueError` for empty, malformed, whitespace/control-containing, non-HTTP(S), hostless, or invalid-port input. It preserves the supplied string; it does not trim, normalize case, percent-decode, IDNA-convert, or canonicalize it. Unicode URL strings are supported as Python strings. Character lengths count Unicode code points.

## A. Included deterministic URL features

These are explicit calculations made from the supplied string only. Inclusion means reproducible under this project's definition; it does not mean every field is appropriate as a model predictor or exactly compatible with UCI's undocumented preprocessing.

| Feature | Definition and calculation | Type | PhiUSIIL field | Derived? | Edge cases / caveats |
|---|---|---|---|---|---|
| `URLLength` | `len(url)`: Unicode code point count of the exact input | `int` | `URLLength` | Yes | No trimming or escape decoding. UCI values often differ by one; compatibility is unresolved. |
| `Domain` | `urlsplit(url).netloc`, preserved as written | `str` | `Domain` | Yes | This is the authority, not a normalized registered domain; it can contain userinfo and a port and is high-cardinality. Do not automatically use it as a predictor. |
| `DomainLength` | `len(urlsplit(url).netloc)` | `int` | `DomainLength` | Yes | Counts any userinfo and port because the source `Domain` field is the authority representation. |
| `IsDomainIP` | `1` if `ipaddress.ip_address(urlsplit(url).hostname)` accepts the whole parsed hostname; otherwise `0` | `int` (0/1) | `IsDomainIP` | Yes | Syntactic IPv4/IPv6 literal check only; no DNS. Deliberately does not flag an IP-looking substring inside a longer hostname. |
| `NoOfDegitsInURL` | Count of ASCII `0`–`9` characters over the complete input | `int` | `NoOfDegitsInURL` | Yes | Source spelling is retained. Includes path, query, fragment, port, and percent-escape digits. UCI's exact counting scope is undocumented. |
| `NoOfEqualsInURL` | `url.count('=')` over the complete input | `int` | `NoOfEqualsInURL` | Yes | Counts literal equals signs wherever they occur, including fragment text and nested URL strings. |
| `NoOfQMarkInURL` | `url.count('?')` over the complete input | `int` | `NoOfQMarkInURL` | Yes | Counts literal question marks, including nested URL/fragment text and a terminal `?`. |
| `NoOfAmpersandInURL` | `url.count('&')` over the complete input | `int` | `NoOfAmpersandInURL` | Yes | Counts literal ampersands; it does not interpret `&amp;` or decode escapes. UCI mismatches remain unresolved. |
| `IsHTTPS` | `1` iff `urlsplit(url).scheme.casefold() == 'https'`, else `0` | `int` (0/1) | `IsHTTPS` | Yes | Scheme only; does not test TLS. Nested HTTPS text in an HTTP URL does not make this value 1. |

`URL` is the input to the extractor, not a returned model feature. `FILENAME` is dataset metadata/identifier and `label` is the target; neither is returned.

### Why the returned set is deliberately limited

The schema contains nine fields with simple, explicit string/parser definitions. Some have source compatibility gaps. The extractor's definition remains deterministic and is the value future training must use; do not combine these recomputed values with the precomputed UCI columns without an explicit compatibility decision. `Domain` is retained for source mapping and analysis, but its high cardinality and possible userinfo make it a poor default predictor. `URLLength` is fully defined here even though the UCI values do not consistently equal `len(URL)`.

## B. Excluded post-fetch features

The following PhiUSIIL fields need webpage source/content, resource inspection, or network behavior and are excluded from this offline extractor:

`LineOfCode`, `LargestLineLength`, `HasTitle`, `Title`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`.

Titles, forms, and resource counts require page data. Redirect fields require observing requests. `Robots` requires a robots response or equivalent crawl result. The exact definitions of `Bank`, `Pay`, and `Crypto` are not supplied by UCI; their names suggest page/content features, so they remain excluded rather than guessed.

## C. Uncertain URL-derived features

These names may suggest URL parsing, but their exact UCI definitions or required reference data cannot be reproduced reliably from the published schema. They are not implemented and are not returned:

| Feature | Why uncertain |
|---|---|
| `TLD` | UCI does not define suffix extraction; public-suffix conventions are unspecified. The data contains numeric-only and port-like values in this column. |
| `TLDLength` | Depends on the undefined `TLD` extraction rule. |
| `NoOfSubDomain` | Requires a rule for registered domains/public suffixes; no rule is documented. |
| `TLDLegitimateProb` | Requires a probability reference population and provenance; it could encode dataset/source or label information. |
| `URLSimilarityIndex` | Similarity reference set and formula are unpublished; it could encode target-informed or dataset-specific information. |
| `CharContinuationRate` | UCI calls it derived but does not publish the calculation. |
| `URLCharProb` | Probability reference data and formula are unpublished. |
| `HasObfuscation` | The exact characters/conditions considered obfuscation are undefined. |
| `NoOfObfuscatedChar` | The set of obfuscated characters/encoding rules are undefined. |
| `ObfuscationRatio` | Depends on the undefined obfuscation count and denominator. |
| `NoOfLettersInURL` | The dataset's count does not consistently match a literal count over the stored URL; the counting scope is undocumented. |
| `LetterRatioInURL` | Depends on the undocumented letter count and denominator. |
| `DegitRatioInURL` | Depends on the source's digit count and denominator, whose observed values are not fully reproducible. |
| `NoOfOtherSpecialCharsInURL` | The included/excluded character classes are not defined. |
| `SpacialCharRatioInURL` | Depends on the undefined special-character count and denominator. |

UCI's dataset-level description says some features are derived but provides no field-level formulas. A future authoritative definition or provenance record may change these classifications; until then they remain out of scope.

## Validation against PhiUSIIL

Run `python scripts/validate_features.py`. It reads the original CSV and compares extractor values on a deterministic, label-stratified sample of 5,000 rows per class (`random_state=2026`). It does not modify the dataset. Numeric fields include mean, median, and maximum absolute differences. String fields use exact equality.

| Feature | Exact matches / sample | Match | Classification |
|---|---:|---:|---|
| `URLLength` | 2,434 / 10,000 | 24.340% | UNRESOLVED |
| `Domain` | 10,000 / 10,000 | 100.000% | MATCH |
| `DomainLength` | 10,000 / 10,000 | 100.000% | MATCH |
| `IsDomainIP` | 9,999 / 10,000 | 99.990% | MINOR/EXPECTED DIFFERENCE |
| `NoOfDegitsInURL` | 9,935 / 10,000 | 99.350% | UNRESOLVED |
| `NoOfEqualsInURL` | 9,990 / 10,000 | 99.900% | MINOR/EXPECTED DIFFERENCE |
| `NoOfQMarkInURL` | 9,999 / 10,000 | 99.990% | MINOR/EXPECTED DIFFERENCE |
| `NoOfAmpersandInURL` | 9,829 / 10,000 | 98.290% | UNRESOLVED |
| `IsHTTPS` | 9,972 / 10,000 | 99.720% | UNRESOLVED |

**Value-level exact agreement:** 82,158 / 90,000 comparisons (**91.2867%**). At the field level, 2/9 (22.22%) are exact matches on every sampled row, 3/9 (33.33%) have a documented minor edge-case difference, and 4/9 (44.44%) remain unresolved. Numeric absolute-difference results are printed by the validation script.

Interpretation:

- `Domain` and `DomainLength` match exactly in the sample because this implementation uses the URL authority/netloc convention found in the CSV.
- The one `IsDomainIP` difference is an IPv4-looking label inside a longer hostname; the extractor requires the complete parsed hostname to be an IP literal.
- `NoOfEqualsInURL` differences were terminal `=` characters; `NoOfQMarkInURL`'s difference was a terminal `?`, which this implementation counts literally.
- `URLLength` differs on 75.66% of the sample, always by one character in this run. Do not claim compatibility or use the stored UCI column as though it were this implementation's calculation.
- Digit and ampersand counts differ on some rows, and `IsHTTPS` differs where HTTP URLs contain nested HTTPS text. The exact UCI preprocessing is not documented, so the mapping remains unresolved even though this implementation's own string/scheme rules are deterministic.

This is a sample result, not a proof over every dataset row. Re-run the script after any extractor/schema change. Future ML experiments must generate training and inference values through this same schema and must not silently substitute precomputed values where validation is unresolved.

## Machine-readable schema and future use

`src/features/schema.py` is the single source of truth for feature names, order, expected Python types, definitions, corresponding PhiUSIIL columns, derived status, and caveats. The extractor uses that ordering. The training pipeline and future FastAPI inference should call `extract_url_features` and use `FEATURE_NAMES`; they should not maintain parallel feature lists.

No model was trained, no feature selection was performed, and no raw or processed dataset was created in this phase.
