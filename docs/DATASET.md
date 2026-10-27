# PhiUSIIL Dataset Acquisition and Inspection

## Identity, source, and reproducibility

- **Dataset:** PhiUSIIL Phishing URL (Website)
- **UCI dataset ID:** 967
- **Official page:** <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- **Official data endpoint used:** <https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip>
- **Acquisition method:** downloaded the UCI-hosted ZIP archive directly over HTTPS and extracted its sole CSV member without editing its contents. The CSV is `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`.
- **Raw CSV size:** 56,854,345 bytes
- **Raw CSV SHA-256:** `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), as stated by UCI.
- **Citation:** Prasad, A. & Chandra, S. (2024). *PhiUSIIL Phishing URL (Website)* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.1016/j.cose.2023.103545>.

To reacquire the official archive, download the endpoint above, extract `PhiUSIIL_Phishing_URL_Dataset.csv`, and place it under `data/raw/`. No third-party mirror was used. The ZIP is not retained; the original extracted CSV is retained untouched and ignored by Git. Run `python scripts/inspect_dataset.py` to repeat the inspection. The download API/package was not added as a dependency.

## Verified file summary

All data-quality figures below were computed from the local CSV using pandas; they are not copied from UCI's approximate display metadata.

- **Shape:** 235,795 rows × 56 columns
- **Feature count:** 54 UCI feature-role columns, plus `FILENAME` (UCI role `Other`) and `label` (target)
- **Missing values:** 0 missing cells across all 56 columns
- **Fully duplicated rows:** 0
- **Repeated URLs:** 425 rows repeat a URL beyond its first occurrence; 425 distinct URL values appear more than once
- **Constant columns:** none
- **In-memory DataFrame memory:** 174,242,818 bytes (pandas deep memory accounting)
- **Target:** `label`; UCI defines `1` as legitimate and `0` as phishing

| Label | Meaning | Rows | Percentage |
|---:|---|---:|---:|
| 0 | Phishing | 100,945 | 42.8105% |
| 1 | Legitimate | 134,850 | 57.1895% |

## Exact source columns and inferred availability

UCI's variable metadata endpoint assigns roles and broad types. For this dataset, its per-variable `description` fields are empty; the page says the collection combines URL and webpage source-code features and says `FILENAME` can be ignored. The categories below use the published column names and this dataset-level description. The split is a methodological assessment, not a detailed definition supplied by UCI for every column.

### A. URL-only / pre-fetch

These can be calculated from the submitted URL string without requesting the destination page. Probability/similarity statistics still require a defined, fixed reference corpus or lookup; they are not necessarily derivable from the string alone.

`URL`, `URLLength`, `Domain`, `DomainLength`, `IsDomainIP`, `TLD`, `URLSimilarityIndex`, `CharContinuationRate`, `URLCharProb`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`.

`TLDLegitimateProb` is based on TLD reputation/probability information rather than just the URL's characters. Treat it as an auxiliary, versioned lookup or a derived statistic that needs provenance; it is not a free URL parser output. UCI's summary calls `CharContinuationRate`, `URLTitleMatchScore`, `URLCharProb`, and `TLDLegitimateProb` derived from existing features but does not publish their formulas in the variable table.

### B. Webpage/content-derived / post-fetch

These depend on retrieving or parsing the destination response, its HTML, page title, resources, or behavior. They cannot be reproduced safely from a URL string alone.

`LineOfCode`, `LargestLineLength`, `HasTitle`, `Title`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`.

This is a conservative operational grouping inferred from names and UCI's dataset-level statement that source-code features are included. For example, `Robots` and redirect counts imply network/page behavior, while title, forms, and resource counts require response content. UCI provides no detailed variable definitions for these fields in its variable descriptions. The final application must not visit an arbitrary submitted URL simply to obtain these training-time features. A URL-only application would need an explicitly scoped URL-only feature set and a separate evaluation; no such feature selection is performed in this phase.

### Metadata / identifier field

`FILENAME` is categorized by UCI as `Other` and UCI explicitly says it can be ignored. It has 235,795 distinct values (one per row) in the downloaded file and should be treated as an identifier/non-feature candidate. Do not assume its numbering is harmless without checking its ordering/provenance.

### Target label

`label` is the classification target, not an input feature. UCI's mapping is `0 = phishing`, `1 = legitimate`.

## Schema and observed types

Types are those inferred when loading the CSV with the project environment's pandas. In source order:

```text
FILENAME: str
URL: str
URLLength: int64
Domain: str
DomainLength: int64
IsDomainIP: int64
TLD: str
URLSimilarityIndex: float64
CharContinuationRate: float64
TLDLegitimateProb: float64
URLCharProb: float64
TLDLength: int64
NoOfSubDomain: int64
HasObfuscation: int64
NoOfObfuscatedChar: int64
ObfuscationRatio: float64
NoOfLettersInURL: int64
LetterRatioInURL: float64
NoOfDegitsInURL: int64
DegitRatioInURL: float64
NoOfEqualsInURL: int64
NoOfQMarkInURL: int64
NoOfAmpersandInURL: int64
NoOfOtherSpecialCharsInURL: int64
SpacialCharRatioInURL: float64
IsHTTPS: int64
LineOfCode: int64
LargestLineLength: int64
HasTitle: int64
Title: str
DomainTitleMatchScore: float64
URLTitleMatchScore: float64
HasFavicon: int64
Robots: int64
IsResponsive: int64
NoOfURLRedirect: int64
NoOfSelfRedirect: int64
HasDescription: int64
NoOfPopup: int64
NoOfiFrame: int64
HasExternalFormSubmit: int64
HasSocialNet: int64
HasSubmitButton: int64
HasHiddenFields: int64
HasPasswordField: int64
Bank: int64
Pay: int64
Crypto: int64
HasCopyrightInfo: int64
NoOfImage: int64
NoOfCSS: int64
NoOfJS: int64
NoOfSelfRef: int64
NoOfEmptyRef: int64
NoOfExternalRef: int64
label: int64
```

The five string columns and unique counts are: `FILENAME` (235,795), `URL` (235,370), `Domain` (220,086), `TLD` (695), `Title` (197,874). These counts describe distinct observed strings, not feature importance.

## Leakage and deployment risks to assess later

No columns have been removed or transformed. These are candidates for future methodological review, not automatic exclusions:

- **Direct target leakage:** `label` is the target and must never be an input. Any derived feature (`URLSimilarityIndex`, `TLDLegitimateProb`, `CharContinuationRate`, `URLCharProb`, title-match scores) needs formula/provenance review to ensure it was not calculated using labels, a target-informed source list, or information unavailable at prediction time. UCI says some are derived but does not specify the formulas in its variable table.
- **Identifier/source leakage:** `FILENAME` is unique per row and explicitly marked by UCI as ignorable. Its sequence or generation may encode acquisition order/source. Raw `URL`, `Domain`, and `Title` are high-cardinality identifiers; memorizing exact domains/sites can inflate random row-split results and obscure generalization. Domain-level or time-aware evaluation may be needed later.
- **Duplicated information:** URL length, character counts, ratios, obfuscation counts, domain/TLD fields, and URL similarity/probability scores overlap by construction or may be mathematically derived from one another. `URL` and `Domain` repeat exactly across rows; there are 425 duplicate URL occurrences. These relationships can weight evidence multiple times and require analysis, but raw data remains untouched.
- **Dataset-source leakage:** the authors say the data combines URLs and corresponding webpages and describes most analyzed URLs as latest; source/time collection details and exact sampling procedure are not fully specified in UCI's variable table. Source-specific templates, acquisition artifacts, or ordering can become shortcuts. `FILENAME` warrants particular scrutiny.
- **Unrealistic deployment assumptions:** page title, HTML, forms, redirects, resource references, and other post-fetch properties require network access and introduce arbitrary-URL retrieval risks. A final URL-only service cannot claim performance using these fields unless a separately designed and authorized safe retrieval component exists. TLD probability and similarity values also need documented, reproducible reference data.
- **Repeated samples:** identical URLs across rows can cross evaluation partitions if a future split is row-random. This phase reports, but does not deduplicate or split, the source data.

## Inspection script

`scripts/inspect_dataset.py` loads the raw CSV read-only and prints shape, source-order columns, dtypes, missing counts, row/URL duplicates, target distribution, categorical cardinalities/values (full values for low-cardinality fields and examples for very high-cardinality fields), constants, likely identifiers, and deep memory usage. It does not train a model or alter the CSV.

## Limitations

The UCI page reports 54 features and 235,795 instances, consistent with the 54 feature-role columns and 235,795 rows in this CSV; the file also includes `FILENAME` and `label`, giving 56 total columns. UCI's variable-level descriptions are blank, so the field-level URL-only/post-fetch assignment is reasoned from names and the dataset-level statement and should be revisited if an authoritative detailed feature specification becomes available. Class proportions and quality checks describe this dataset snapshot only; they do not establish current real-world prevalence or deployment performance.
