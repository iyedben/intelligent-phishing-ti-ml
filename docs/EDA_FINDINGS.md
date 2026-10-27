# Phase 3 EDA Findings

The analysis is in [`ml/notebooks/01_eda.ipynb`](../ml/notebooks/01_eda.ipynb). It reads the official raw CSV directly and leaves it unchanged. The notebook completed end to end in the project's Python 3.12 environment. Dense distribution plots use a fixed-seed sample of up to 8,000 rows per class (`random_state=2026`); numerical summaries, correlations, and univariate rank comparisons use all rows. No model, split, transform, resample, or processed dataset was created.

## 1. Dataset characteristics

- **235,795 rows × 56 columns:** 54 UCI feature-role columns, `FILENAME`, and target `label`.
- **Missingness:** zero missing values across every column.
- **Duplicates:** no fully duplicated rows. There are 425 repeated URL occurrences across 425 URL values. None of these duplicate URL groups has conflicting labels.
- **Constants:** none.
- **Deep DataFrame memory:** about 174.2 MB; exact pandas accounting varies slightly with string-object accounting and evaluation order.
- Source labels remain unchanged: **0 = phishing**, **1 = legitimate**.

## 2. Class balance

| Label | Meaning | Rows | Share |
|---:|---|---:|---:|
| 0 | Phishing | 100,945 | 42.8105% |
| 1 | Legitimate | 134,850 | 57.1895% |

This is a moderately imbalanced but not extreme sample distribution. It is not evidence of real-world phishing prevalence.

## 3. Important feature groups and availability

### Conservative URL-only candidates

The notebook defines `URL_ONLY_FEATURES` as:

`URL`, `URLLength`, `Domain`, `DomainLength`, `IsDomainIP`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`.

Availability means string parsing only. It excludes DNS, requests, content downloads, JavaScript, and rendering. `Domain`, `DomainLength`, and `IsDomainIP` are also called out as domain/parser fields. `IsHTTPS` describes the scheme in the supplied URL, not whether TLS succeeds.

### Uncertain URL-derived/suffix candidates

`TLD`, `TLDLength`, `NoOfSubDomain`, `TLDLegitimateProb`, `URLSimilarityIndex`, `CharContinuationRate`, and `URLCharProb` are marked **UNCERTAIN**. UCI leaves the per-variable descriptions empty, and these fields depend on undocumented suffix rules, derived formulas, or reference data. In particular, a probability or similarity feature needs its calculation corpus and label provenance documented before it can be trusted as deployment-ready.

### Post-fetch fields

The notebook defines `POST_FETCH_FEATURES` as:

`LineOfCode`, `LargestLineLength`, `HasTitle`, `Title`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`.

These require page HTML/content/resources, a robots response, or network redirect behavior. Several exact constructions, especially `Bank`, `Pay`, and `Crypto`, are not defined in UCI's variable descriptions. They are not suitable inputs for the URL-only production track under the project's constraints.

## 4. Strong univariate differences

For numeric fields, the notebook reports a rank-based univariate AUC defined as `P(phishing value > legitimate value)`. A score near 0 or 1 indicates strong separation in this dataset; it does **not** establish model performance, causal importance, target-independent provenance, or deployment feasibility.

- URL-only string-derived examples: `NoOfOtherSpecialCharsInURL` AUC **0.8238**; `SpacialCharRatioInURL` **0.8076**; `NoOfLettersInURL` **0.7422**; `URLLength` **0.7324**; `NoOfDegitsInURL` **0.7280**; `DegitRatioInURL` **0.7261**; `LetterRatioInURL` **0.7151**. These show this dataset contains useful univariate URL-text differences, subject to redundancy and external validation.
- `IsHTTPS` AUC **0.2461** under the stated direction, meaning legitimate rows more often have value 1 in this sample; it does not validate HTTPS security.
- Strongest numeric separations included `URLSimilarityIndex` (**0.0039**, phishing values much lower), `LineOfCode` (**0.0097**), `NoOfExternalRef` (**0.0120**), `NoOfImage` (**0.0204**), `NoOfSelfRef` (**0.0275**), and `NoOfJS` (**0.0292**). Most are post-fetch; the similarity score is formula/provenance-uncertain. Their extreme separation warrants leakage/source-provenance review.

The notebook also reports full-data descriptive statistics, class-conditional mean/median/spread, skew, and IQR flags. Some counts are highly skewed (for example `NoOfCSS`, `NoOfEmptyRef`, and `NoOfDegitsInURL`); binary and discrete fields can trigger IQR flags for mechanical reasons, so these are not grounds for removing rows.

## 5. Correlated and potentially redundant features

Spearman analysis found eight pairs with absolute correlation at least 0.90:

| Pair | Spearman rho | Interpretation to investigate |
|---|---:|---|
| `NoOfObfuscatedChar` – `ObfuscationRatio` | 1.0000 | Near-identical/derived signal |
| `HasObfuscation` – `NoOfObfuscatedChar` | 1.0000 | Indicator follows count almost exactly |
| `HasObfuscation` – `ObfuscationRatio` | 1.0000 | Indicator follows derived ratio |
| `NoOfDegitsInURL` – `DegitRatioInURL` | 0.9967 | Count and ratio encode the same digits |
| `URLLength` – `NoOfLettersInURL` | 0.9721 | Length and letter count overlap |
| `DomainTitleMatchScore` – `URLTitleMatchScore` | 0.9642 | Similar title-matching measures; both page-dependent |
| `NoOfLettersInURL` – `LetterRatioInURL` | 0.9307 | Count and ratio overlap |
| `NoOfEqualsInURL` – `NoOfQMarkInURL` | 0.9097 | Co-occurrence/distribution in this sample; check formulas and marginal counts |

Additional webpage resource counts move together (for example CSS/JS, line count/images, and internal/external references). Correlation is descriptive here; no feature is automatically removed.

## 6. High-cardinality and categorical findings

- `FILENAME`: 235,795 unique values, 100% singleton rows. UCI says it can be ignored; do not treat it as a model feature without compelling provenance.
- `Domain`: 220,086 unique values; 93.34% unique-to-row ratio and about 90.99% of rows are singleton domains. There are 54 repeated domain groups containing both labels. Domain-level memorization and grouping should be considered in later validation.
- `Title`: 197,874 unique values; 83.92% unique-to-row ratio and about 82.08% singleton rows. It is fetched page content and may identify site/templates.
- `TLD`: 695 observed values. Its top values are `com`, `org`, and `net`; however 625 rows have numeric-only or colon/port-like `TLD` values (126 distinct values). Validate extraction before treating this column as a clean suffix feature.
- Some repeated titles also span both labels (222 repeated title groups), so raw title memorization is not a reliable URL-only signal.

No full Domain or Title one-hot encoding was attempted.

## 7. Leakage concerns

- `FILENAME` is a unique identifier and may encode source/order; UCI explicitly says it can be ignored.
- Exact `URL`, `Domain`, and `Title` values can memorize records or sites. Repeated URLs must not cross future validation partitions; group-level or temporal evaluation may be necessary.
- The 425 repeated URL groups happen to agree on label in this file, but duplicates can still inflate a row-random evaluation.
- Derived probability/similarity/title scores need formulas and provenance reviewed for target-informed statistics, reference-corpus overlap, and dataset-source shortcuts.
- Post-fetch fields create an unrealistic deployment assumption for the URL-only service and may also reflect page-collection methodology.
- Lengths/counts/ratios overlap mathematically and may repeatedly weight the same evidence.

The notebook's leakage table gives a reason and provisional future action for each concern. It does not delete columns or make final feature-selection decisions.

## 8. Implications for later experiments

Maintain two clearly separated tracks. The full-dataset benchmark may assess appropriate fields but must label post-fetch dependence and investigate source leakage. The eventual URL-only detector should use only confirmed string/parser fields unless the user explicitly changes the no-contact deployment constraint. Keep uncertain TLD and derived statistics out of claims about URL-only availability until their construction is reproduced. Later evaluation should account for repeated URLs and domains, and compare results with/without high-cardinality identifiers and unexplained derived scores.

## 9. Readiness

The dataset is ready for the next **feature-engineering design** phase. Before implementing a deployment extractor, document precise parsing rules for the conservative URL-only fields and resolve the UCI formulas/provenance for uncertain candidates. This EDA has not trained models or selected a final feature set.
