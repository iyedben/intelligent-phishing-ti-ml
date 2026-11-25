# Phase 11B — Hostname-Grouped Model Comparison

## Scope

This is a preliminary hostname-generalization experiment using the Phase 11A
`hostname_fallback_no_offline_psl` grouping policy. It isolates parsed hostnames
only; it does **not** establish registrable-domain/eTLD+1 isolation. The raw
dataset, canonical extractor, historical URL-grouped split, and historical
Random Forest artifact were not changed.

## Protocol

- Dataset SHA-256: `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
- Seed: `2026`
- Features: the canonical eight-feature schema, in the existing order
- Target: `1 = phishing`, `0 = legitimate`
- Split: the Phase 11A deterministic hostname-grouped 70/15/15 split
- CV: three deterministic hostname-isolated folds inside training only
- Threshold grid: `0.10, 0.20, ..., 0.90`, selected on validation only
- Test: scored once after candidate and threshold freeze

The candidates were the existing fixed Logistic Regression pipeline, Random
Forest control, and XGBoost as the already-installed gradient-boosting model.
No feature engineering, class rebalancing, hyperparameter search, or test-set
tuning was performed.

## Candidate results

| Model | CV PR-AUC mean ± SD | CV phishing recall mean ± SD | Validation PR-AUC | Validation recall | Validation precision | Validation FPR | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.994299 ± 0.000641 | 0.929364 ± 0.002876 | 0.995610 | 0.981777 | 0.977711 | 0.016769 | 0.20 |
| Random Forest | **0.996566 ± 0.000374** | **0.985619 ± 0.000311** | **0.996737** | **0.987257** | **0.999198** | **0.000594** | **0.50** |
| XGBoost | 0.995052 ± 0.000379 | 0.929447 ± 0.006055 | 0.994908 | 0.955236 | 0.981680 | 0.013356 | 0.40 |

The Random Forest was selected because it led validation PR-AUC and grouped-CV
PR-AUC, while also providing the strongest validation phishing recall,
precision, and FPR trade-off. This decision was made without accessing test
metrics.

## Frozen test result

The selected Random Forest was evaluated once on the untouched hostname-grouped
test partition at threshold `0.50`:

| Metric | Result |
|---|---:|
| PR-AUC | 0.996145 |
| ROC-AUC | 0.995017 |
| Phishing precision | 0.998794 |
| Phishing recall | 0.984225 |
| Phishing F1 | 0.991456 |
| False-positive rate | 0.000891 |
| False-negative rate | 0.015775 |
| Confusion matrix (TN, FP, FN, TP) | (20,178, 18, 239, 14,912) |

This result is separate from the historical URL-grouped baseline and should not
be presented as registrable-domain or production performance.

## Artifacts and reproducibility

The machine-readable record is
[`reports/hostname_grouped_phase11b_model_comparison.json`](../reports/hostname_grouped_phase11b_model_comparison.json).
The three research-only fitted artifacts are in
`models/hostname_grouped_phase11b/`, with sidecar metadata in `reports/`.
Each metadata record includes the dataset hash, feature schema and ordering,
target mapping, grouping policy, partition statistics, leakage checks, CV
strategy, model configuration, validation threshold table, selection rationale,
final test metrics for the selected candidate, and artifact SHA-256.

Re-run from the repository root with:

```text
python -m scripts.run_hostname_grouped_experiment --model-dir models/hostname_grouped_phase11b --report reports/hostname_grouped_phase11b_model_comparison.json
```

The command refuses to overwrite an existing artifact. The historical artifact
`models/url_only_random_forest_seed2026_candidate.joblib` remains the
URL-grouped reference and is not a fair domain-grouped training comparison.

## Limitations

Hostname grouping is a reproducible fallback because no offline Public Suffix
List implementation is installed. Sibling subdomains can therefore remain in
different groups. Shared eight-feature vectors still cross partitions (as
reported in the Phase 11A benchmark) because different URLs can have identical
length/count/scheme features. A future eTLD+1 benchmark with a pinned offline
suffix list is required before making registrable-domain generalization claims.
