# Phase 11A — Domain-Grouped Benchmark Preparation

## Purpose

This benchmark is a separate evaluation protocol for measuring generalization beyond the exact URL groups used by the Phase 5 baseline. The existing URL-grouped split and its Random Forest artifact remain the historical baseline and are not replaced.

## Grouping method

The environment has no installed Public Suffix List package and the repository does not contain a local suffix-list snapshot. The benchmark therefore uses the explicit fallback `hostname_fallback_no_offline_psl`, implemented by `domain_group_key` in `src/ml/experiments.py`.

The grouping key is:

- `ip:<canonical address>` for IPv4 and IPv6 literals;
- `hostname:<case-folded hostname>` for DNS names, including `localhost` and single-label hosts;
- malformed, empty, non-HTTP(S), or hostless URLs are rejected.

This keeps repeated URLs and subdomains of the same hostname together. It does not establish registrable-domain (eTLD+1) equivalence across sibling hosts such as `a.example.com` and `b.example.com`. The benchmark must not be described as registrable-domain-disjoint until a pinned, offline Public Suffix List implementation is supplied.

URL leakage checks use the documented normalization key from `normalized_url_key`: case-fold the full URL and remove trailing slash characters. The original URL-grouped baseline remains unchanged.

## Split protocol

`domain_grouped_three_way_split` assigns whole hostname groups to approximately 70/15/15 partitions using seed `2026`. Groups are processed deterministically by descending group size with seeded tie ordering and assigned to the partition with the lowest normalized row and phishing-target load. Mixed-label hostname groups remain intact; no rows or labels are removed.

The function returns positional `train`, `validation`, and `test` index arrays. It verifies zero overlap for grouping keys, exact URL strings, and normalized URL keys before returning.

## Current dataset statistics

These figures were computed from the current raw CSV and canonical eight-feature matrix without fitting a model:

| Partition | Rows | Phishing | Legitimate | Hostname groups | Row share | Phishing share |
|---|---:|---:|---:|---:|---:|---:|
| Train | 165,086 | 70,648 | 94,438 | 154,086 | 70.0125% | 42.7947% |
| Validation | 35,362 | 15,146 | 20,216 | 33,002 | 14.9969% | 42.8313% |
| Test | 35,347 | 15,151 | 20,196 | 32,989 | 14.9906% | 42.8636% |

The small deviation from exact 70/15/15 proportions is caused by whole-domain assignment. Class proportions remain close to the source distribution.

Pairwise leakage checks:

| Partition pair | Hostname groups | Exact URLs | Normalized URLs | Shared eight-feature vectors |
|---|---:|---:|---:|---:|
| Train / validation | 0 | 0 | 0 | 2,188 |
| Train / test | 0 | 0 | 0 | 2,132 |
| Validation / test | 0 | 0 | 0 | 1,310 |

Feature-vector overlap is expected and is not treated as domain leakage: different domains can produce identical values for the eight coarse URL features. It remains an intrinsic limitation of the URL-only representation and must be reported with benchmark results.

## Phase 11B protocol

Training fits models only on the domain-grouped training partition. Cross-validation must be performed inside that partition with the same hostname grouping key, so a hostname cannot occur in both a CV training fold and its validation fold.

Model and hyperparameter choices, feature decisions, and any operating threshold must be selected using training/CV and the held-out domain-grouped validation partition. The domain-grouped test partition remains untouched until the candidate and threshold are frozen.

For a fair comparison, Phase 11B must train a new Random Forest and alternative algorithms from the same domain-grouped training partition and evaluate them on the same validation and test partitions. The existing artifact `models/url_only_random_forest_seed2026_candidate.joblib` was trained under the URL-grouped protocol; any use of it is historical diagnostic context, not a domain-disjoint-trained comparison.

## Phase 11B completed comparison

The controlled comparison used the fixed configurations already defined by
`make_baseline_models`: Logistic Regression with a training-fitted
`StandardScaler`, Random Forest with 200 trees, and XGBoost as the installed
gradient-boosting implementation. Each candidate used the same eight features,
the same hostname-grouped training/validation/test rows, and three hostname-
isolated CV folds inside training. No hyperparameter search was performed.

Validation selection was frozen before the test partition was scored. Candidate
selection ranked validation PR-AUC, grouped-CV PR-AUC, phishing recall,
phishing precision, and lower FPR. The threshold grid was 0.10 through 0.90 in
0.10 increments; the threshold tie-break ranked validation F1, recall,
precision, lower FPR, then lower threshold.

| Candidate | CV PR-AUC mean ± SD | Validation PR-AUC | Validation phishing recall | Validation FPR | Selected threshold |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.994299 ± 0.000641 | 0.995610 | 0.981777 | 0.016769 | 0.20 |
| Random Forest | **0.996566 ± 0.000374** | **0.996737** | **0.987257** | **0.000594** | **0.50** |
| XGBoost | 0.995052 ± 0.000379 | 0.994908 | 0.955236 | 0.013356 | 0.40 |

Random Forest was frozen as the research candidate. Its one-time hostname-
grouped test result at threshold 0.50 was PR-AUC **0.996145**, ROC-AUC
**0.995017**, phishing precision **0.998794**, phishing recall **0.984225**,
F1 **0.991456**, FPR **0.000891**, and FNR **0.015775** (TN 20,178; FP 18;
FN 239; TP 14,912). These values are hostname-grouped benchmark results, not
registrable-domain generalization claims.

Reproducibility record: [comparison metadata](../reports/hostname_grouped_phase11b_model_comparison.json),
[model comparison report](HOSTNAME_GROUPED_MODEL_COMPARISON.md), and the
`python -m scripts.run_hostname_grouped_experiment` command. The fitted
artifacts are under `models/hostname_grouped_phase11b/`; their sidecar metadata
records the dataset hash, feature schema, grouping policy, split statistics,
CV results, selection rationale, threshold, and artifact hash.

## Limitations and required follow-up

The hostname fallback is reproducible and enforces hostname separation, but it cannot guarantee registrable-domain separation. Before making unseen-domain claims, add and pin an offline Public Suffix List implementation or a versioned local suffix snapshot, then regenerate this benchmark under an explicitly named eTLD+1 grouping policy. A time-aware or source-aware holdout should also be considered if collection metadata becomes available.
