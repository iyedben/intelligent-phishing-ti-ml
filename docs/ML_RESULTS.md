# Phase 5 — URL-Only Baseline Experiment Results

## Scope and interpretation

These controlled baseline experiments use only numerical values recomputed from raw URLs by `src/features/extract_url_features`. The raw CSV and its labels were not modified. Source label `0` (phishing) was mapped in memory to model-positive `1`; source label `1` (legitimate) was mapped to model-negative `0`. Every precision, recall, F1, PR-AUC, confusion count, FPR, and FNR below is therefore for **phishing as the positive class**.

These results describe this PhiUSIIL snapshot and this random URL-group split. They do not prove generalization to unseen domains, time periods, data sources, or current web traffic. They are not a claim of comprehensive or production-ready phishing protection.

## Dataset construction and features

- Dataset: official UCI PhiUSIIL Phishing URL Dataset, ID 967.
- Rows: **235,795**, all retained.
- Source CSV SHA-256: `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`.
- Model feature count: **8 numerical features**.
- Target: source `0 = phishing`, `1 = legitimate`; model `1 = phishing`, `0 = legitimate`.
- Construction: each URL was passed to the Phase 4 extractor; precomputed UCI feature values were not used as predictors.
- Excluded from the matrix: `Domain` (high-cardinality authority identifier), raw `URL`, `FILENAME`, `Title`, target, all post-fetch features, and uncertain features. `Domain` is retained by the extractor for audit/source mapping only.
- Canonical feature order: `URLLength`, `DomainLength`, `IsDomainIP`, `NoOfDegitsInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `IsHTTPS`.
- Schema version: `1.0.0`.

All features were generated from the exact URL text, without DNS, HTTP, page access, or external services. Because several extractor definitions do not exactly match stored UCI values (see [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)), the same extractor must be used to rebuild training, validation, and inference values. Mixing these features with the stored UCI columns is not supported.

## Data quality

| Check | Result |
|---|---:|
| Rows / model features | 235,795 / 8 |
| Missing model feature values | 0 |
| Infinite/non-finite model values | 0 |
| Source rows excluded | 0 |
| Fully duplicate source rows | 0 |
| Repeated URL occurrences beyond first | 425 |
| Distinct repeated URLs | 425 |
| Repeated URL groups with conflicting labels | 0 |
| Duplicate model-feature-vector occurrences beyond first | 218,054 |
| Distinct duplicate feature vectors with mixed labels | 132 |

The 132 mixed-label feature-vector groups remain in the dataset. Their rows have identical eight numeric inputs but different labels, so these features cannot distinguish them. No rows were dropped to improve scores.

## Group-aware split

Exact URL strings were the grouping key. The implementation factorizes unique URLs, independently shuffles each class's groups with `numpy.random.default_rng(2026)`, and partitions those groups into 20 near-equal folds. Folds 0–13 are training, 14–16 validation, and 17–19 test. This is a deterministic, class-stratified group allocation; whole duplicate URL groups stay together. It avoids applying a row-level split to repeated URLs.

| Partition | Rows | Share | Phishing rows | Phishing share |
|---|---:|---:|---:|---:|
| Train | 165,062 | 70.0023% | 70,664 | 42.8106% |
| Validation | 35,365 | 14.9982% | 15,139 | 42.8079% |
| Test | 35,368 | 14.9995% | 15,142 | 42.8127% |

All three pairwise URL intersections were verified empty. This split controls exact-URL leakage, but does not make domains, registrable sites, acquisition sources, or time periods disjoint.

## Preprocessing and baseline configurations

No scaling is used for the tree models. Logistic Regression uses `StandardScaler` inside a scikit-learn `Pipeline`; it is fitted on training rows only and refitted within every CV training fold. No preprocessing was fitted on validation or test data.

| Model | Fixed baseline configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `random_state=2026`, default regularization/class weights; `StandardScaler` pipeline |
| Random Forest | `n_estimators=200`, `max_features='sqrt'`, `min_samples_leaf=1`, `class_weight=None`, `n_jobs=1`, `random_state=2026` |
| XGBoost | `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `binary:logistic`, `logloss`, `hist`, `n_jobs=1`, `random_state=2026` |

No hyperparameter search or class rebalancing was performed. The exact machine-readable configuration is [experiment_config.json](../ml/experiment_config.json).

## Test results at the baseline threshold 0.50

The test set was scored only after validation comparisons, threshold review, training-only cross-validation, and candidate selection were complete. All three fixed baselines are reported for comparison. PR-AUC is average precision (AP).

| Model | Accuracy | Phishing precision | Phishing recall | Phishing F1 | ROC-AUC | PR-AUC (AP) | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.970397 | 0.995640 | 0.934949 | 0.964340 | 0.994704 | 0.995400 | 0.003065 | 0.065051 |
| Random Forest | **0.993299** | **0.998862** | **0.985471** | **0.992121** | **0.995588** | **0.996414** | **0.000841** | **0.014529** |
| XGBoost | 0.965081 | 0.987931 | 0.929798 | 0.957983 | 0.994638 | 0.994872 | 0.008504 | 0.070202 |

Phishing-positive confusion counts at 0.50:

| Model | TN: legitimate cleared | FP: legitimate flagged | FN: phishing missed | TP: phishing detected |
|---|---:|---:|---:|---:|
| Logistic Regression | 20,164 | 62 | 985 | 14,157 |
| Random Forest | 20,209 | 17 | 220 | 14,922 |
| XGBoost | 20,054 | 172 | 1,063 | 14,079 |

Definitions: TP is a phishing URL correctly classified as phishing; FP is a legitimate URL incorrectly flagged as phishing; TN is a legitimate URL correctly classified as legitimate; FN is a phishing URL missed. `FPR = FP/(FP+TN)` and `FNR = FN/(FN+TP)`.

## Validation threshold analysis

Thresholds were explored on validation data for the two models with the highest validation PR-AUC (Random Forest and Logistic Regression). They were not tuned on the test set. Lower thresholds raise phishing recall but can increase false positives. No final deployment threshold is selected here.

| Model | Threshold | Phishing precision | Phishing recall | Phishing F1 | FP | FN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.10 | 0.990597 | 0.988176 | 0.989385 | 142 | 179 | 0.007021 | 0.011824 |
| Random Forest | 0.20 | 0.996002 | 0.987318 | 0.991641 | 60 | 192 | 0.002966 | 0.012682 |
| Random Forest | 0.30 | 0.997996 | 0.987053 | 0.992495 | 30 | 196 | 0.001483 | 0.012947 |
| Random Forest | 0.40 | 0.999398 | 0.986855 | 0.993087 | 9 | 199 | 0.000445 | 0.013145 |
| Random Forest | 0.50 | 0.999599 | 0.986789 | 0.993153 | 6 | 200 | 0.000297 | 0.013211 |
| Random Forest | 0.60 | 0.999732 | 0.986723 | 0.993185 | 4 | 201 | 0.000198 | 0.013277 |
| Random Forest | 0.70 | 0.999799 | 0.986459 | 0.993084 | 3 | 205 | 0.000148 | 0.013541 |
| Random Forest | 0.80 | 0.999933 | 0.985468 | 0.992648 | 1 | 220 | 0.000049 | 0.014532 |
| Random Forest | 0.90 | 1.000000 | 0.985402 | 0.992647 | 0 | 221 | 0.000000 | 0.014598 |
| Logistic Regression | 0.10 | 0.888087 | 0.992272 | 0.937293 | 1,893 | 117 | 0.093592 | 0.007728 |
| Logistic Regression | 0.20 | 0.975448 | 0.981505 | 0.978467 | 374 | 280 | 0.018491 | 0.018495 |
| Logistic Regression | 0.30 | 0.987933 | 0.968030 | 0.977880 | 179 | 484 | 0.008850 | 0.031970 |
| Logistic Regression | 0.40 | 0.993216 | 0.957461 | 0.975011 | 99 | 644 | 0.004895 | 0.042539 |
| Logistic Regression | 0.50 | 0.995317 | 0.940551 | 0.967159 | 67 | 900 | 0.003313 | 0.059449 |
| Logistic Regression | 0.60 | 0.997015 | 0.926679 | 0.960561 | 42 | 1,110 | 0.002077 | 0.073321 |
| Logistic Regression | 0.70 | 0.998404 | 0.908977 | 0.951594 | 22 | 1,378 | 0.001088 | 0.091023 |
| Logistic Regression | 0.80 | 0.999331 | 0.888566 | 0.940699 | 9 | 1,687 | 0.000445 | 0.111434 |
| Logistic Regression | 0.90 | 1.000000 | 0.860096 | 0.924787 | 0 | 2,118 | 0.000000 | 0.139904 |

## Group-aware cross-validation on training only

Five group-aware folds were constructed inside the training partition only, stratifying/shuffling exact URL groups within each class. Duplicate URLs did not cross fold boundaries. Mean and sample standard deviation are shown at threshold 0.50; PR-AUC is average precision.

| Model | Accuracy mean ± SD | Phishing precision mean ± SD | Phishing recall mean ± SD | Phishing F1 mean ± SD | ROC-AUC mean ± SD | PR-AUC mean ± SD | FNR mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.968103 ± 0.001248 | 0.994376 ± 0.000561 | 0.930758 ± 0.002718 | 0.961514 ± 0.001543 | 0.994167 ± 0.000528 | 0.994936 ± 0.000421 | 0.069242 ± 0.002718 |
| Random Forest | **0.993439 ± 0.000512** | **0.999154 ± 0.000234** | **0.985509 ± 0.001100** | **0.992284 ± 0.000606** | **0.995563 ± 0.000373** | **0.996509 ± 0.000255** | **0.014491 ± 0.001100** |
| XGBoost | 0.969236 ± 0.002338 | 0.990209 ± 0.000888 | 0.937409 ± 0.005233 | 0.963080 ± 0.002898 | 0.994638 ± 0.000345 | 0.995035 ± 0.000201 | 0.062591 ± 0.005233 |

## Model feature-importance summaries (not SHAP)

Random Forest and XGBoost importances are built-in split/gain summaries. Logistic Regression values are coefficients on standardized inputs. These are model-specific summaries, not SHAP explanations or causal effects.

| Feature | Random Forest importance | XGBoost importance | Logistic Regression standardized coefficient |
|---|---:|---:|---:|
| `IsHTTPS` | 0.411839 | 0.621566 | -7.573709 |
| `URLLength` | 0.230286 | 0.083208 | 93.120611 |
| `NoOfDegitsInURL` | 0.191917 | 0.244945 | 11.327541 |
| `DomainLength` | 0.151539 | 0.047574 | -20.334986 |
| `NoOfQMarkInURL` | 0.007715 | 0.002707 | 1.615328 |
| `NoOfEqualsInURL` | 0.005367 | 0.000000 | 0.345226 |
| `NoOfAmpersandInURL` | 0.001239 | 0.000000 | 0.355955 |
| `IsDomainIP` | 0.000097 | 0.000000 | 10.670230 |

## Candidate model

**Random Forest** is the selected research candidate. All three models were within 0.01 validation PR-AUC of the best, so selection compared training-only CV PR-AUC, validation phishing recall, and simplicity. Random Forest led validation and CV PR-AUC and phishing recall, then led the final test comparison with 98.55% phishing recall, 99.89% phishing precision, and 1.45% FNR at the baseline threshold. Its test performance materially exceeds the simpler Logistic Regression baseline on these same features, making the trade-off in interpretability reasonable for the next XAI phase. This is a candidate, not a finalized production model or threshold.

## Saved artifact and reproducibility

- Candidate artifact: `models/url_only_random_forest_seed2026_candidate.joblib` (locally ignored by Git).
- Artifact status: research candidate, not production.
- Fitted on the training partition only; test set was not used for fitting.
- Bundle contains the fitted estimator, feature names/order, schema version, positive-class mapping, dataset hash, configuration, metrics, and runtime versions.
- Reload check passed: schema version and feature order matched; the loaded model produced finite probabilities for 20 URL-derived feature rows.
- Dataset SHA-256: `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`.
- Runtime: Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, XGBoost 3.4.1, joblib 1.6.0.
- Seed: 2026. See [experiment_config.json](../ml/experiment_config.json) and the executed [experiment notebook](../ml/notebooks/02_ml_experiments.ipynb).

## Limitations and next phase

1. The eight inputs are intentionally limited and include simple length/count/scheme features. The high test metrics must be interpreted against this dataset and its labeling/collection process, not assumed to generalize.
2. The dataset has 218,054 repeated numeric feature-vector occurrences and 132 mixed-label vector groups. Several distinct URLs therefore become identical model inputs, sometimes with opposing labels.
3. Exact-URL grouping prevents duplicate URLs crossing partitions, but domains/sites and collection time/source were not grouped. Shared infrastructure or source artifacts can still produce optimistic estimates.
4. Some explicit extractor definitions do not exactly reproduce the stored UCI values. Training correctly recomputed them from each URL; authoritative UCI formulas remain unavailable for several fields.
5. No external phishing feeds, recent independent dataset, calibration study, cost model, adversarial test, or deployment monitoring was used.
6. Threshold 0.50 is only the baseline reporting point. Threshold results show operational trade-offs but do not finalize a risk decision threshold.
7. Feature importances are not explainability. SHAP, API/frontend, and Threat Intelligence work remain for later phases.

The feature layer and first controlled ML comparison are ready for the next XAI/SHAP phase, subject to these limitations and with no claim of universal phishing detection.
