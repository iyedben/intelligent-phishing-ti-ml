# Phase 10B — ML Evaluation Review and Project Scope Alignment

**Status:** Evaluation specification only. No experiment, retraining, threshold change, application change, or live provider request was performed for this plan.

## 1. Purpose and scope

This document specifies how to assess the existing URL-only Random Forest candidate more carefully before strengthening claims about its performance. The current saved candidate and its schema remain unchanged. Any future split that changes which rows are in the training partition necessarily requires fitting a separate evaluation copy; that would be a later, explicitly authorized experiment and must not overwrite the current candidate.

The system's current ML output is a model prediction and phishing probability; the current XAI layer is SHAP for class 1; the current TI layer returns separate provider evidence and a qualitative correlation assessment. A numeric risk score is not implemented. See [README, current API](../README.md#19-current-api), [Correlation Rules](CORRELATION.md), and [ML results](ML_RESULTS.md).

## 2. Existing baseline and evidence

### Dataset and inputs

The Phase 5 report records the UCI PhiUSIIL dataset (ID 967), 235,795 rows, and raw CSV SHA-256 `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`. This is the hash recorded in project documentation; this plan does not re-acquire or rewrite the CSV. Source labels are `0 = phishing` and `1 = legitimate`; the modeling target is mapped to `1 = phishing` (positive class) and `0 = legitimate`.

The reported model matrix contains eight numeric values recomputed by the canonical offline extractor, in order:

1. `URLLength`
2. `DomainLength`
3. `IsDomainIP`
4. `NoOfDegitsInURL`
5. `NoOfEqualsInURL`
6. `NoOfQMarkInURL`
7. `NoOfAmpersandInURL`
8. `IsHTTPS`

The raw URL and `Domain` are not model inputs. The `Domain` authority may be used for a future grouping diagnostic, but that does not make it a predictive feature. These eight values describe URL text, parsed authority/IP-literal status, and scheme. They do not use DNS, webpage content, redirects, or browser behavior. The extractor's definitions and incomplete agreement with several stored PhiUSIIL values are documented in [Feature Engineering](FEATURE_ENGINEERING.md). The experiment pipeline constructs the model inputs by calling that extractor (`build_dataset` in [`src/ml/experiments.py`](../src/ml/experiments.py#L54-L115)).

### Existing split and reported results

The existing split groups exact URL strings, shuffles the unique URL groups within each class with NumPy `default_rng(2026)`, then assigns 20 folds: 0–13 train, 14–16 validation, and 17–19 test. The realized partitions were:

| Partition | Rows | Phishing | Legitimate | Phishing share |
|---|---:|---:|---:|---:|
| Train | 165,062 | 70,664 | 94,398 | 42.8106% |
| Validation | 35,365 | 15,139 | 20,226 | 42.8079% |
| Test | 35,368 | 15,142 | 20,226 | 42.8127% |

The split implementation checks pairwise exact-URL disjointness. Five-fold cross-validation is reported on the training partition only and also groups exact URLs. Details are in [ML Results: Group-aware split](ML_RESULTS.md#group-aware-split), [cross-validation](ML_RESULTS.md#group-aware-cross-validation-on-training-only), and [`group_aware_three_way_split`](../src/ml/experiments.py#L118-L171).

At the **reported baseline threshold 0.50**, the Random Forest test results are:

| Metric | Reported value |
|---|---:|
| Accuracy | 0.993299 |
| Phishing precision | 0.998862 |
| Phishing recall | 0.985471 |
| Phishing F1 | 0.992121 |
| ROC-AUC | 0.995588 |
| PR-AUC (average precision) | 0.996414 |
| False-positive rate | 0.000841 |
| False-negative rate | 0.014529 |

Reported phishing-positive confusion matrix counts are **TN 20,209, FP 17, FN 220, TP 14,922**. These figures are taken from [ML Results: test results](ML_RESULTS.md#test-results-at-the-baseline-threshold-050); they are not an independent re-run in this plan. The report explores thresholds 0.10–0.90 on validation data for Random Forest and Logistic Regression, but retains 0.50 only as the baseline reporting point and does not select a deployment threshold ([validation threshold analysis](ML_RESULTS.md#validation-threshold-analysis)).

### Duplicate and conflicting-vector evidence

The report records zero fully duplicated source rows; 425 repeated URL occurrences over 425 repeated URL values, with no conflicting labels among those exact repeated URL groups; 218,054 duplicate model-feature-vector occurrences; and 132 distinct feature vectors observed with mixed labels. The mixed-label vector groups were retained. Those counts are documented in [ML Results: data quality](ML_RESULTS.md#data-quality), and calculated by `build_dataset` in [`src/ml/experiments.py`](../src/ml/experiments.py#L54-L115).

The current documentation does not report how many identical feature vectors or near-duplicate URLs cross the train/validation/test boundaries. Because the split groups by exact URL—not by feature vector—identical vectors from different URLs are allowed to land in different partitions. This is a **confirmed property of the split method**; the actual cross-partition vector-overlap count is **not established by the recorded results**.

## 3. Limitations and threats to validity

### What the current evaluation establishes

It reports how three fixed baseline classifiers performed on a held-out, random, class-stratified set of exact URL groups from the same PhiUSIIL snapshot, using this project's eight-feature extractor and label mapping. The test set was described as untouched until model comparison, threshold review, training-only CV, and candidate selection were complete. The reported test metrics are therefore evidence for that experiment and data distribution.

### What it does not establish

- **Unseen domains/sites:** Domains were not split into disjoint partitions. `Domain` was excluded as a model input, so direct memorization of the raw domain string is not the demonstrated issue; nevertheless, the same sites, infrastructure, templates, or URL patterns can occur across partitions. EDA reports 54 repeated domain groups that contain both labels ([EDA Findings](EDA_FINDINGS.md#6-high-cardinality-and-categorical-findings)). The exact number of domains shared across the existing partitions is not reported.
- **Near-duplicate URL separation:** Only exact URL strings are grouped. Variants such as changed paths, query strings, host spellings, or encodings could be split across partitions. A near-duplicate analysis was not reported, so actual overlap is unknown.
- **Feature-vector independence:** Repeated vectors are numerous and 132 vector groups contain opposing labels. As above, vector overlap across partitions is methodologically possible and its realized extent is unreported. Identical inputs with conflicting labels are not distinguishable by this model; their impact should be shown, not hidden by removing rows.
- **Temporal or campaign generalization:** The documented dataset schema and provenance do not establish reliable per-URL collection timestamps or a defensible chronological order. The UCI documentation described in [Dataset](DATASET.md#leakage-and-deployment-risks-to-assess-later) leaves source/time collection details and sampling procedure unclear. A time-ordered claim is unsupported without trustworthy timestamps and provenance.
- **External-source generalization:** No external dataset evaluation is recorded. The results do not establish performance on another collection, current traffic, or future campaigns.
- **Webpage performance:** The evaluated model uses URL-only features. It does not establish how a page-content or post-fetch detector would perform.
- **Production utility or calibration:** No deployment loss/cost constraint or final decision threshold is selected. The metrics do not establish calibrated probabilities, expected operational false-positive burden, or suitability as a production security control.

### Confirmed leakage checks versus plausible risks

The target label is excluded from the predictor matrix and transformed in memory only. The model uses recomputed URL-only features rather than the precomputed UCI feature columns. Exact duplicate URL groups are kept together. These are positive controls evidenced by [`src/ml/experiments.py`](../src/ml/experiments.py#L54-L170) and [ML Results](ML_RESULTS.md).

The available evidence does **not** demonstrate that a target-derived feature leaked into the selected eight inputs, nor does it prove that domain leakage or source leakage occurred. It does show that domain/time/source independence was not imposed, the split does not group by feature vector, and the dataset contains mixed-label vector collisions. Near-duplicate and cross-partition feature-vector overlap remain questions to measure.

## 4. Proposed experiments and methodology

All experiments below are proposals for later approval. Preserve the existing candidate, original split, and its published metrics as the reference baseline. Do not change the canonical feature schema or definitions.

### Required: reproduce the reference baseline

1. Verify the UCI dataset identity and SHA-256 against the recorded source snapshot.
2. Rebuild `X` and `y` with the canonical extractor and current schema version; assert the eight input names and exact order.
3. Recreate the seed-2026 exact-URL group allocation and compare row counts, class counts, URLs, and saved-model predictions with the recorded baseline where artifacts permit.
4. Assert no exact URL crosses train, validation, or test. Report each partition's sample counts and label prevalence.
5. Recompute the documented baseline metrics at 0.50 without selecting a new threshold or using the test set to make decisions. Any discrepancy from `ML_RESULTS.md` must be investigated and documented, not silently overwritten.

### Required: duplicate, near-duplicate, and feature-vector audit

Before interpreting new metrics, report:

- exact row and exact URL duplicate counts, label agreement, and partition intersections;
- URL near-duplicate groups under a specified, conservative diagnostic rule (for example, separately reported normalized-host/path or similarity buckets); preserve the original URL and do not silently merge semantically distinct URLs;
- exact eight-feature-vector groups, their sizes, label compositions, and intersections across each split;
- confusion/error counts for the 132 mixed-label vector groups, keeping all rows unless a separate, justified protocol is approved.

The core evaluation should retain row-level metrics for comparability. A feature-vector-group-disjoint split is a **recommended sensitivity experiment**: assign every identical model vector, including all labels in a mixed-label group, to one partition only. Use a deterministic group assignment and report realized class/row proportions rather than forcing target proportions. Do not treat vector-grouped metrics as a substitute for the original baseline.

### Required for unseen-domain claims: domain-disjoint evaluation

Create a separate split in which domains/sites in the test partition do not occur in training or validation. The current URL data supports hostname extraction from the URL strings, but a registered-domain grouping rule must be specified before splitting. Prefer a reproducible registrable-domain rule using a fixed, documented public-suffix snapshot; if that is unavailable, use exact hostnames and state that subdomains may still overlap.

Keep groups intact across train/validation/test, use seed `2026` for deterministic allocation, and publish the actual numbers of groups, rows, and class counts. Do not force 70/15/15 if domain group sizes or class composition prevent it. Fit a **separate evaluation instance with the existing Random Forest configuration** using only the new training partition; do not overwrite or relabel the existing candidate. Select any evaluation threshold using the new validation partition only, then evaluate once on the untouched domain-held-out test groups. If adequate domain-disjoint assignment cannot be achieved, report the limitation rather than relaxing the grouping silently.

### Optional, conditional: temporal evaluation

Do not infer chronology from row order or `FILENAME`. Run a chronological holdout only if trustworthy URL collection timestamps and their provenance become available. Define the cutoff before looking at test outcomes; train on earlier records, use a later validation period for threshold decisions, and reserve the latest period for final evaluation. If dates, source capture times, or ordering semantics cannot be independently documented, mark temporal evaluation **not applicable**.

### Optional, conditional: external dataset evaluation

No external dataset is selected in this plan. Before using one, identify its source, license/redistribution terms, collection period, label definitions, and whether raw URLs are available. Define a defensible mapping to `1 = phishing`, deduplicate overlap with PhiUSIIL, and recompute only the approved eight features through the canonical extractor. Document missing/invalid URL handling without silent row removal. Freeze all decisions before evaluating the external set; do not tune or select the model against it. If no suitable, legally usable, compatible dataset is established, report that external evaluation was not performed.

## 5. Metrics and threshold protocol

Use **phishing = positive class 1** consistently. For each partition and experiment, report at minimum:

- phishing precision, phishing recall, and phishing F1;
- PR-AUC (average precision), with the class prevalence stated;
- confusion matrix with labels/orientation explicitly shown (TN legitimate, FP legitimate flagged phishing, FN phishing missed, TP phishing detected);
- false-positive rate and false-negative rate;
- accuracy as a secondary summary, not the selection criterion.

ROC-AUC may be included as a secondary ranking metric. Report support counts, missing/invalid records and any prespecified exclusions, and uncertainty intervals where feasible. For grouped domain evaluations, calculate uncertainty at the domain/group level rather than treating every row as independent. Do not invent a minimum acceptable score.

Keep 0.50 as the reference threshold for baseline comparability. For any proposed operational threshold, define the decision objective with the project owner before examining the final test results—for example, a specified acceptable false-positive rate or an explicit false-positive/false-negative cost trade-off. Evaluate the candidate thresholds on validation data only, record the selected rule and resulting validation metrics, freeze it, then report the test metrics once. If no operational constraint is agreed, show validation trade-off curves/tables and state that no deployment threshold is selected. Do not calibrate or interpret `predict_proba` as a calibrated real-world risk without a separate calibration study.

## 6. Reproducibility requirements

For each later experiment, preserve a machine-readable record of:

- UCI dataset ID and source, exact CSV SHA-256, row count, label mapping, and any external dataset provenance/license if used;
- Git commit, Python and library versions, operating environment, and canonical feature schema version `1.0.0`;
- exact model feature names/order, extractor source revision, split/group key definition, allocation algorithm, random seed, and realized partition counts;
- fixed Random Forest configuration, fitting partition, threshold-selection rule, and validation/test separation;
- duplicate URL, near-duplicate, feature-vector collision, mixed-label, and cross-partition overlap summaries;
- all requested metrics, class counts, confusion matrix, uncertainty method, and any failed/skipped experiment;
- model artifact SHA-256 and a distinct artifact name for any separately fitted evaluation copy.

Do not overwrite the existing candidate artifact. Preserve raw data unchanged. Tests should confirm split disjointness for the relevant group key and label/metric semantics. Reports must distinguish measurements from proposed or unavailable checks.

## 7. Acceptance criteria for reporting results

Results may be reported as a **reproduced PhiUSIIL random-URL-group baseline** only when the dataset hash, feature order, seed/split allocation, class counts, and key metrics are reproducible and discrepancies are explained.

Claims about **unseen domains** require a domain-disjoint test with zero registered-domain group overlap between partitions, a documented domain parsing rule, no test-guided threshold/model selection, and complete phishing-positive metrics. Claims about **future campaigns** additionally require a defensible chronological dataset and provenance. Claims about **external generalization** require a suitable independent dataset, documented licensing and label mapping, overlap checks, and a frozen one-time evaluation.

No evaluation result alone supports claims of universal detection, guaranteed safety, production readiness, or a calibrated risk score. If group overlap, contradictory labels, provenance, or test-set separation cannot be resolved, report the limitation and narrow the claim.

## 8. Explicit exclusions and unresolved questions

This plan does not run experiments, retrain or tune the Random Forest, alter its threshold, alter the feature schema, make feature-selection decisions, remove duplicate rows, modify labels, change SHAP/TI/correlation behavior, select an external dataset, or introduce a numeric risk score.

Questions to resolve before later evaluation:

1. Is an unseen-domain generalization claim required for the project, and should grouping use exact hosts or registrable domains with a pinned suffix list?
2. What operational false-positive/false-negative trade-off, if any, should guide a future threshold? No value is assumed here.
3. Is temporal provenance available from the dataset provider or an independently documented source? Current project documentation does not establish it.
4. Is there an external URL dataset with usable labels, compatible URL records, and reviewed licensing/redistribution terms? None is selected here.
5. Who will approve the data/model redistribution and attribution terms before any external dataset or model artifact is published?
6. How should the project describe risk assessment: retain the implemented qualitative correlation, or separately specify and evaluate a numeric score in a future phase?

