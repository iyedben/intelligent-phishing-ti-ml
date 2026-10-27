# Phase 6 — SHAP explainability results

## Scope and model

This phase explains the existing Phase 5 research candidate without fitting, tuning, or modifying it. The loaded artifact is `models/url_only_random_forest_seed2026_candidate.joblib`, a joblib bundle with `artifact_status: research_candidate_not_production`. Its estimator is a fitted `sklearn.ensemble.RandomForestClassifier` with 200 trees, trained in Phase 5. The artifact's positive class is model label `1 = phishing`; label `0 = legitimate`.

The canonical feature schema is version `1.0.0`. The exact model input order is:

`URLLength`, `DomainLength`, `IsDomainIP`, `NoOfDegitsInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `IsHTTPS`.

`Domain` remains an extractor output but is not a numeric model input. The explainer verifies the artifact's feature names, schema version, estimator input count, and any fitted feature names before initializing SHAP. Local explanations use `src/features/url_features.py`; no URL is contacted.

## SHAP configuration and output semantics

- SHAP version: `0.54.1`.
- Explainer: `shap.TreeExplainer`.
- Configuration: `model_output="raw"`, `feature_perturbation="tree_path_dependent"`, with the saved forest's training-path counts as the reference distribution.
- Output selected: class index corresponding to `estimator.classes_ == 1` (phishing); the code locates this index from the fitted model rather than assuming output order.
- For this scikit-learn `RandomForestClassifier`, SHAP returns class-specific outputs with shape `(rows, features, classes)`. The selected class-1 output and its SHAP contributions are in probability units. The base value plus the class-1 contributions reconstructs `predict_proba(...)[class 1]`.

The official [TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) describes tree-path-dependent reference values, output spaces, and multi-output classifier shapes. Actual additivity was also checked against this loaded artifact.

## Evaluation sample

The original dataset and Phase 5 split are reconstructed using the URL-only extractor and the recorded group allocator with seed `2026`. The held-out test partition contains 35,368 rows (15,142 phishing and 20,226 legitimate). It is used only for explanations; it is not used to fit or tune the model.

For global summaries, the notebook takes 300 rows per class from the test partition (`random_state=2026`), for 600 rows total. This deterministic stratified sample keeps the summary plots readable and explanation runtime manageable. No rows are removed from the source data. Maximum absolute additivity error across the 600 rows was `3.62e-14`.

## Global SHAP ranking

Mean absolute SHAP values below are for class `1 = phishing` on the 600-row sample. Values measure average contribution magnitude in the model's phishing-probability output space; they do not say whether a feature always pushes toward phishing.

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | `IsHTTPS` | 0.222788 |
| 2 | `URLLength` | 0.174947 |
| 3 | `DomainLength` | 0.109532 |
| 4 | `NoOfDegitsInURL` | 0.094459 |
| 5 | `NoOfQMarkInURL` | 0.006345 |
| 6 | `NoOfEqualsInURL` | 0.003404 |
| 7 | `NoOfAmpersandInURL` | 0.000789 |
| 8 | `IsDomainIP` | 0.000034 |

The full-precision ranking is stored in [`global_shap_importance.csv`](../reports/xai/global_shap_importance.csv). Plots are [`shap_summary_beeswarm.png`](../reports/xai/shap_summary_beeswarm.png) and [`shap_summary_bar.png`](../reports/xai/shap_summary_bar.png).

This ranking differs in meaning from Phase 5's built-in Random Forest feature importance. Built-in feature importance is a model-level summary based on tree split behavior. SHAP assigns per-row additive contributions to a specified output, then the ranking here averages their absolute magnitudes. Neither is causal, and correlated inputs can share or shift attribution.

## Local examples

Examples are the first correctly classified phishing and legitimate rows in the deterministic test sample. The displayed prediction threshold is the Phase 5 baseline `0.5`; it is not proposed as a production risk threshold.

### Phishing example

Dataset URL: `https://page-review-0901513.firebaseapp.com/`  
Predicted phishing probability: `1.000000`  
SHAP base value: `0.428266`; contribution sum: `0.571734`; reconstructed probability: `1.000000`.

Largest contributions toward phishing were `URLLength` (`+0.271763`), `NoOfDegitsInURL` (`+0.267867`), and `DomainLength` (`+0.084408`). `IsHTTPS` contributed toward legitimate (`-0.050187`) for this particular prediction. This is model behavior, not evidence that HTTPS is unsafe or safe.

### Legitimate example

Dataset URL: `https://www.bojeux.com`  
Predicted phishing probability: `0.005580`  
SHAP base value: `0.428266`; contribution sum: `-0.422686`; reconstructed probability: `0.005580`.

Largest contributions toward legitimate were `URLLength` (`-0.328981`), `IsHTTPS` (`-0.161993`), and `NoOfDegitsInURL` (`-0.054293`). `DomainLength` contributed toward phishing (`+0.129372`) while the overall model output remained low.

The reusable interface is `URLShapExplainer.from_artifact().explain_url(url)`. Its JSON-compatible result includes the model prediction, phishing probability, ordered numeric features, class-1 base value, per-feature SHAP value and direction, and an additivity error. Human-readable statements say a feature “contributed toward” one class; they do not claim cause.

## Consistency and validation

The additive check is performed for each explained row:

`class-1 base value + sum(class-1 SHAP values) = model phishing probability`

The notebook's maximum absolute error across the global sample was `3.62e-14`. The local phishing and legitimate rows reconstructed with errors `2.66e-15` and `1.70e-15`, respectively. Probability values were read from the loaded estimator, not created or hard-coded.

Tests load the existing artifact, initialize `TreeExplainer`, verify schema order and output dimensions, compare predictions with the 0.5 decision rule, check repeatability and JSON serialization, validate additivity, and block socket connection/DNS calls during URL explanation.

## Limitations

1. SHAP explains the fitted model, not the real-world cause of phishing.
2. Correlated or redundant features can make individual attribution difficult or unstable across explanation assumptions.
3. The model was trained on PhiUSIIL using this project's URL-only extractor and feature representation.
4. Explanations inherit limitations of the underlying model, labels, dataset collection, and feature definitions.
5. URL-only inputs cannot observe webpage content or behavior.
6. SHAP does not verify whether a URL is actually malicious.
7. Strong performance on PhiUSIIL does not establish universal real-world performance.
8. Tree-path-dependent SHAP uses the model's training-path distribution and does not resolve feature dependence into a causal interpretation.
9. A positive `IsHTTPS` or any other contribution is specific to this model and sample. HTTPS presence alone is not a safety guarantee.

## Reproducibility and status

Run the complete notebook from the repository root with the project's Python environment:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 ml/notebooks/03_xai_shap.ipynb
```

The notebook records its outputs and writes the ranking and plots beneath `reports/xai/`. Runtime used for validation: Python `3.12.10`, SHAP `0.54.1`, scikit-learn `1.9.1`, pandas `3.1.0`, and NumPy `2.5.4`. Dataset SHA-256 is the Phase 5 artifact value `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`.

The XAI utilities can be called from a future service layer, subject to review of the candidate status and the limitations above. This phase does not implement or integrate FastAPI, and it does not establish a final production threshold.
