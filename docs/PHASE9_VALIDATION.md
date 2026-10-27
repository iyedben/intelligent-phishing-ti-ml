# Phase 9: Application Hardening, End-to-End Validation, and Reproducibility

## 1. Objective

Validate the existing Phase 4–8C application across API, feature extraction, saved-model inference, SHAP, provider result handling, frontend serving, and correlation. Preserve the current model and evidence semantics; do not train or fetch submitted destinations.

## 2. Architecture tested

The tests exercise the existing FastAPI route and `AnalysisService`. A model-backed integration test loads the existing Random Forest joblib bundle through `URLShapExplainer.from_artifact()`, extracts the canonical URL-only feature vector, runs class-1 prediction and SHAP, queries three deterministic fake providers, applies the existing `CorrelationEngine`, and validates the Pydantic response. The candidate was not retrained or modified.

The actual local candidate loaded successfully as a `RandomForestClassifier`. Its artifact metadata and the canonical schema agree on eight numeric model features in schema order, positive class `1 = phishing`, and schema version `1.0.0`. The artifact file is intentionally ignored by Git; a fresh checkout needs the same validated artifact supplied separately.

## 3. Security boundary

The submitted destination is treated as a URL string. Feature extraction uses URL parsing and literal IP parsing only; it does not resolve hostnames. The only production network operations are the fixed provider/feed clients. PhishTank and URLhaus use finite timeouts and disable redirects; OpenPhish reads its fixed community-feed URL and locally compares exact URL strings. A provider no-match is not represented as safe.

The model-backed SSRF tests submitted loopback, localhost, link-local metadata, and private IPv4 URL strings while replacing Requests calls, `urllib` open methods, socket DNS functions, and socket connection creation with immediate failures. The requests completed through local inference and fake providers without triggering any guarded network/DNS function. This verifies those code paths against the covered primitives; it is not a formal sandbox or proof about every possible future dependency.

## 4. End-to-end tests

The added `tests/test_phase9_end_to_end.py` covers real saved-model and SHAP execution with fake provider results for:

- phishing and legitimate predictions;
- provider no-match and all-provider unavailable;
- confirmed PhishTank phishing evidence and explicit model/TI disagreement;
- URLhaus malware-distribution evidence kept distinct from phishing;
- provider exception isolation and safe API 500 handling;
- exact HTTP and HTTPS URL preservation;
- empty, malformed, unsupported-scheme, and over-limit input;
- the 4,096-character limit (accepted at the limit, rejected at 4,097).

Existing tests also cover frontend static serving, `/health`, `/docs`, OpenAPI, no-match versus unavailable semantics, sanitized errors, lazy service reuse, mocked provider responses, and credential isolation. The complete suite contains **87 tests** after Phase 9 additions; final status is recorded after running the full suite.

## 5. SSRF/network safety tests

The five reserved/private destination strings tested are `127.0.0.1`, `localhost`, `169.254.169.254`, `10.0.0.1`, and `192.168.1.1`. They are sent only to the in-process test API. Provider doubles record the string; no request is made to these destinations. Guarded Requests, urllib, DNS, and socket connection APIs fail the test if called.

Static review found frontend `fetch()` calls only for same-origin `/health` and `/api/v1/analyze`; the submitted URL is serialized as JSON and is never passed as a fetch destination. The code contains no `innerHTML`, `outerHTML`, `eval`, or `new Function` usage for returned or user-controlled content.

## 6. Credential security verification

- `.env` is excluded by `.gitignore`; `git check-ignore .env` confirmed it is ignored, and `.env` is not Git-tracked.
- `.env.example` contains only empty `PHISHTANK_APP_KEY` and `URLHAUS_AUTH_KEY` assignments.
- `tests/conftest.py` removes inherited provider-key environment values and redirects dotenv loading to a nonexistent temporary file for each test.
- Provider tests use fake keys and verify credentials do not appear in normalized results or logs. API tests check a fake credential is absent from output after failure.
- No `.env` contents or real credentials were read, printed, or modified during this phase.

## 7. API validation

The API accepts only absolute HTTP/HTTPS URLs with a hostname and rejects whitespace, malformed authorities/ports, unsupported schemes, and empty strings with HTTP 422. A concrete integration gap was found: the UI already limits the field to 4,096 characters, but the Pydantic request did not have a size bound. `AnalyzeRequest.url` now has a 4,096-character maximum, matching the existing UI. Unexpected analysis exceptions remain sanitized as HTTP 500; provider exceptions remain isolated as normalized provider errors. No model/provider loading occurs during app factory creation or `/health`; the existing analysis service factory is lazy and reused per app process.

## 8. Frontend validation

FastAPI integration tests verify `/` returns the dashboard, `/static/css/styles.css` and `/static/js/app.js` are served locally, and `/health`, `/docs`, and `/api/v1/analyze` remain available. Static frontend checks cover the 422, 500, network-error, empty-provider/empty-contribution, disabled-button/loading, and input-length paths, plus same-origin-only fetch targets and safe DOM rendering. Node.js was already installed in the environment; `node --check frontend/js/app.js` was used only as a syntax check. No browser automation framework was added. Interactive timing and accessibility behavior have not been re-tested in a browser as part of Phase 9; Phase 8C's manual browser test remains the available visual review.

## 9. Performance observations

One local in-process measurement used the real candidate and SHAP explainer with deterministic fake providers (therefore excluding provider network latency):

| Observation | Measured |
|---|---:|
| Lazy model + SHAP service initialization | 0.0712 s |
| First complete local analysis request | 0.1305 s |
| Repeated local analysis request | 0.0281 s |

`/health` left the service uninitialized, and the first analysis created it once. The repeated request reused the loaded model/explainer/provider instances. Values are a single run on the current Windows/Python environment, not performance guarantees. Live provider latency was not measured in this phase. Existing provider code can cache successful URLhaus/PhishTank results in process-local memory for five minutes when a cache is supplied; OpenPhish keeps its feed snapshot for 12 hours. Automated tests verify OpenPhish feed reuse with a mock; URLhaus/PhishTank cache TTL behavior was inspected in code but not separately measured here. No real provider calls were made for these timings.

## 10. Reproducibility checks

- Python runtime: **3.12.10**.
- Raw PhiUSIIL CSV is present locally and remains Git-ignored. Its computed SHA-256 is `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`, matching `docs/DATASET.md` and `ml/experiment_config.json`.
- Locally available candidate artifact SHA-256: `ba205afaa56fb532bd4ceea566eb441d6494c39e0a811e976732e76550b7debc`. The artifact is ignored by Git and is not present in a fresh clone unless supplied separately.
- Canonical numeric model feature count: **8**; feature schema version: **1.0.0**.
- Runtime/development dependencies were already installed in the project `.venv`; no package was added for Phase 9.
- `python -m pip check`, `python -m compileall src scripts tests`, `node --check frontend/js/app.js`, and `git diff --check` were run.
- A real local Uvicorn process served `/`, `/health`, `/docs`, and both static assets with HTTP 200. In-process API tests cover the model-backed analysis route. No Phase 9 live Threat Intelligence request was made. Earlier manual verification documented in `docs/THREAT_INTELLIGENCE.md` is historical and provider/network outcomes can change.

## 11. Test count

The full test suite result is **87 passed** (66 pre-Phase-9 tests plus 21 new Phase 9 coverage cases), subject to the final verification run. One existing warning is emitted by the installed Starlette `TestClient` integration about HTTPX deprecation; it does not fail tests. No dependency was added to silence it.

## 12. Known limitations

- The research candidate and raw dataset are not Git-tracked. Model-backed validation on a clean clone requires separately obtaining the exact model artifact; the real-model integration test skips if it is absent.
- External provider availability, rate limiting, account access, current feed contents, and live latency are not guaranteed by these tests. PhishTank previously returned HTTP 403; restricted environments have returned normalized request errors.
- Frontend behavior received static-source checks in Phase 9, not a new automated browser interaction run. Browser checks for delayed UI transitions are therefore partial.
- The measurement excludes external threat-intelligence latency and uses test doubles.
- The optional in-memory caches are process-local and not persistent or distributed. The URL cache has no explicit maximum entry count; expiry cleanup is lazy.
- Authentication, API rate limiting, distributed throttling, production deployment controls, and database/history are not implemented.
- A URL-only model cannot observe page content or runtime behavior. Model and provider data may be incomplete, stale, or inconsistent.

## 13. Remaining risks

Submitting a URL sends it to configured external TI sources; users should consider that disclosure before using the service. No-match is not proof of safety. Provider APIs or feeds may be unavailable or return changed formats. The locally ignored model artifact is a reproducibility dependency until an artifact distribution mechanism is chosen. The current local research application is not a production security control.

## 14. Final Phase 9 status

**Application hardening and local validation: complete for the audited scope.** The API input-size gap was closed, model-backed end-to-end and SSRF-boundary tests were added, documentation was corrected, and the final test/quality command results are recorded in the completion report. Live provider availability and automated browser interaction remain unverified in this phase.

## Model artifact GitHub Release preparation checklist

The release has not been published. These steps remain pending:

- [ ] Reconfirm the exact local artifact path, file size, and SHA-256 immediately before release.
- [ ] Review model provenance and obtain human review of applicable license, dataset attribution, and redistribution terms; permission has not been confirmed by this report.
- [ ] Create the approved GitHub Release and upload only `url_only_random_forest_seed2026_candidate.joblib` as the release asset.
- [ ] Download the uploaded asset and verify its SHA-256 against the release notes and README before announcing the release.
- [ ] Test download, checksum verification, dependency setup, and model loading from a clean checkout without the local ignored artifact.
- [ ] After the release exists, replace the README `<release-tag>` placeholder with the actual tag and verified asset URL.
