# Phase 13.1 — Running Application Validation

## Purpose and scope

`python scripts/validate_running_application.py` checks the running FastAPI
application against a deterministic balanced sample from the frozen Phase 11A
hostname-grouped **test** partition. It maps the source labels in memory
(`0 = phishing` to evaluation target `1`; `1 = legitimate` to `0`) and maps the
API prediction strings into the same target space.

This is application-integration verification, not a new independent model
evaluation and not a replacement for the complete Phase 11B benchmark. It uses
the existing `domain_grouped_three_way_split`, seed `2026`, dataset checksum,
canonical extractor, and selected Phase 11B model identity. Hostname grouping
is not registrable-domain/eTLD+1 isolation. A legitimate prediction does not
prove that a URL is safe.

## Threat-intelligence isolation

The repository's normal `create_app()` factory constructs PhishTank, URLhaus,
and OpenPhish providers. It has no environment-variable or command-line switch
for disabling all providers; the existing `--without-openphish` option applies
only to `scripts/test_threat_intel.py` and does not disable the FastAPI service.

For a no-network validation run, start the app by injecting the existing
`AnalysisService` with an empty provider sequence. From the repository root,
use this manual command in a separate terminal:

```powershell
python -c "import uvicorn; from src.api import create_app; from src.services.analysis_service import AnalysisService; from src.threat_intel.correlation import CorrelationEngine; from src.xai import URLShapExplainer; app=create_app(analysis_service=AnalysisService(explainer=URLShapExplainer.from_artifact(), providers=[], correlation_engine=CorrelationEngine())); uvicorn.run(app, host='127.0.0.1', port=8000)"
```

The regular documented factory command remains available for ordinary local
development, but it is not an offline validation mode:

```powershell
python -m uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

The validator requires `--confirm-offline-ti` as an operator confirmation. The
flag is not proof of server configuration. Each response must additionally have
an empty `threat_intelligence` list and correlation status
`ml_only_no_ti_providers`; otherwise the run stops and records an offline-mode
blocker rather than continuing.

## Usage

Default sample: 50 phishing and 50 legitimate URLs, selected without
replacement using seed `2026`:

```powershell
python scripts/validate_running_application.py `
  --base-url http://127.0.0.1:8000 `
  --samples-per-class 50 `
  --seed 2026 `
  --timeout 10 `
  --confirm-offline-ti
```

The endpoint is restricted to loopback hosts (`localhost`, `127.0.0.1`, or
`::1`), requests are sequential, redirects are not followed, and the script
never opens or crawls the submitted URLs. It does not start or stop the server.

Reports are written to `reports/application_validation/`:

- `application_validation_summary.json` — configuration, model identity,
  sample counts, failures, metrics, confusion matrix, and limitations;
- `application_validation_results.csv` — one attempted sample per row, using a
  SHA-256 URL identifier rather than printing full URLs.

The confusion matrix uses explicit label order `[0, 1]` and is documented as:

`[[TN, FP], [FN, TP]]`

Failed requests and invalid responses are excluded from successful-prediction
metrics and reported separately. Connection failures stop the run early with a
start-the-server instruction; provider-bearing responses stop the run as an
offline-mode blocker.
