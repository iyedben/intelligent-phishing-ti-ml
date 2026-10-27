"""Manually run one URL analysis with ML and configured TI providers.

Example: python scripts/test_threat_intel.py "https://example.com/"
The submitted URL is sent only as provider lookup data; this script never opens it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.threat_intel import (  # noqa: E402
    CorrelationEngine,
    MLPrediction,
    OpenPhishProvider,
    PhishTankProvider,
    URLhausProvider,
)
from src.threat_intel.base import MemoryTTLCache  # noqa: E402
from src.xai import URLShapExplainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="absolute HTTP(S) URL to analyze")
    parser.add_argument(
        "--without-openphish",
        action="store_true",
        help="skip downloading and checking the optional free OpenPhish text feed",
    )
    args = parser.parse_args()

    try:
        xai = URLShapExplainer.from_artifact()
        ml_result = xai.explain_url(args.url)
        ml = MLPrediction(
            prediction=ml_result["prediction"],
            phishing_probability=ml_result["phishing_probability"],
            threshold=ml_result["threshold"],
            explanation_metadata={
                "method": ml_result["explanation"]["method"],
                "explained_class": ml_result["explanation"]["explained_class"],
                "top_contributions": ml_result["explanation"]["contributions"][:3],
            },
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Could not run the local ML model: {exc}", file=sys.stderr)
        return 2

    # Reusing these objects keeps their process-local caches effective if this
    # CLI is extended to accept a batch; current invocation performs one lookup.
    cache = MemoryTTLCache()
    providers = [PhishTankProvider(cache=cache), URLhausProvider(cache=cache)]
    if not args.without_openphish:
        providers.append(OpenPhishProvider())
    provider_results = []
    for provider in providers:
        # Each provider returns structured unavailable/error data instead of
        # raising for network, authentication, HTTP, or response failures.
        provider_results.append(provider.check_url(args.url))

    assessment = CorrelationEngine().assess(args.url, ml, provider_results)
    print(json.dumps(assessment.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
