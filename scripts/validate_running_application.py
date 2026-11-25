"""Validate the running FastAPI ML application on a deterministic test sample.

This is an application-integration check, not a new model evaluation. It uses
only the frozen Phase 11A hostname-grouped test partition and requires the
operator to confirm that the running server has no threat-intelligence
providers configured.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import ipaddress
import json
from pathlib import Path
import csv
import math
import sys
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import joblib
import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.experiments import (  # noqa: E402
    DOMAIN_GROUPING_METHOD,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_NAMES,
    RANDOM_STATE,
    build_dataset,
    dataset_sha256,
    domain_grouped_three_way_split,
)
from src.xai.shap_explainer import DEFAULT_ARTIFACT_PATH  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
EXPECTED_DATASET_SHA256 = "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6"
EXPECTED_MODEL_SHA256 = "ceed3f58cedca8493a3a591a7ce2f3ce674a6f6e5fb8b70d42d35637f41793f4"
FROZEN_THRESHOLD = 0.50
POSITIVE_CLASS = 1
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "application_validation"
OFFLINE_CORRELATION_STATUS = "ml_only_no_ti_providers"
CSV_FIELDS = (
    "row_index",
    "url_sha256",
    "original_label",
    "expected_target",
    "http_status",
    "success",
    "prediction",
    "predicted_target",
    "phishing_probability",
    "threshold",
    "positive_class",
    "error",
)


class ResponseValidationError(ValueError):
    """Raised when the API response is missing or violates its ML contract."""


class OfflineModeError(ResponseValidationError):
    """Raised when the API response shows configured threat-intelligence providers."""


@dataclass(frozen=True, slots=True)
class ValidationSample:
    row_index: int
    url: str
    original_label: int
    expected_target: int


def map_source_label(label: int) -> int:
    """Map PhiUSIIL source labels (0 phishing, 1 legitimate) to target labels."""
    if int(label) == 0:
        return 1
    if int(label) == 1:
        return 0
    raise ValueError(f"unsupported source label: {label!r}")


def map_application_prediction(prediction: str) -> int:
    """Map the API's explicit prediction string to evaluation target space."""
    if prediction == "phishing":
        return 1
    if prediction == "legitimate":
        return 0
    raise ValueError(f"unsupported application prediction: {prediction!r}")


def normalize_local_base_url(base_url: str) -> str:
    """Accept only HTTP(S) loopback application addresses."""
    parsed = urlsplit(base_url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an HTTP(S) loopback URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("base URL must identify a loopback host")
    normalized = hostname.rstrip(".").casefold()
    is_loopback = normalized == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        pass
    if not is_loopback:
        raise ValueError("base URL must point to a loopback host")
    return base_url.rstrip("/")


def _url_identifier(url: str) -> str:
    return sha256(url.encode("utf-8")).hexdigest()


def sample_test_partition(
    dataset: Any,
    partitions: dict[str, np.ndarray],
    *,
    samples_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample a balanced, deterministic set exclusively from the test indices."""
    if samples_per_class < 1:
        raise ValueError("samples-per-class must be at least 1")
    if "test" not in partitions:
        raise ValueError("partitions must contain a test partition")
    test_indices = np.asarray(partitions["test"], dtype=np.int64)
    targets = np.asarray(dataset.y.iloc[test_indices], dtype=np.int8)
    selected_by_class: list[np.ndarray] = []
    generator = np.random.default_rng(seed)
    for target, label_name in ((1, "phishing"), (0, "legitimate")):
        candidates = test_indices[targets == target]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"test partition has only {len(candidates)} {label_name} examples; "
                f"cannot sample {samples_per_class} without replacement"
            )
        selected_by_class.append(np.sort(generator.choice(candidates, size=samples_per_class, replace=False)))
    selected = np.concatenate(selected_by_class)
    generator.shuffle(selected)

    samples: list[dict[str, Any]] = []
    test_set = set(int(index) for index in test_indices)
    for index in selected:
        row_index = int(index)
        if row_index not in test_set:
            raise RuntimeError("sampling escaped the frozen test partition")
        original_label = int(dataset.source_labels.iloc[row_index])
        expected_target = map_source_label(original_label)
        samples.append(
            {
                "row_index": row_index,
                "url": str(dataset.urls.iloc[row_index]),
                "original_label": original_label,
                "expected_target": expected_target,
            }
        )
    return samples


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, Any]:
    """Calculate phishing-positive metrics with explicit [0, 1] matrix ordering."""
    truth = np.asarray(list(y_true), dtype=np.int8)
    predicted = np.asarray(list(y_pred), dtype=np.int8)
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths")
    if len(truth) == 0:
        return {
            "total_successful": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "specificity": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "confusion_matrix_order": "[[TN, FP], [FN, TP]]",
            "limitations": ["no successful predictions"],
        }
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = float(tn / (tn + fp)) if tn + fp else None
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else None
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else None
    limitations: list[str] = []
    if len(set(truth.tolist())) < 2:
        limitations.append("successful predictions do not contain both ground-truth classes")
    return {
        "total_successful": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "specificity": specificity,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_order": "[[TN, FP], [FN, TP]]",
        "limitations": limitations,
    }


def validate_api_response(response: Any, submitted_url: str) -> dict[str, Any]:
    """Validate and normalize one successful API response without inferring labels."""
    if response.status_code != 200:
        raise ResponseValidationError(f"http_status_{response.status_code}")
    try:
        body = response.json()
    except Exception as exc:
        raise ResponseValidationError("malformed_json") from exc
    if not isinstance(body, dict):
        raise ResponseValidationError("response_body_not_object")
    if body.get("url") != submitted_url:
        raise ResponseValidationError("response_url_mismatch")
    ml = body.get("ml")
    if not isinstance(ml, dict):
        raise ResponseValidationError("missing_ml_object")
    prediction = ml.get("prediction")
    probability = ml.get("phishing_probability")
    threshold = ml.get("threshold")
    positive_class = ml.get("positive_class")
    if prediction not in {"phishing", "legitimate"}:
        raise ResponseValidationError("invalid_prediction")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ResponseValidationError("invalid_phishing_probability")
    if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        raise ResponseValidationError("invalid_phishing_probability")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isclose(float(threshold), FROZEN_THRESHOLD):
        raise ResponseValidationError("unexpected_threshold")
    if positive_class != POSITIVE_CLASS:
        raise ResponseValidationError("unexpected_positive_class")

    threat_intelligence = body.get("threat_intelligence")
    correlation = body.get("correlation")
    if threat_intelligence != [] or not isinstance(correlation, dict) or correlation.get("status") != OFFLINE_CORRELATION_STATUS:
        raise OfflineModeError("server response does not prove providers are disabled")
    return {
        "prediction": prediction,
        "predicted_target": map_application_prediction(prediction),
        "phishing_probability": float(probability),
        "threshold": float(threshold),
        "positive_class": int(positive_class),
    }


def write_reports(summary: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON summary and URL-identifier-only CSV records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "application_validation_summary.json"
    csv_path = output_dir / "application_validation_results.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _artifact_identity() -> dict[str, Any]:
    if not DEFAULT_ARTIFACT_PATH.is_file():
        raise FileNotFoundError(f"selected model artifact is missing: {DEFAULT_ARTIFACT_PATH}")
    actual_hash = sha256(DEFAULT_ARTIFACT_PATH.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(f"selected model SHA-256 mismatch: {actual_hash}")
    bundle = joblib.load(DEFAULT_ARTIFACT_PATH)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError("selected model artifact is not a recognized bundle")
    if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("selected model feature schema version does not match the application")
    if tuple(bundle.get("feature_names", ())) != tuple(MODEL_FEATURE_NAMES):
        raise ValueError("selected model feature order does not match the application schema")
    if bundle.get("positive_class") != {"value": 1, "meaning": "phishing"}:
        raise ValueError("selected model positive-class metadata is not 1=phishing")
    classes = tuple(int(value) for value in getattr(bundle["estimator"], "classes_", ()))
    if classes != (0, 1):
        raise ValueError(f"selected model classes are unexpected: {classes!r}")
    selected_threshold = bundle.get("metadata", {}).get("selection", {}).get("selected_threshold")
    if selected_threshold != FROZEN_THRESHOLD:
        raise ValueError("selected model metadata threshold is not 0.50")
    return {
        "path": str(DEFAULT_ARTIFACT_PATH),
        "expected_sha256": EXPECTED_MODEL_SHA256,
        "actual_sha256": actual_hash,
        "threshold": FROZEN_THRESHOLD,
        "positive_class": POSITIVE_CLASS,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "classes": list(classes),
    }


def run_validation(
    *,
    base_url: str,
    samples_per_class: int,
    seed: int,
    timeout: float,
    output_dir: Path,
    session: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run sequential local requests and persist a partial-safe validation report."""
    endpoint = f"{normalize_local_base_url(base_url)}/api/v1/analyze"
    artifact = _artifact_identity()
    actual_dataset_sha = dataset_sha256(DATASET_PATH)
    if actual_dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(f"dataset SHA-256 mismatch: {actual_dataset_sha}")
    dataset = build_dataset(DATASET_PATH)
    partitions = domain_grouped_three_way_split(dataset.X, dataset.y, dataset.urls, random_state=RANDOM_STATE)
    samples = sample_test_partition(dataset, partitions, samples_per_class=samples_per_class, seed=seed)

    client = session or requests.Session()
    rows: list[dict[str, Any]] = []
    successful_targets: list[int] = []
    predicted_targets: list[int] = []
    stop_reason: str | None = None
    for sample in samples:
        row: dict[str, Any] = {
            "row_index": sample["row_index"],
            "url_sha256": _url_identifier(sample["url"]),
            "original_label": sample["original_label"],
            "expected_target": sample["expected_target"],
            "http_status": None,
            "success": False,
            "prediction": None,
            "predicted_target": None,
            "phishing_probability": None,
            "threshold": None,
            "positive_class": None,
            "error": None,
        }
        try:
            response = client.post(
                endpoint,
                json={"url": sample["url"]},
                timeout=timeout,
                allow_redirects=False,
            )
            row["http_status"] = int(response.status_code)
            parsed = validate_api_response(response, sample["url"])
            row.update(parsed)
            row["success"] = True
            successful_targets.append(sample["expected_target"])
            predicted_targets.append(parsed["predicted_target"])
        except OfflineModeError as exc:
            row["error"] = str(exc)
            stop_reason = "offline_ti_not_verified"
            rows.append(row)
            break
        except (requests.ConnectionError, requests.Timeout) as exc:
            row["error"] = exc.__class__.__name__
            stop_reason = "application_unreachable"
            rows.append(row)
            break
        except (requests.RequestException, ResponseValidationError, ValueError, TypeError) as exc:
            row["error"] = str(exc)
        rows.append(row)

    metrics = calculate_metrics(successful_targets, predicted_targets)
    metrics["limitations"].extend([
        "sampled application-integration check; not a replacement for the complete Phase 11B benchmark",
        "hostname grouping is not registrable-domain/eTLD+1 isolation",
        "a legitimate prediction does not prove that a URL is safe",
    ])
    failed_count = sum(not bool(row["success"]) for row in rows)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_type": "application_integration_sample_not_independent_model_evaluation",
        "configuration": {
            "endpoint": endpoint,
            "base_url": normalize_local_base_url(base_url),
            "timeout_seconds": timeout,
            "dataset_path": str(DATASET_PATH),
            "dataset_sha256": actual_dataset_sha,
            "grouping_policy": DOMAIN_GROUPING_METHOD,
            "split_seed": RANDOM_STATE,
            "sampling_seed": seed,
            "samples_per_class_requested": samples_per_class,
            "sampled_test_partition_only": True,
            "threshold": FROZEN_THRESHOLD,
            "positive_class": POSITIVE_CLASS,
            "label_mapping": {
                "source_0": "phishing -> evaluation 1",
                "source_1": "legitimate -> evaluation 0",
                "api_phishing": "evaluation 1",
                "api_legitimate": "evaluation 0",
            },
            "offline_ti": {
                "confirmation_flag_required": True,
                "operator_confirmed": True,
                "response_status_required": OFFLINE_CORRELATION_STATUS,
                "verified_by_response": stop_reason != "offline_ti_not_verified" and any(row["success"] for row in rows),
                "note": "The flag is operator confirmation, not proof of server startup configuration.",
            },
        },
        "model": artifact,
        "sampling": {
            "requested_per_class": samples_per_class,
            "selected_total": len(samples),
            "selected_phishing": sum(sample["expected_target"] == 1 for sample in samples),
            "selected_legitimate": sum(sample["expected_target"] == 0 for sample in samples),
            "attempted": len(rows),
            "successful": len(successful_targets),
            "failed": failed_count,
        },
        "metrics": metrics,
        "failures": [
            {key: row[key] for key in ("row_index", "url_sha256", "http_status", "error")}
            for row in rows if not row["success"]
        ],
        "stop_reason": stop_reason,
        "exclusions": {"count": 0, "reasons": []},
    }
    write_reports(summary, rows, output_dir)
    return summary, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="local API base URL (loopback only)")
    parser.add_argument("--samples-per-class", type=int, default=50, help="number of test URLs per class")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="deterministic sampling seed")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="report output directory")
    parser.add_argument(
        "--confirm-offline-ti",
        action="store_true",
        help="confirm that the running server was started with all TI providers disabled; this is not auto-verifiable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_offline_ti:
        print(
            "Refusing to send requests: pass --confirm-offline-ti only after confirming "
            "the server has no configured threat-intelligence providers.",
            file=sys.stderr,
        )
        return 2
    if args.samples_per_class < 1 or args.timeout <= 0:
        print("--samples-per-class must be >= 1 and --timeout must be > 0", file=sys.stderr)
        return 2
    try:
        summary, _ = run_validation(
            base_url=args.base_url,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            timeout=args.timeout,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Validation could not start: {exc}", file=sys.stderr)
        print(
            "Start the local API first; the repository's documented command is "
            "python -m uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000",
            file=sys.stderr,
        )
        return 2
    print(json.dumps({
        "selected_total": summary["sampling"]["selected_total"],
        "attempted": summary["sampling"]["attempted"],
        "successful": summary["sampling"]["successful"],
        "failed": summary["sampling"]["failed"],
        "metrics": summary["metrics"],
        "stop_reason": summary["stop_reason"],
        "reports": [
            str(args.output_dir / "application_validation_summary.json"),
            str(args.output_dir / "application_validation_results.csv"),
        ],
    }, indent=2, allow_nan=False))
    return 0 if summary["sampling"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
