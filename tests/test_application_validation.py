"""Focused offline tests for the running-application validation utility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from scripts import validate_running_application as validation
from scripts.validate_running_application import (
    OFFLINE_CORRELATION_STATUS,
    ResponseValidationError,
    OfflineModeError,
    calculate_metrics,
    map_application_prediction,
    map_source_label,
    normalize_local_base_url,
    sample_test_partition,
    validate_api_response,
    write_reports,
)


def test_label_mappings_are_explicit_and_phishing_positive() -> None:
    assert map_source_label(0) == 1
    assert map_source_label(1) == 0
    assert map_application_prediction("phishing") == 1
    assert map_application_prediction("legitimate") == 0


def test_sampling_is_balanced_deterministic_and_test_only() -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "urls": pd.Series([f"https://host-{i}.example/path" for i in range(12)]),
            "source_labels": pd.Series([0, 1] * 6, dtype="int8"),
            "y": pd.Series([1, 0] * 6, dtype="int8"),
        },
    )()
    partitions = {"train": np.array([0, 1]), "validation": np.array([2, 3]), "test": np.arange(4, 12)}

    first = sample_test_partition(dataset, partitions, samples_per_class=3, seed=2026)
    second = sample_test_partition(dataset, partitions, samples_per_class=3, seed=2026)

    assert first == second
    assert len(first) == 6
    assert sum(row["expected_target"] == 1 for row in first) == 3
    assert sum(row["expected_target"] == 0 for row in first) == 3
    assert all(row["row_index"] in set(partitions["test"]) for row in first)


def test_sampling_rejects_insufficient_class_examples() -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "urls": pd.Series(["https://one.example/a", "https://two.example/b"]),
            "source_labels": pd.Series([0, 0], dtype="int8"),
            "y": pd.Series([1, 1], dtype="int8"),
        },
    )()
    with pytest.raises(ValueError, match="legitimate"):
        sample_test_partition(dataset, {"test": np.array([0, 1])}, samples_per_class=1, seed=2026)


def test_metrics_use_confusion_matrix_order_tn_fp_fn_tp() -> None:
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1])

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["specificity"] == 0.5


def test_api_response_validation_checks_prediction_threshold_class_and_offline_ti() -> None:
    response = type(
        "Response",
        (),
        {
            "status_code": 200,
            "json": lambda self: {
                "url": "https://example.test/",
                "ml": {
                    "prediction": "phishing",
                    "phishing_probability": 0.8,
                    "threshold": 0.5,
                    "positive_class": 1,
                },
                "threat_intelligence": [],
                "correlation": {"status": OFFLINE_CORRELATION_STATUS},
            },
        },
    )()

    parsed = validate_api_response(response, "https://example.test/")
    assert parsed["prediction"] == "phishing"
    assert parsed["predicted_target"] == 1


def test_api_response_rejects_invalid_and_provider_bearing_results() -> None:
    invalid = type("Response", (), {"status_code": 200, "json": lambda self: {"ml": {}}})()
    with pytest.raises(ResponseValidationError):
        validate_api_response(invalid, "https://example.test/")

    provider_bearing = type(
        "Response",
        (),
        {
            "status_code": 200,
            "json": lambda self: {
                "url": "https://example.test/",
                "ml": {
                    "prediction": "legitimate",
                    "phishing_probability": 0.1,
                    "threshold": 0.5,
                    "positive_class": 1,
                },
                "threat_intelligence": [{"source": "urlhaus"}],
                "correlation": {"status": "ml_only_no_ti_match"},
            },
        },
    )()
    with pytest.raises(OfflineModeError):
        validate_api_response(provider_bearing, "https://example.test/")


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"])
def test_base_url_accepts_loopback_only(base_url: str) -> None:
    assert normalize_local_base_url(base_url).startswith("http")


@pytest.mark.parametrize("base_url", ["https://example.com", "http://192.0.2.1:8000", "not-a-url"])
def test_base_url_rejects_non_loopback_hosts(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        normalize_local_base_url(base_url)


def test_cli_requires_explicit_offline_ti_confirmation() -> None:
    assert validation.main(["--samples-per-class", "1"]) == 2


def test_report_writer_creates_json_and_safe_identifier_csv(tmp_path: Path) -> None:
    summary = {"metrics": {"precision": 1.0}, "failed_requests": 0}
    rows = [{
        "row_index": 4,
        "url_sha256": "a" * 64,
        "submitted_url": "https://secret.example/",  # writer must omit this field
        "success": True,
        "prediction": "legitimate",
    }]

    json_path, csv_path = write_reports(summary, rows, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["precision"] == 1.0
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "url_sha256" in csv_text
    assert "secret.example" not in csv_text


@pytest.mark.parametrize("error", [requests.Timeout("timeout"), requests.ConnectionError("refused")])
def test_run_validation_stops_cleanly_on_local_request_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "urls": pd.Series(["https://a.example/", "https://b.example/" ]),
            "source_labels": pd.Series([0, 1], dtype="int8"),
            "y": pd.Series([1, 0], dtype="int8"),
            "X": pd.DataFrame({"URLLength": [1, 1]}),
        },
    )()
    monkeypatch.setattr(validation, "dataset_sha256", lambda _: validation.EXPECTED_DATASET_SHA256)
    monkeypatch.setattr(validation, "build_dataset", lambda _: dataset)
    monkeypatch.setattr(validation, "domain_grouped_three_way_split", lambda *args, **kwargs: {
        "train": np.array([], dtype=int),
        "validation": np.array([], dtype=int),
        "test": np.array([0, 1], dtype=int),
    })
    monkeypatch.setattr(validation, "_artifact_identity", lambda: {"path": "test", "threshold": 0.5})

    class FailingSession:
        def post(self, *args: object, **kwargs: object) -> None:
            raise error

    summary, rows = validation.run_validation(
        base_url="http://127.0.0.1:8000",
        samples_per_class=1,
        seed=2026,
        timeout=1.0,
        output_dir=tmp_path,
        session=FailingSession(),
    )

    assert summary["sampling"]["failed"] == 1
    assert summary["stop_reason"] in {"application_unreachable"}
    assert rows[0]["success"] is False
