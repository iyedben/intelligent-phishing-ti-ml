"""Compare deterministic URL features with PhiUSIIL on a fixed stratified sample."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_SCHEMA, extract_url_features


DATASET = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
RANDOM_STATE = 2026
SAMPLE_PER_CLASS = 5_000
TARGET = "label"


def _classify(feature: str, mismatches: pd.DataFrame) -> tuple[str, str]:
    """Classify observed differences without treating unknown formulas as matches."""
    if mismatches.empty:
        return "MATCH", "Every sampled value is exactly equal."

    urls = mismatches["URL"].astype(str)
    expected = mismatches["expected"]
    actual = mismatches["actual"]

    if feature == "NoOfQMarkInURL":
        edge = urls.str.endswith("?") & ((actual - expected) == 1)
        if bool(edge.all()):
            return "MINOR/EXPECTED DIFFERENCE", "All mismatches are a terminal '?' counted literally by this extractor."
    if feature == "NoOfEqualsInURL":
        edge = urls.str.endswith("=") & ((actual - expected) == 1)
        if bool(edge.all()):
            return "MINOR/EXPECTED DIFFERENCE", "All mismatches are a terminal '=' counted literally by this extractor."
    if feature == "IsDomainIP":
        embedded_ipv4 = urls.map(
            lambda url: bool(re.search(r"(?:\d{1,3}\.){3}\d{1,3}", url))
        )
        if bool(embedded_ipv4.all()):
            return (
                "MINOR/EXPECTED DIFFERENCE",
                "UCI flags an IPv4-looking hostname label; this extractor requires the entire parsed hostname to be a valid IP literal.",
            )
    return (
        "UNRESOLVED",
        "The UCI value differs and its field-level formula/preprocessing is not sufficiently documented to explain it safely.",
    )


def main() -> None:
    if not DATASET.is_file():
        raise FileNotFoundError(f"Official UCI CSV not found: {DATASET}")

    data = pd.read_csv(DATASET)
    needed = {"URL", TARGET, *(feature.phiusiil_column for feature in FEATURE_SCHEMA)}
    missing_columns = sorted(needed - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    # Deterministic class-stratified sample so both source labels are represented.
    sample = pd.concat(
        [
            group.sample(n=min(SAMPLE_PER_CLASS, len(group)), random_state=RANDOM_STATE)
            for _, group in data.groupby(TARGET, sort=True)
        ],
        ignore_index=False,
    ).sort_index()

    extracted_rows = [extract_url_features(url) for url in sample["URL"].astype(str)]
    extracted = pd.DataFrame(extracted_rows, index=sample.index)
    rows: list[dict[str, object]] = []
    total_equal = 0
    total_cells = 0

    print(f"Dataset: {DATASET.relative_to(ROOT)}")
    print(
        f"Validation sample: {len(sample):,} rows ({SAMPLE_PER_CLASS:,} per class when available); "
        f"random_state={RANDOM_STATE}"
    )
    print("Comparison uses raw CSV fields and the extractor's explicit definitions; no rows are modified.")

    for feature in FEATURE_SCHEMA:
        source_column = feature.phiusiil_column
        assert source_column is not None
        actual = extracted[feature.name]
        expected = sample[source_column]
        equal = actual.eq(expected).fillna(False)
        mismatch_indices = sample.index[~equal]
        total_equal += int(equal.sum())
        total_cells += len(sample)

        if feature.python_type is int:
            actual_numeric = pd.to_numeric(actual, errors="raise").astype("int64")
            expected_numeric = pd.to_numeric(expected, errors="raise").astype("int64")
            absolute_difference = (actual_numeric - expected_numeric).abs()
            mean_abs = float(absolute_difference.mean())
            median_abs = float(absolute_difference.median())
            max_abs = int(absolute_difference.max())
        else:
            mean_abs = median_abs = None
            max_abs = None

        mismatch_frame = pd.DataFrame(
            {
                "URL": sample.loc[mismatch_indices, "URL"],
                "expected": expected.loc[mismatch_indices],
                "actual": actual.loc[mismatch_indices],
            },
            index=mismatch_indices,
        )
        status, explanation = _classify(feature.name, mismatch_frame)
        rows.append(
            {
                "feature": feature.name,
                "exact_matches": int(equal.sum()),
                "mismatches": int((~equal).sum()),
                "exact_match_pct": float(equal.mean() * 100),
                "mismatch_pct": float((~equal).mean() * 100),
                "mean_absolute_difference": mean_abs,
                "median_absolute_difference": median_abs,
                "max_absolute_difference": max_abs,
                "status": status,
                "explanation": explanation,
            }
        )

    report = pd.DataFrame(rows)
    print("\nPer-feature comparison (numeric absolute differences are in source feature units):")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(
        f"\nExact value matches: {total_equal:,}/{total_cells:,} "
        f"({total_equal / total_cells * 100:.4f}%)"
    )
    status_counts = report["status"].value_counts()
    print("Feature-level classifications:")
    for status in ("MATCH", "MINOR/EXPECTED DIFFERENCE", "UNRESOLVED"):
        count = int(status_counts.get(status, 0))
        print(f"  {status}: {count}/{len(report)} ({count / len(report) * 100:.2f}%)")


if __name__ == "__main__":
    main()
