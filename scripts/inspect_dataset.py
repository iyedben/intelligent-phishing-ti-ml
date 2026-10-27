"""Inspect the locally acquired official UCI PhiUSIIL CSV without modifying it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
TARGET_CANDIDATES = ("label", "Label", "target", "Target", "class", "Class")
IDENTIFIER_CANDIDATES = ("FILENAME", "filename", "id", "ID", "index")
MAX_ENUMERATED_VALUES = 1_000


def find_target(columns: pd.Index) -> str | None:
    """Find the dataset target using UCI's known target name, then common names."""
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def main() -> None:
    if not DEFAULT_DATASET.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DEFAULT_DATASET}\n"
            "Acquire UCI dataset 967 and place its original CSV in data/raw/."
        )

    frame = pd.read_csv(DEFAULT_DATASET)
    print(f"Dataset: {DEFAULT_DATASET.relative_to(ROOT)}")
    print(f"Shape: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
    print("\nColumn names (in source order):")
    for column in frame.columns:
        print(f"  {column}")

    print("\nData types:")
    print(frame.dtypes.to_string())

    missing = frame.isna().sum()
    print("\nMissing values by column:")
    print(missing.to_string())
    print(f"Total missing cells: {int(missing.sum()):,}")

    duplicate_rows = int(frame.duplicated().sum())
    print(f"\nDuplicate rows (all columns): {duplicate_rows:,}")
    if "URL" in frame.columns:
        duplicate_url_rows = int(frame["URL"].duplicated().sum())
        duplicate_url_keys = int(frame.loc[frame["URL"].duplicated(keep=False), "URL"].nunique())
        print(f"Duplicate URL rows (repeated occurrences beyond first): {duplicate_url_rows:,}")
        print(f"Distinct URL values appearing more than once: {duplicate_url_keys:,}")
    else:
        print("Duplicate URL rows: not evaluated (no URL column)")

    target = find_target(frame.columns)
    if target is None:
        print("\nTarget column: not identified; inspect the UCI variable metadata.")
    else:
        print(f"\nTarget column: {target}")
        counts = frame[target].value_counts(dropna=False).sort_index()
        percentages = frame[target].value_counts(dropna=False, normalize=True).sort_index() * 100
        print("Class counts and percentages:")
        for value, count in counts.items():
            print(f"  {value!r}: {int(count):,} ({percentages.loc[value]:.4f}%)")

    categorical = frame.select_dtypes(include=["object", "category", "str"]).columns
    print("\nCategorical column unique values:")
    for column in categorical:
        values = frame[column].drop_duplicates().tolist()
        print(f"  {column}: {len(values):,} unique values")
        if len(values) <= MAX_ENUMERATED_VALUES:
            print(f"    values: {values!r}")
        else:
            print(f"    first 10 source-order examples (full list omitted): {values[:10]!r}")

    constants = [column for column in frame if frame[column].nunique(dropna=False) <= 1]
    print(f"\nConstant columns: {constants if constants else 'none'}")

    identifiers = [column for column in IDENTIFIER_CANDIDATES if column in frame.columns]
    # URL is the submitted indicator and is useful for URL-only deployment, but is
    # also a high-cardinality identifier that can memorize samples if used raw.
    for column, note in (
        ("URL", "high-cardinality sample identifier; also the input indicator"),
        ("Domain", "high-cardinality site identifier"),
        ("Title", "high-cardinality page-content identifier"),
    ):
        if column in frame.columns:
            identifiers.append(f"{column} ({note})")
    print("Suspicious identifier / non-feature columns:")
    print("  " + ("\n  ".join(identifiers) if identifiers else "none identified by name"))

    print(f"\nMemory usage (deep, in-memory DataFrame): {int(frame.memory_usage(deep=True).sum()):,} bytes")


if __name__ == "__main__":
    main()
