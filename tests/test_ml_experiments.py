"""Tests for group splits and phishing-positive metric definitions."""

import numpy as np
import pandas as pd

from src.ml.experiments import calculate_metrics, group_aware_three_way_split


def test_group_split_keeps_duplicate_urls_together_and_is_reproducible() -> None:
    urls = pd.Series([f"https://site-{group}.example/" for group in range(100) for _ in range(2)])
    y = pd.Series([group % 2 for group in range(100) for _ in range(2)], dtype="int8")
    X = pd.DataFrame({"feature": np.arange(len(y), dtype=np.int64)})

    first = group_aware_three_way_split(X, y, urls)
    second = group_aware_three_way_split(X, y, urls)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    url_sets = {name: set(urls.iloc[index]) for name, index in first.items()}
    assert not (url_sets["train"] & url_sets["validation"])
    assert not (url_sets["train"] & url_sets["test"])
    assert not (url_sets["validation"] & url_sets["test"])
    assert all(len(index) > 0 for index in first.values())


def test_metrics_define_phishing_as_positive_class() -> None:
    # Truth: phishing, legitimate, phishing, legitimate.
    # Predictions at 0.5: phishing, phishing, legitimate, legitimate.
    metrics = calculate_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]), 0.5)

    assert metrics["tp_phishing_correct"] == 1
    assert metrics["fp_legitimate_flagged_phishing"] == 1
    assert metrics["fn_phishing_missed"] == 1
    assert metrics["tn_legitimate_correct"] == 1
    assert metrics["precision_phishing"] == 0.5
    assert metrics["recall_phishing"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
