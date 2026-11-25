"""Focused regression tests for the Phase 11B hostname-grouped experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hostname_grouped_experiment import (
    GROUPING_METADATA,
    select_candidate,
    select_validation_threshold,
)
from src.ml.experiments import (
    DOMAIN_GROUPING_METHOD,
    MODEL_FEATURE_NAMES,
    hostname_grouped_cv_indices,
    make_baseline_models,
    phishing_probability,
)


class _ReversedProbabilityEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        phishing = np.asarray(X["phishing_probability"], dtype=float)
        return np.column_stack([phishing, 1.0 - phishing])


def test_phishing_probability_uses_class_label_not_array_position() -> None:
    estimator = _ReversedProbabilityEstimator()
    features = pd.DataFrame({"phishing_probability": [0.8, 0.2]})

    np.testing.assert_allclose(phishing_probability(estimator, features), [0.8, 0.2])


def test_hostname_cv_is_deterministic_and_isolates_hostname_groups() -> None:
    urls = pd.Series([
        f"https://host-{group}.example/path-{row}"
        for group in range(12)
        for row in range(2)
    ])
    y = pd.Series([group % 2 for group in range(12) for _ in range(2)], dtype="int8")

    first = hostname_grouped_cv_indices(y, urls, n_splits=3, random_state=2026)
    second = hostname_grouped_cv_indices(y, urls, n_splits=3, random_state=2026)

    for (first_train, first_valid), (second_train, second_valid) in zip(first, second):
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_valid, second_valid)
        train_hosts = {urls.iloc[index].split("/")[2] for index in first_train}
        valid_hosts = {urls.iloc[index].split("/")[2] for index in first_valid}
        assert not train_hosts & valid_hosts


def test_all_candidates_use_the_canonical_feature_schema() -> None:
    assert set(make_baseline_models()) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert MODEL_FEATURE_NAMES == (
        "URLLength",
        "DomainLength",
        "IsDomainIP",
        "NoOfDegitsInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "IsHTTPS",
    )


def test_threshold_selection_is_validation_only_and_deterministic() -> None:
    truth = np.array([1, 0, 0, 0])
    probabilities = np.array([0.51, 0.49, 0.10, 0.05])

    selected, table = select_validation_threshold(truth, probabilities)

    assert selected == 0.5
    assert len(table) == 9


def test_candidate_selection_uses_validation_and_cv_not_test_metrics() -> None:
    records = [
        {
            "model": "Random Forest",
            "cv_metrics": {"pr_auc_mean": 0.94},
            "validation_metrics": {
                "pr_auc": 0.95,
                "recall_phishing": 0.90,
                "precision_phishing": 0.99,
                "false_positive_rate": 0.01,
            },
            "test_metrics": {"pr_auc": 0.10},
        },
        {
            "model": "Logistic Regression",
            "cv_metrics": {"pr_auc_mean": 0.93},
            "validation_metrics": {
                "pr_auc": 0.94,
                "recall_phishing": 0.99,
                "precision_phishing": 0.90,
                "false_positive_rate": 0.10,
            },
            "test_metrics": {"pr_auc": 0.99},
        },
    ]

    assert select_candidate(records)["model"] == "Random Forest"


def test_metadata_labels_hostname_fallback_and_not_registrable_domain() -> None:
    assert GROUPING_METADATA["method"] == DOMAIN_GROUPING_METHOD
    assert "hostname" in GROUPING_METADATA["claim_boundary"]
    assert "not registrable-domain" in GROUPING_METADATA["claim_boundary"]
