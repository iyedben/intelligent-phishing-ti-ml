"""Tests for group splits and phishing-positive metric definitions."""

import numpy as np
import pandas as pd
import pytest

from src.ml.experiments import (
    build_dataset,
    calculate_metrics,
    domain_group_key,
    domain_grouped_three_way_split,
    group_aware_three_way_split,
    normalized_url_key,
)


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


def test_source_labels_transform_to_phishing_positive_targets(tmp_path) -> None:
    dataset_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "URL": ["https://phish.example/", "https://legitimate.example/"],
            "label": [0, 1],
        }
    ).to_csv(dataset_path, index=False)

    dataset = build_dataset(dataset_path)

    assert dataset.source_labels.tolist() == [0, 1]
    assert dataset.y.tolist() == [1, 0]


def test_domain_group_key_handles_ip_localhost_and_single_label_hosts() -> None:
    assert domain_group_key("https://192.0.2.1/path") == "ip:192.0.2.1"
    assert domain_group_key("https://LOCALHOST/path") == "hostname:localhost"
    assert domain_group_key("https://intranet/path") == "hostname:intranet"


@pytest.mark.parametrize("url", ["", "example.com", "ftp://example.com/path", "https://bad host/"])
def test_domain_group_key_rejects_malformed_urls(url: str) -> None:
    with pytest.raises(ValueError):
        domain_group_key(url)


def test_domain_grouped_split_is_deterministic_and_isolates_domain_and_normalized_urls() -> None:
    urls = pd.Series(
        [
            "https://a.example.test/one",
            "https://sub.example.test/two",
            "https://b.example.test/one",
            "https://c.example.test/one",
            "https://d.example.test/one",
            "https://e.example.test/one",
            "https://f.example.test/one",
            "https://g.example.test/one",
            "https://h.example.test/one",
            "https://i.example.test/one",
            "https://j.example.test/one",
            "https://k.example.test/one",
        ]
    )
    X = pd.DataFrame({"feature": np.arange(len(urls), dtype=np.int64)})
    y = pd.Series([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype="int8")

    first = domain_grouped_three_way_split(X, y, urls)
    second = domain_grouped_three_way_split(X, y, urls)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert sorted(np.concatenate(list(first.values())).tolist()) == list(range(len(urls)))
    assert all(len(indices) > 0 for indices in first.values())
    assert sum(len(indices) for indices in first.values()) == len(urls)
    assert sum(int(y.iloc[indices].sum()) for indices in first.values()) == int(y.sum())
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_domains = {domain_group_key(urls.iloc[i]) for i in first[left]}
        right_domains = {domain_group_key(urls.iloc[i]) for i in first[right]}
        left_urls = {urls.iloc[i] for i in first[left]}
        right_urls = {urls.iloc[i] for i in first[right]}
        left_normalized = {normalized_url_key(urls.iloc[i]) for i in first[left]}
        right_normalized = {normalized_url_key(urls.iloc[i]) for i in first[right]}
        assert not left_domains & right_domains
        assert not left_urls & right_urls
        assert not left_normalized & right_normalized


def test_domain_grouped_split_keeps_mixed_label_domains_together() -> None:
    urls = pd.Series(
        [
            "https://mixed.example.test/phishing",
            "https://mixed.example.test/legitimate",
            "https://other-a.example.test/one",
            "https://other-b.example.test/one",
            "https://other-c.example.test/one",
            "https://other-d.example.test/one",
        ]
    )
    X = pd.DataFrame({"feature": np.arange(len(urls), dtype=np.int64)})
    y = pd.Series([1, 0, 1, 0, 1, 0], dtype="int8")

    partitions = domain_grouped_three_way_split(X, y, urls)
    mixed_parts = [name for name, indices in partitions.items() if set(indices) & {0, 1}]

    assert mixed_parts == ["train"]
