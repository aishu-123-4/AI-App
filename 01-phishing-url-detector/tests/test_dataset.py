"""Tests for the dataset generator."""

from src.dataset import generate_dataset


def test_balanced_and_sized():
    rows = generate_dataset(n_per_class=100, seed=1)
    assert len(rows) == 200
    labels = [lbl for _, lbl in rows]
    assert labels.count(0) == 100
    assert labels.count(1) == 100


def test_deterministic_with_seed():
    a = generate_dataset(n_per_class=50, seed=7)
    b = generate_dataset(n_per_class=50, seed=7)
    assert a == b


def test_urls_are_unique():
    rows = generate_dataset(n_per_class=200, seed=3)
    urls = [u for u, _ in rows]
    assert len(urls) == len(set(urls))


def test_urls_have_scheme():
    rows = generate_dataset(n_per_class=30, seed=5)
    assert all(u.startswith(("http://", "https://")) for u, _ in rows)
