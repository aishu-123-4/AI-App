"""Tests for NSL-KDD preprocessing that don't require the dataset download."""

import pandas as pd
import pytest

from src.preprocess import (
    ATTACK_CATEGORIES,
    CATEGORICAL,
    COLUMNS,
    _categorize,
    feature_columns,
    load_split,
)


def test_column_count():
    # 41 features + label + difficulty.
    assert len(COLUMNS) == 43


def test_categorize_normal_and_attacks():
    assert _categorize("normal") == "normal"
    assert _categorize("neptune") == "DoS"
    assert _categorize("satan") == "Probe"
    assert _categorize("guess_passwd") == "R2L"
    assert _categorize("buffer_overflow") == "U2R"


def test_categorize_unknown_label():
    assert _categorize("some_future_attack") == "unknown"


def test_all_four_categories_present():
    assert set(ATTACK_CATEGORIES.values()) == {"DoS", "Probe", "R2L", "U2R"}


def test_feature_columns_exclude_targets():
    feats = feature_columns()
    for excluded in ("label", "difficulty", "category", "is_attack"):
        assert excluded not in feats
    assert len(feats) == 41


def test_categoricals_are_features():
    feats = feature_columns()
    assert all(c in feats for c in CATEGORICAL)


def test_load_split_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_split("DoesNotExist+.txt")
