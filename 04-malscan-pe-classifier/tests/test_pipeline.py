"""Tests for MalScan that don't require the dataset download.

We build a tiny synthetic frame that mimics the ClaMP schema (a packer_type
categorical + numeric features + a `class` label) and verify the pipeline
trains, predicts, and handles unseen categories.
"""

import numpy as np
import pandas as pd

from src.data import CATEGORICAL, LABEL, numeric_columns, split_features_label
from src.train import build_pipeline


def _toy_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Two separable numeric signals + a categorical packer.
    y = np.array([0, 1] * (n // 2))
    f1 = y * 5 + rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    packers = np.where(y == 1, "UPX", "NoPacker")
    return pd.DataFrame({
        "packer_type": packers,
        "feat1": f1,
        "feat2": f2,
        LABEL: y,
    })


def test_split_features_label():
    df = _toy_frame()
    X, y, feats = split_features_label(df)
    assert LABEL not in X.columns
    assert set(feats) == {"packer_type", "feat1", "feat2"}
    assert y.tolist().count(1) == 30


def test_numeric_columns_excludes_categorical_and_label():
    df = _toy_frame()
    num = numeric_columns(df)
    assert "packer_type" not in num
    assert LABEL not in num
    assert set(num) == {"feat1", "feat2"}


def test_pipeline_fits_and_predicts():
    df = _toy_frame()
    X, y, _ = split_features_label(df)
    pipe = build_pipeline(numeric_columns(df))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    # Should learn this trivially separable toy problem well.
    assert (preds == y).mean() > 0.9


def test_pipeline_handles_unseen_packer():
    df = _toy_frame()
    X, y, _ = split_features_label(df)
    pipe = build_pipeline(numeric_columns(df))
    pipe.fit(X, y)
    novel = X.iloc[[0]].copy()
    novel["packer_type"] = "SomeBrandNewPacker"
    # handle_unknown="ignore" means this must not raise.
    out = pipe.predict(novel)
    assert out.shape == (1,)
