"""Loading and column helpers for the ClaMP PE dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV = DATA / "ClaMP_Integrated-5184.csv"

LABEL = "class"                 # 1 = malware, 0 = benign
CATEGORICAL = ["packer_type"]   # the one non-numeric feature (packer family)


def load() -> pd.DataFrame:
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV} not found. Run: python -m src.download_data"
        )
    return pd.read_csv(CSV)


def split_features_label(df: pd.DataFrame):
    """Return (X, y, feature_names)."""
    y = df[LABEL].astype(int)
    X = df.drop(columns=[LABEL])
    return X, y, list(X.columns)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != LABEL and c not in CATEGORICAL]
