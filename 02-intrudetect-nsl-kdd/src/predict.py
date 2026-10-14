"""Load trained NSL-KDD models and classify connection records (CLI)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .preprocess import load_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load(target: str):
    path = MODEL_DIR / f"{target}_model.joblib"
    if not path.exists():
        raise ModelNotTrained(f"No model at {path}. Run: python -m src.train")
    return joblib.load(path)


def classify_frame(df: pd.DataFrame, target: str = "category") -> pd.DataFrame:
    """Return the input frame with a ``prediction`` column for the given target."""
    bundle = _load(target)
    preds = bundle["pipeline"].predict(df[bundle["features"]])
    out = df.copy()
    out["prediction"] = preds
    return out


def _demo(n: int = 10) -> int:
    """Classify the first n rows of the test split and show predictions."""
    try:
        df = load_split("KDDTest+.txt").head(n)
        res = classify_frame(df, "category")
    except ModelNotTrained as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"{'actual':10} {'predicted':10} {'proto':6} {'service':12} flag")
    print("-" * 50)
    for _, r in res.iterrows():
        mark = "OK " if r["category"] == r["prediction"] else "XX "
        print(f"{mark}{r['category']:10} {r['prediction']:10} "
              f"{r['protocol_type']:6} {r['service']:12} {r['flag']}")
    return 0


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    raise SystemExit(_demo(count))
