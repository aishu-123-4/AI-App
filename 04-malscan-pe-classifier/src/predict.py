"""Load the trained MalScan model and classify PE samples (CLI demo)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib

from .data import LABEL, load

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "malscan_model.joblib"


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrained(f"No model at {MODEL_PATH}. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def classify(df):
    """Classify a DataFrame of PE feature rows. Returns (preds, probabilities)."""
    bundle = _load_model()
    pipe = bundle["pipeline"]
    X = df[bundle["features"]]
    proba = pipe.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return preds, proba


def _demo(n: int = 12) -> int:
    try:
        df = load().sample(n, random_state=0).reset_index(drop=True)
        preds, proba = classify(df)
    except (ModelNotTrained, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    actual = df[LABEL].astype(int).tolist()
    print(f"{'actual':8} {'predicted':10} {'P(malware)':12} packer_type")
    print("-" * 56)
    for i in range(len(df)):
        a = "malware" if actual[i] else "benign"
        p = "malware" if preds[i] else "benign"
        mark = "OK " if actual[i] == preds[i] else "XX "
        print(f"{mark}{a:8} {p:10} {proba[i]:<12.4f} {df.loc[i, 'packer_type']}")
    acc = sum(int(a == p) for a, p in zip(actual, preds)) / len(df)
    print(f"\nSample accuracy: {acc:.0%} on {len(df)} random held-in rows")
    return 0


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    raise SystemExit(_demo(count))
