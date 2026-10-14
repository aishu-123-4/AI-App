"""Load the trained model and classify URLs — importable + CLI."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrained(
            f"No model at {MODEL_PATH}. Run: python -m src.train"
        )
    return joblib.load(MODEL_PATH)


def classify(url: str) -> dict:
    """Return a prediction dict for a single URL."""
    bundle = _load_model()
    pipe = bundle["pipeline"]
    feats = extract_features(url)
    vec = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
    proba = float(pipe.predict_proba(vec)[0, 1])
    label = int(proba >= 0.5)
    return {
        "url": url,
        "label": "phishing" if label else "legit",
        "is_phishing": bool(label),
        "phishing_probability": round(proba, 4),
        "features": feats,
    }


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m src.predict <url> [<url> ...]", file=sys.stderr)
        return 2
    try:
        for url in argv[1:]:
            r = classify(url)
            flag = "PHISHING" if r["is_phishing"] else "legit   "
            print(f"[{flag}] p={r['phishing_probability']:.3f}  {url}")
    except ModelNotTrained as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
