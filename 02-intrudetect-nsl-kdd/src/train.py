"""Train intrusion-detection models on NSL-KDD.

Two tasks share the same feature pipeline:
  * ``binary``      — normal (0) vs attack (1)
  * ``multiclass``  — normal / DoS / Probe / R2L / U2R

We use the *official* NSL-KDD train/test split. The test set is drawn from a
different attack distribution than the train set (by design), so test accuracy
is meaningfully lower than train/CV accuracy — this is the well-known NSL-KDD
generalization gap, and reporting it honestly is the point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import CATEGORICAL, feature_columns, load_train_test

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def build_pipeline(numeric: list[str], categorical: list[str], seed: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _train_one(train_df, test_df, target: str, seed: int = 42) -> dict:
    feats = feature_columns()
    numeric = [c for c in feats if c not in CATEGORICAL]

    X_train, y_train = train_df[feats], train_df[target]
    X_test, y_test = test_df[feats], test_df[target]

    pipe = build_pipeline(numeric, CATEGORICAL, seed)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    secs = time.time() - t0

    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # .tolist() yields native Python ints/strs so metrics are JSON-serializable.
    labels = sorted(y_test.unique().tolist())
    report = classification_report(
        y_test, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "features": feats, "target": target, "labels": labels},
        MODEL_DIR / f"{target}_model.joblib",
    )

    metrics = {
        "target": target,
        "train_seconds": round(secs, 2),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "labels": labels,
        "confusion_matrix": cm,
        "per_class": {
            str(lbl): {
                "precision": round(float(report[str(lbl)]["precision"]), 4),
                "recall": round(float(report[str(lbl)]["recall"]), 4),
                "f1-score": round(float(report[str(lbl)]["f1-score"]), 4),
                "support": int(report[str(lbl)]["support"]),
            }
            for lbl in labels if str(lbl) in report
        },
    }

    print("=" * 64)
    print(f"NSL-KDD — {target.upper()}  (official train/test split)")
    print("=" * 64)
    print(f"Train accuracy: {metrics['train_accuracy']}  |  "
          f"Test accuracy: {metrics['test_accuracy']}  |  "
          f"Macro-F1: {metrics['test_macro_f1']}")
    print(f"Train time: {metrics['train_seconds']}s\n")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    return metrics


def train_all(seed: int = 42) -> dict:
    train_df, test_df = load_train_test()
    print(f"Loaded train={len(train_df):,}  test={len(test_df):,} rows\n")

    results = {
        "binary": _train_one(train_df, test_df, "is_attack", seed),
        "multiclass": _train_one(train_df, test_df, "category", seed),
    }
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nModels + metrics saved to {MODEL_DIR.relative_to(ROOT)}/")
    return results


if __name__ == "__main__":
    train_all()
