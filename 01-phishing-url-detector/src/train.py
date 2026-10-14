"""Train and evaluate the phishing URL classifier.

Pipeline:
  1. Load (or generate) the labeled URL dataset.
  2. Convert each URL to a lexical feature vector.
  3. Train a RandomForest inside a scikit-learn Pipeline (scaler + model).
  4. Evaluate on a held-out test set: accuracy, precision, recall, F1,
     ROC-AUC, confusion matrix, and cross-validated F1.
  5. Persist the fitted pipeline + feature names + metrics with joblib.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import generate_dataset
from .features import FEATURE_NAMES, features_to_vector

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "urls.csv"
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def _load_rows() -> list[tuple[str, int]]:
    if DATA_PATH.exists():
        import csv

        with DATA_PATH.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [(r["url"], int(r["label"])) for r in reader]
    # Fall back to generating a fresh dataset.
    return generate_dataset()


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_evaluate(random_state: int = 42) -> dict:
    rows = _load_rows()
    urls = [u for u, _ in rows]
    y = np.array([lbl for _, lbl in rows])
    X = np.array([features_to_vector(u) for u in urls], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    pipe = build_pipeline(random_state)

    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_secs = time.time() - t0

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_f1 = cross_val_score(
        build_pipeline(random_state), X, y, cv=cv, scoring="f1"
    )

    # Feature importances from the underlying forest.
    forest = pipe.named_steps["clf"]
    importances = sorted(
        zip(FEATURE_NAMES, forest.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )

    metrics = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "train_seconds": round(train_secs, 3),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_std": round(float(cv_f1.std()), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "top_features": [[n, round(float(v), 4)] for n, v in importances[:10]],
    }

    # Persist model + metadata.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "feature_names": list(FEATURE_NAMES), "metrics": metrics},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 60)
    print("PHISHING URL DETECTOR — EVALUATION")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']}  Features: {metrics['n_features']}")
    print(f"Train time: {metrics['train_seconds']}s")
    print()
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))
    print(f"ROC-AUC:        {metrics['roc_auc']}")
    print(f"CV F1 (5-fold): {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print()
    print("Top features:")
    for name, val in metrics["top_features"]:
        print(f"  {name:28} {val}")
    print(f"\nModel saved to {MODEL_PATH.relative_to(ROOT)}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
