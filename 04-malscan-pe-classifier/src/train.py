"""Train a malware classifier on ClaMP PE-header features.

Model: HistGradientBoostingClassifier (fast, strong gradient boosting for
tabular data) inside a Pipeline that one-hot encodes the `packer_type` feature
and passes numeric features through untouched (tree models don't need scaling).

Evaluation: held-out test split + 5-fold CV, with accuracy, precision, recall,
F1, ROC-AUC, confusion matrix, and permutation feature importance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, load, numeric_columns, split_features_label

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "malscan_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def build_pipeline(numeric: list[str], seed: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", numeric),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=None,
        l2_regularization=1.0, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_evaluate(seed: int = 42) -> dict:
    df = load()
    X, y, feats = split_features_label(df)
    numeric = numeric_columns(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    pipe = build_pipeline(numeric, seed)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    secs = time.time() - t0

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_f1 = cross_val_score(build_pipeline(numeric, seed), X, y, cv=cv, scoring="f1")

    # Permutation importance on the test set (model-agnostic, honest).
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=seed, scoring="f1"
    )
    top = sorted(zip(X.columns, perm.importances_mean),
                 key=lambda kv: kv[1], reverse=True)[:12]

    metrics = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_balance": {"benign_0": int((y == 0).sum()),
                          "malware_1": int((y == 1).sum())},
        "train_seconds": round(secs, 2),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_std": round(float(cv_f1.std()), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "top_features": [[n, round(float(v), 4)] for n, v in top],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": feats}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 62)
    print("MALSCAN — PE malware classifier (ClaMP)")
    print("=" * 62)
    print(f"Samples: {metrics['n_samples']}  Features: {metrics['n_features']}  "
          f"Balance: {metrics['class_balance']}")
    print(f"Train time: {metrics['train_seconds']}s\n")
    print(classification_report(y_test, y_pred, target_names=["benign", "malware"]))
    print(f"ROC-AUC:        {metrics['roc_auc']}")
    print(f"CV F1 (5-fold): {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")
    print(f"Confusion [[TN, FP],[FN, TP]]: {metrics['confusion_matrix']}\n")
    print("Top features (permutation importance):")
    for name, val in metrics["top_features"]:
        print(f"  {name:24} {val}")
    print(f"\nModel saved to {MODEL_PATH.relative_to(ROOT)}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
