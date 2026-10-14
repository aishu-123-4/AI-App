"""Detect anomalous / brute-force source IPs from sshd logs.

Two complementary detectors:
  * **IsolationForest** — unsupervised ML that scores how much each IP's
    behavior deviates from the norm (no labels needed).
  * **Rule-based brute-force flag** — a transparent heuristic
    (many failed attempts and/or many distinct usernames tried).

The logs are unlabeled, so we validate the ML detector by checking that its
flagged IPs agree with the interpretable rule — a standard way to sanity-check
unsupervised security models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import FEATURE_COLUMNS, events_to_ip_table
from .parse import parse_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "data" / "OpenSSH.log"

# Rule-based brute-force thresholds.
BRUTE_FORCE_FAILS = 20       # total failed auth attempts
BRUTE_FORCE_USERS = 5        # distinct usernames tried


def rule_flag(df: pd.DataFrame) -> pd.Series:
    fails = (df["failed_password"] + df["failed_invalid_user"]
             + df["invalid_user"] + df["auth_failure"])
    return (fails >= BRUTE_FORCE_FAILS) | (df["distinct_users"] >= BRUTE_FORCE_USERS)


def score_anomalies(df: pd.DataFrame, contamination: float = 0.2,
                    seed: int = 42) -> pd.DataFrame:
    """Add ``anomaly_score`` and ``is_anomaly`` columns via IsolationForest.

    Lower (more negative) score = more anomalous.
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed
    )
    iso.fit(X)
    out = df.copy()
    out["anomaly_score"] = iso.score_samples(X).round(4)
    out["is_anomaly"] = (iso.predict(X) == -1)
    out["brute_force_rule"] = rule_flag(df)
    return out.sort_values("anomaly_score")


def analyze(log_path: str | Path = DEFAULT_LOG,
            contamination: float = 0.2) -> pd.DataFrame:
    events = parse_file(log_path)
    df = events_to_ip_table(events)
    if df.empty:
        raise ValueError(f"No IP events parsed from {log_path}")
    return score_anomalies(df, contamination)


def _report(log_path: str | Path = DEFAULT_LOG) -> int:
    if not Path(log_path).exists():
        print(f"Error: {log_path} not found. Run: python -m src.download_data")
        return 1
    result = analyze(log_path)

    total_ips = len(result)
    n_anom = int(result["is_anomaly"].sum())
    n_rule = int(result["brute_force_rule"].sum())
    agree = int((result["is_anomaly"] & result["brute_force_rule"]).sum())

    print("=" * 70)
    print("LogSentry — SSH anomaly report")
    print("=" * 70)
    print(f"Source IPs analyzed:        {total_ips}")
    print(f"IsolationForest anomalies:  {n_anom}")
    print(f"Rule-based brute-forcers:   {n_rule}")
    print(f"Agreement (both flagged):   {agree}/{max(n_anom, n_rule)}")
    print()
    cols = ["total_events", "failed_password", "failed_invalid_user",
            "invalid_user", "distinct_users", "anomaly_score",
            "is_anomaly", "brute_force_rule"]
    with pd.option_context("display.width", 140,
                           "display.max_columns", None):
        print("Top 12 most anomalous source IPs:")
        print(result[cols].head(12).to_string())
    return 0


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    raise SystemExit(_report(path))
