"""Aggregate parsed sshd events into a per-source-IP feature table.

Each source IP becomes one row. The features summarize that IP's behavior in a
way that separates noisy brute-force attackers from benign clients.
"""

from __future__ import annotations

import pandas as pd

from .parse import Event

# Event types that represent a failed / hostile authentication attempt.
FAIL_TYPES = {"failed_password", "failed_invalid_user", "invalid_user",
              "auth_failure", "break_in", "no_ident"}

FEATURE_COLUMNS = [
    "total_events", "failed_password", "failed_invalid_user", "invalid_user",
    "auth_failure", "break_in", "accepted", "distinct_users", "fail_ratio",
    "duration_sec", "events_per_min",
]


def events_to_ip_table(events: list[Event]) -> pd.DataFrame:
    """Return a DataFrame indexed by source IP with behavioral features."""
    rows: dict[str, dict] = {}
    for ev in events:
        if not ev.ip:
            continue
        r = rows.setdefault(ev.ip, {
            "ip": ev.ip, "total_events": 0,
            "failed_password": 0, "failed_invalid_user": 0, "invalid_user": 0,
            "auth_failure": 0, "break_in": 0, "accepted": 0,
            "_users": set(), "_first": None, "_last": None,
        })
        r["total_events"] += 1
        if ev.event_type in r:
            r[ev.event_type] += 1
        if ev.user and ev.user not in ("unknown",):
            r["_users"].add(ev.user)
        if ev.ts is not None:
            r["_first"] = ev.ts if r["_first"] is None else min(r["_first"], ev.ts)
            r["_last"] = ev.ts if r["_last"] is None else max(r["_last"], ev.ts)

    records = []
    for r in rows.values():
        fails = sum(r[t] for t in FAIL_TYPES if t in r)
        duration = (
            (r["_last"] - r["_first"]).total_seconds()
            if r["_first"] and r["_last"] else 0.0
        )
        records.append({
            "ip": r["ip"],
            "total_events": r["total_events"],
            "failed_password": r["failed_password"],
            "failed_invalid_user": r["failed_invalid_user"],
            "invalid_user": r["invalid_user"],
            "auth_failure": r["auth_failure"],
            "break_in": r["break_in"],
            "accepted": r["accepted"],
            "distinct_users": len(r["_users"]),
            "fail_ratio": fails / r["total_events"] if r["total_events"] else 0.0,
            "duration_sec": round(duration, 1),
            "events_per_min": round(
                r["total_events"] / (duration / 60), 3
            ) if duration > 0 else float(r["total_events"]),
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.set_index("ip").sort_values("total_events", ascending=False)
    return df
