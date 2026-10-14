# 03 · LogSentry — SSH Log Anomaly Detector  📜🚨

An **ML × Security** project that reads real **`sshd` auth logs** and flags
anomalous / brute-forcing source IPs — **without any labels**, using
unsupervised machine learning cross-checked against a transparent rule.

This is the kind of triage a SOC analyst wants: "out of the thousands of IPs in
today's logs, which 6 should I actually look at?"

## What it does

1. **Parse** raw `sshd` log lines into structured events (timestamp, PID,
   source IP, attempted user, normalized event type).
2. **Aggregate** events per source IP into behavioral features (failed
   passwords, invalid-user attempts, distinct usernames tried, fail ratio,
   event rate, session duration…).
3. **Detect** anomalies two ways:
   - **IsolationForest** (unsupervised ML) — scores how far each IP deviates
     from normal behavior.
   - **Rule-based brute-force flag** — ≥20 failed attempts *or* ≥5 distinct
     usernames tried (interpretable, auditable).
4. **Report** the most anomalous IPs and how well the two detectors agree.

## Results (on the bundled real log)

The dataset is a real OpenSSH log from a server under active attack (~2,000
lines, 30 distinct source IPs, ~4 hours). With no labels at all:

| Metric | Value |
|--------|-------|
| Source IPs analyzed | 30 |
| IsolationForest anomalies | 6 |
| Rule-based brute-forcers | 6 |
| **Agreement (both flagged)** | **5 / 6** |

Top flagged IPs (excerpt):

| IP | total events | failed pw | invalid user | distinct users | anomaly score | ML | rule |
|----|-------------:|----------:|-------------:|---------------:|--------------:|:--:|:----:|
| 187.141.143.180 | 349 | 51 | 29 | 28 | -0.70 | ✅ | ✅ |
| 183.62.140.253  | 867 | 277 | 9 | 10 | -0.68 | ✅ | ✅ |
| 103.99.0.122    | 172 | 11 | 35 | 19 | -0.65 | ✅ | ✅ |
| 5.188.10.180    | 47  | 1  | 16 | 6 | -0.46 | ✅ | ✅ |
| 112.95.230.3    | 80  | 24 | 2 | 3 | -0.45 | ✅ | ✅ |

### Why the two detectors are both useful

Because the logs are **unlabeled**, there's no accuracy to report — so instead
we validate the ML detector against the interpretable rule, and the
*disagreements* are the interesting part:

- **183.62.140.253** — 867 events, 277 failed passwords: an obvious
  password-spray attacker. Both detectors agree.
- **IF-only flag** (an IP the rule missed) shows IsolationForest catching
  behavior that's *statistically* odd without tripping the fixed threshold —
  the value of ML over static rules.
- **Rule-only flag** shows the rule catching a textbook brute-forcer whose
  feature profile wasn't extreme enough for IF at this contamination setting —
  the value of keeping an interpretable backstop.

That combination — ML for coverage, rules for auditability — is exactly how
real detection pipelines are built.

## Quickstart

```bash
cd 03-logsentry-ssh-anomaly
pip install -r requirements.txt

python -m src.download_data      # fetch the real OpenSSH log into data/
python -m src.detect             # print the anomaly report
python -m src.detect /path/to/your/auth.log   # analyze your own sshd log
```

## Project layout

```
03-logsentry-ssh-anomaly/
├── README.md
├── requirements.txt
├── data/                 # downloaded log (git-ignored)
└── src/
    ├── download_data.py
    ├── parse.py          # regex log parser -> Event objects
    ├── features.py       # per-IP behavioral feature table
    └── detect.py         # IsolationForest + rule + report
tests/
└── test_parse.py         # 8 tests (synthetic lines, no download needed)
```

## Tests

```bash
python -m pytest -q      # 8 tests
```

## Dataset & credit

OpenSSH logs from **Loghub** (LogPAI). Real `sshd` logs from a host under
brute-force attack, used here for research/education.

## Possible extensions

- Time-windowed detection (per-hour feature snapshots) for streaming alerts.
- GeoIP / ASN enrichment of flagged IPs.
- Sequence models (log-template mining + Markov / LSTM) for rarer attacks.
- Auto-generate `fail2ban` / firewall rules from the flagged set.

> **Scope:** analyze only logs from systems you own or are authorized to
> monitor.
