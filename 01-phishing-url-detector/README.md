# 01 · Phishing URL Detector  🛡️🤖

An **ML × Security** project: detect phishing URLs from their text alone, with
no network request. It extracts interpretable *lexical* features from a URL,
trains a Random Forest classifier, and serves predictions through both a CLI
and a REST API.

This is the kind of lightweight model that runs inside an email gateway or
browser extension to flag suspicious links in real time.

## Why this project

- **Security relevance:** phishing is the #1 initial-access vector in real
  breaches. Classifying URLs is a classic, practical detection task.
- **ML lifecycle:** feature engineering → training → honest evaluation
  (cross-validation, confusion matrix, ROC-AUC) → persistence → serving.
- **Engineering:** clean package layout, a CLI, a REST API, and a test suite.

## Results

Trained on a balanced, reproducible dataset of **4,000 URLs** (2,000 legit /
2,000 phishing), 20 lexical features, 75/25 train/test split:

| Metric | Score |
|--------|-------|
| Accuracy | **0.995** |
| Precision | 0.994 |
| Recall | 0.996 |
| F1 | 0.995 |
| ROC-AUC | 0.9999 |
| Cross-val F1 (5-fold) | 0.9948 ± 0.002 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[497, 3], [2, 498]]`

**Most informative features** (Gini importance): `num_subdomains`,
`digit_ratio`, `num_dots`, `num_digits`, `path_length`, `suspicious_tld`,
`num_suspicious_keywords`, `hostname_entropy`. Note the signal is spread across
many features — no single feature trivially separates the classes, which is
what we want from a robust detector.

> The dataset is **synthesized deterministically** from documented phishing
> patterns (see *Dataset* below). It is realistic enough to be non-trivial but
> is **educational** — for production you would retrain on a labeled
> URL-reputation feed. The code path is identical.

## How it works

```
URL ──► features.py ──► 20-dim vector ──► StandardScaler ──► RandomForest ──► P(phishing)
```

**Lexical features** (`src/features.py`) include: URL / hostname / path length,
counts of `.`, `-`, `@`, `?`, `&`, `=`, digits; digit ratio; number of
subdomains; whether the host is a raw IP; HTTPS; non-standard port; known URL
shortener; suspicious TLD; count of urgency/brand keywords
(`login`, `verify`, `account`, …); Shannon entropy of the hostname; and
punycode (`xn--`) presence. These map to well-known phishing tells — e.g. the
`@` trick (`paypal.com@evil.xyz`), IP-literal hosts, and lookalike domains.

## Quickstart

```bash
cd 01-phishing-url-detector
pip install -r requirements.txt

# 1. Generate the dataset (deterministic; writes data/urls.csv)
python -m src.dataset --n-per-class 2000

# 2. Train + evaluate (writes models/phishing_model.joblib + metrics.json)
python -m src.train

# 3a. Classify URLs from the command line
python -m src.predict "http://192.168.1.10:8080/bank/confirm.php?id=x9a" \
                      "https://docs.python.org/3/library/"

# 3b. …or serve a REST API
python app.py   # http://127.0.0.1:5000
```

Example CLI output:

```
[PHISHING] p=0.996  http://192.168.1.10:8080/bank/confirm.php?id=x9a
[legit   ] p=0.281  https://docs.python.org/3/library/
```

### API

```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"urls": ["https://www.google.com", "http://185.3.2.1/wallet/unlock"]}'
```

```json
{"results": [
  {"url": "https://www.google.com", "label": "legit", "is_phishing": false, "phishing_probability": 0.0001},
  {"url": "http://185.3.2.1/wallet/unlock", "label": "phishing", "is_phishing": true, "phishing_probability": 0.5905}
]}
```

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/health` | GET | — | service + model status |
| `/predict` | POST | `{"url": "..."}` | one prediction |
| `/predict` | POST | `{"urls": [...]}` | batch predictions |

## Project layout

```
01-phishing-url-detector/
├── README.md
├── requirements.txt
├── app.py                  # Flask REST API
├── data/urls.csv           # generated dataset (git-ignored)
├── models/                 # saved model + metrics (git-ignored)
└── src/
    ├── features.py         # URL → feature vector
    ├── dataset.py          # reproducible dataset generator
    ├── train.py            # train + evaluate + persist
    └── predict.py          # load model + classify (CLI)
tests/
├── test_features.py
└── test_dataset.py
```

## Tests

```bash
python -m pytest -q      # 17 tests
```

## Dataset

`src/dataset.py` generates a balanced set from:

- **Legitimate:** popular registered domains with realistic subdomains and
  paths, *plus* a share of modern-TLD services (`.app`, `.io`, `.xyz`) and
  noisy-but-benign query strings, so a modern TLD or a long path is **not** a
  phishing giveaway.
- **Phishing:** seven documented attack patterns — IP-literal hosts, lookalike
  domains (`amaz0n`, `secure-paypal`), brand-stuffed subdomains, URL shorteners,
  high-entropy random hosts, the `@` redirect trick, and deep random paths.
  Crucially, ~45% use *common* TLDs (`.com`, `.net`) because real phishing does
  too — this removes easy shortcuts and keeps the task honest.

Deterministic given `--seed`, so results reproduce exactly.

## Model card (summary)

- **Intended use:** triage/ranking signal for suspicious URLs; a first-pass
  filter, not a sole authority.
- **Not suitable for:** final block decisions without a human or secondary
  check; URLs whose maliciousness depends on page *content* (lexical features
  can't see that).
- **Known limitations & false negatives:** borderline lookalikes on common TLDs
  (e.g. `secure-paypal.verify-account.tk`) can land near the 0.5 threshold. In a
  security setting, missing phishing (a false negative) is usually costlier than
  a false alarm, so you would **lower the decision threshold** to raise recall,
  or combine this with domain-age / reputation signals.
- **Ethics:** defensive use only; the generated data contains no real victim
  data.

## Possible extensions

- Threshold tuning / precision-recall curve for a recall-first operating point.
- Add host-based features (domain age via WHOIS, DNS, TLS cert age).
- Swap the Random Forest for gradient boosting; compare with `scikit-learn`'s
  `HistGradientBoostingClassifier`.
- SHAP explanations per prediction for analyst-facing tooling.
