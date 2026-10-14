# AI & Security Portfolio

A collection of end-to-end projects spanning **machine learning**, **cybersecurity**,
and the **intersection of the two**. Each project is self-contained, documented,
tested, and runnable, and is designed to demonstrate skills relevant to
ML/Data-Science, Security/AppSec, and Software-Engineering internships.

> Built and maintained by [@aishu-123-4](https://github.com/aishu-123-4).

## Projects

| # | Project | Domain | Stack | Status |
|---|---------|--------|-------|--------|
| 01 | [Phishing URL Detector](./01-phishing-url-detector) | ML × Security | scikit-learn, Flask, pytest | ✅ Complete |
| 02 | _Log Anomaly Detector_ | ML × Security | (planned) | 🧭 Planned |
| 03 | _RAG Document Q&A_ | AI / LLM | (planned) | 🧭 Planned |
| 04 | _Web Vulnerability Scanner_ | Security / AppSec | (planned) | 🧭 Planned |
| 05 | _Tabular ML + Model Card_ | ML / Data Science | (planned) | 🧭 Planned |

Each row links to a project folder with its own README, source, tests, and
instructions. The roadmap below explains what each planned project will cover.

## Roadmap

These are chosen to cover the three target role families while keeping each
project small enough to finish well:

- **01 · Phishing URL Detector** *(ML × Security)* — lexical feature engineering
  on URLs, a trained classifier with a full evaluation report, a CLI, and a REST
  API. Shows the complete ML lifecycle on a real security problem.
- **02 · Log Anomaly Detector** *(ML × Security)* — unsupervised anomaly
  detection over authentication / access logs to flag suspicious activity
  (e.g. brute-force bursts, off-hours access).
- **03 · RAG Document Q&A** *(AI / LLM)* — retrieval-augmented question
  answering over a document set, with chunking, embeddings, and a citation-aware
  answer layer. The "LLM app" employers increasingly ask about.
- **04 · Web Vulnerability Scanner** *(Security / AppSec)* — a defensive scanner
  for common misconfigurations (missing security headers, cookie flags, exposed
  files) against targets **you own or are authorized to test**.
- **05 · Tabular ML + Model Card** *(ML / Data Science)* — a clean classical-ML
  pipeline with cross-validation, calibration, feature importance, and a
  responsible-AI model card.

## Repository layout

```
AI-App/
├── README.md                     # this file
├── 01-phishing-url-detector/     # flagship project (complete)
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   ├── tests/
│   ├── data/
│   └── app.py
└── ...                           # further projects added over time
```

## Conventions

- **Python 3.11+**, standard `requirements.txt` per project.
- Every project has a `README.md` (what/why/how + results) and `tests/`.
- Datasets are either generated reproducibly or downloaded via a documented
  script — raw data is git-ignored, generators are committed.
- Security projects are strictly **defensive / educational** and must only be
  run against systems you own or are explicitly authorized to test.

## Ethics & scope

The security tooling here is for learning and for authorized testing only.
Nothing in this repository is intended to target systems without permission.
