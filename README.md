# AI & Security Portfolio

A collection of end-to-end projects spanning **machine learning**, **cybersecurity**,
and the **intersection of the two**. Each project is self-contained, documented,
tested, and runnable, and is designed to demonstrate skills relevant to
ML/Data-Science, Security/AppSec, and Software-Engineering internships.

> Built and maintained by [@aishu-123-4](https://github.com/aishu-123-4).

## Projects

| # | Project | Domain | Dataset | Stack | Status |
|---|---------|--------|---------|-------|--------|
| 01 | [Phishing URL Detector](./01-phishing-url-detector) | ML × Security | synthetic (reproducible) | scikit-learn, Flask | ✅ Complete |
| 02 | [IntruDetect (NSL-KDD)](./02-intrudetect-nsl-kdd) | ML × Security | **NSL-KDD** | scikit-learn | ✅ Complete |
| 03 | [LogSentry (SSH anomalies)](./03-logsentry-ssh-anomaly) | ML × Security | Loghub OpenSSH | scikit-learn | ✅ Complete |
| 04 | [MalScan (PE malware)](./04-malscan-pe-classifier) | ML × Security | ClaMP PE headers | HistGradientBoosting | ✅ Complete |
| 05 | _SentimentScope (NLP)_ | ML / Data Science | IMDB / AG News | transformers | 🧭 Planned |
| 06 | _DocuMind (RAG Q&A)_ | AI / LLM | your docs | FAISS + Claude API | 🧭 Planned |
| 07 | _WebRecon (vuln scanner)_ | Security / AppSec | — | requests, CLI | 🧭 Planned |
| 08 | _CredCheck (breach auditor)_ | Security / AppSec | HIBP API | requests | 🧭 Planned |
| 09 | _RepoMind_ | AI / LLM × SWE | any repo | RAG over code | 🧭 Planned |
| 10 | _AI Security Pentest Agent_ | AI × Security | — | LLM agent | 🧭 Planned |

Each row links to a project folder with its own README, source, tests, and
instructions. The roadmap below explains what each planned project will cover.

## Roadmap

Chosen to cover all three target role families, weighted toward the
**ML × Security** intersection, while keeping each project small enough to
finish well:

- **01 · Phishing URL Detector** *(ML × Security)* ✅ — lexical URL feature
  engineering, a classifier with a full evaluation report, a CLI, and a REST API.
- **02 · IntruDetect** *(ML × Security)* ✅ — network intrusion detection on the
  **NSL-KDD** benchmark; binary + 4-class (DoS/Probe/R2L/U2R) with honest
  train→test generalization gap.
- **03 · LogSentry** *(ML × Security)* ✅ — unsupervised anomaly detection over
  **real `sshd` logs**, cross-checked against a rule-based brute-force detector.
- **04 · MalScan** *(ML × Security)* ✅ — static PE-header malware classification
  on **ClaMP** with gradient boosting and permutation importance.
- **05 · SentimentScope** *(ML / Data Science)* — NLP text classification with a
  TF-IDF baseline vs. a fine-tuned transformer, plus error analysis.
- **06 · DocuMind** *(AI / LLM)* — retrieval-augmented Q&A over your documents
  with chunking, embeddings, and citation-aware answers.
- **07 · WebRecon** *(Security / AppSec)* — defensive scanner for security
  headers, cookie flags, TLS, and exposed files against **authorized** targets.
- **08 · CredCheck** *(Security / AppSec)* — password strength + Have I Been
  Pwned k-anonymity breach check (password never leaves the machine).
- **09 · RepoMind** *(AI / LLM × SWE)* — ask natural-language questions about a
  code repository; RAG over source with symbol-aware chunking.
- **10 · AI Security Pentest Agent** *(AI × Security)* — an LLM agent that runs a
  guided, **authorized** recon/assessment workflow and writes up findings.

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
