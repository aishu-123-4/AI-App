# 09 · RepoMind — Ask Questions About a Codebase  🧠📁

A **RAG (retrieval-augmented generation)** tool for source code. Point it at any
repository, ask a question in plain English, and get an answer grounded in the
actual code with **`path:line` citations**.

RepoMind works **fully offline** (TF-IDF retrieval, no model download, no key)
and **upgrades automatically** to Claude-generated answers when an
`ANTHROPIC_API_KEY` is present — the "key optional" design.

## Why it's interesting

- **Symbol-aware chunking:** Python files are split with the `ast` module so each
  chunk is a real function or class (citations point at definitions, not
  arbitrary line windows). Other languages use overlapping line windows.
- **Honest RAG architecture:** index → retrieve (cosine similarity over TF-IDF) →
  ground an answer in the retrieved context. Swapping TF-IDF for dense
  embeddings + FAISS is a one-module change (see Extensions).
- **Graceful degradation:** no API key → extractive answer (top chunks +
  citations); key present → Claude writes a grounded prose answer that cites files.

## Quickstart

```bash
cd 09-repomind
pip install -r requirements.txt

# Index any repository (saves to index/repo_index.joblib)
python -m src index /path/to/some/repo

# Ask a question (extractive mode — no key needed)
python -m src ask "how is authentication handled?" --no-llm

# With an API key, answers are generated and grounded:
export ANTHROPIC_API_KEY=sk-ant-...
python -m src ask "how is authentication handled?"
```

You can also index on the fly without saving:

```bash
python -m src ask "where are attack labels mapped?" --repo ../02-intrudetect-nsl-kdd --no-llm
```

### Example (extractive mode, indexing project 02 of this portfolio)

```
[retrieved 3 chunks | extractive mode]

Top matches for: "where are attack labels mapped to categories"
1. src/train.py:55 (_train_one)        [score 0.235]
2. README.md:1                         [score 0.113]
3. README.md:36                        [score 0.090]
    - **R2L and U2R are hard** (recall 0.07 / 0.19). These rare attack types ...
    - **DoS and Probe are strong** (F1 0.85 / 0.76) ...
```

With an API key, `answer.py` feeds those chunks to Claude, which replies in prose
and cites the files it used.

## How it works

```
repo ─► indexer.py ─► chunks (ast for .py, windows otherwise)
                         └─► TfidfVectorizer ─► sparse matrix  ──► index/*.joblib
query ─► retriever.py ─► cosine similarity ─► top-k Hits
         └─► answer.py ─► LLM (Claude, if key) or extractive (always)
```

- `src/indexer.py` — file discovery, symbol-aware chunking, TF-IDF index, save/load.
- `src/retriever.py` — cosine-similarity top-k search.
- `src/answer.py` — LLM-or-extractive answer with auto-detection.
- `src/__main__.py` — `index` / `ask` CLI.

Model for generated answers: `claude-opus-5-5` via the official `anthropic` SDK.
Credentials resolve from `ANTHROPIC_API_KEY` or an `ant auth login` profile — no
key is hardcoded.

## Project layout

```
09-repomind/
├── README.md
├── requirements.txt
├── index/                # saved index (git-ignored, *.joblib)
└── src/
    ├── indexer.py
    ├── retriever.py
    ├── answer.py
    └── __main__.py
tests/
└── test_repomind.py      # 7 tests, no API key required
```

## Tests

```bash
python -m pytest -q      # 7 tests
```

## Possible extensions

- **Dense retrieval:** swap `TfidfVectorizer` for `sentence-transformers` +
  FAISS behind the same `build_index` / `search` interface.
- **Hybrid search:** combine BM25 + dense with reciprocal-rank fusion.
- **Call-graph awareness:** follow imports/definitions to expand a chunk's context.
- **Incremental indexing:** re-index only files changed since the last `git` commit.
- **A small web UI** over the `ask` command.
