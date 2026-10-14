"""Retrieve the most relevant code chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics.pairwise import cosine_similarity

from .indexer import CodeChunk


@dataclass
class Hit:
    chunk: CodeChunk
    score: float


def search(index: dict, query: str, k: int = 5) -> list[Hit]:
    """Return the top-k chunks ranked by cosine similarity to the query."""
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks: list[CodeChunk] = index["chunks"]

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    # Argsort descending, keep only non-zero-similarity hits.
    order = sims.argsort()[::-1][:k]
    return [Hit(chunks[i], float(sims[i])) for i in order if sims[i] > 0.0]
