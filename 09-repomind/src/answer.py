"""Turn retrieved code chunks into an answer.

Two modes, chosen automatically:
  * **LLM mode** — if the `anthropic` package is installed and a credential is
    available (ANTHROPIC_API_KEY or an `ant auth login` profile), Claude writes
    a grounded answer from the retrieved context, citing files.
  * **Extractive mode** — otherwise, return the top chunks with their
    `path:line` citations and a short heuristic lead-in. No key required.

This makes RepoMind fully runnable offline while upgrading to generated answers
when a key is present.
"""

from __future__ import annotations

import os

from .retriever import Hit

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = (
    "You are RepoMind, a code-understanding assistant. Answer the user's "
    "question using ONLY the provided code excerpts. Cite the files you rely "
    "on as `path:line`. If the excerpts are insufficient, say so plainly "
    "rather than inventing details."
)


def llm_available() -> bool:
    """True if we can plausibly make an API call."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile may also provide credentials.
    cfg = os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(cfg)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] {h.chunk.citation}\n"
            f"```\n{h.chunk.text}\n```"
        )
    return "\n\n".join(blocks)


def answer_extractive(query: str, hits: list[Hit]) -> str:
    """Key-free answer: the most relevant excerpts with citations."""
    if not hits:
        return ("No relevant code found for that question. Try rephrasing or "
                "re-indexing the repository.")
    lines = [
        f'Top matches for: "{query}"',
        "(extractive mode — set ANTHROPIC_API_KEY for generated answers)\n",
    ]
    for i, h in enumerate(hits, 1):
        preview = h.chunk.text.strip().splitlines()
        snippet = "\n    ".join(preview[:12])
        lines.append(f"{i}. {h.chunk.citation}   [score {h.score:.3f}]")
        lines.append(f"    {snippet}")
        if len(preview) > 12:
            lines.append("    ...")
        lines.append("")
    return "\n".join(lines)


def answer_llm(query: str, hits: list[Hit]) -> str:
    """Generated answer grounded in the retrieved context."""
    import anthropic

    client = anthropic.Anthropic()
    context = _format_context(hits)
    user = (
        f"Question: {query}\n\n"
        f"Relevant code excerpts:\n\n{context}\n\n"
        "Answer the question, citing files as `path:line`."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in resp.content if b.type == "text"]
    return "\n".join(parts).strip()


def answer(query: str, hits: list[Hit], use_llm: bool | None = None) -> str:
    """Answer `query` from `hits`. `use_llm=None` auto-detects availability."""
    if use_llm is None:
        use_llm = llm_available()
    if use_llm:
        try:
            return answer_llm(query, hits)
        except Exception as exc:  # noqa: BLE001 — fall back gracefully
            return (f"[LLM call failed: {exc}. Falling back to extractive mode.]\n\n"
                    + answer_extractive(query, hits))
    return answer_extractive(query, hits)
