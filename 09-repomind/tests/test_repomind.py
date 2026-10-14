"""Tests for chunking, indexing, and retrieval (no API key needed)."""

from pathlib import Path

import pytest

from src.answer import answer_extractive
from src.indexer import (
    build_index,
    chunk_python,
    load_index,
    save_index,
)
from src.retriever import search


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "auth.py").write_text(
        "def login(user, password):\n"
        "    '''Authenticate a user and return a session token.'''\n"
        "    token = make_token(user)\n"
        "    return token\n\n"
        "class SessionStore:\n"
        "    def save(self, token):\n"
        "        self._db[token] = True\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def multiply(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Demo\nThis project handles user authentication and sessions.\n",
        encoding="utf-8",
    )
    # Should be skipped (binary-ish / excluded dir).
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_chunk_python_extracts_symbols():
    src = "def foo():\n    return 1\n\nclass Bar:\n    pass\n"
    chunks = chunk_python("a.py", src)
    names = {c.symbol for c in chunks}
    assert "foo" in names and "Bar" in names
    foo = next(c for c in chunks if c.symbol == "foo")
    assert foo.start_line == 1


def test_chunk_python_handles_syntax_error():
    # Falls back to generic chunking instead of raising.
    chunks = chunk_python("bad.py", "def (:\n  not valid python\n")
    assert len(chunks) >= 1


def test_build_index_counts_and_skips(sample_repo):
    index = build_index(sample_repo)
    paths = {c.path for c in index["chunks"]}
    assert any("auth.py" in p for p in paths)
    assert any("README.md" in p for p in paths)
    # __pycache__ is excluded.
    assert not any("__pycache__" in p for p in paths)


def test_retrieval_finds_login(sample_repo):
    index = build_index(sample_repo)
    hits = search(index, "how are users authenticated and logged in?", k=3)
    assert hits, "expected at least one hit"
    top_paths = [h.chunk.path for h in hits]
    assert any("auth.py" in p for p in top_paths)


def test_save_and_load_roundtrip(sample_repo, tmp_path):
    index = build_index(sample_repo)
    out = save_index(index, tmp_path / "idx.joblib")
    loaded = load_index(out)
    assert len(loaded["chunks"]) == len(index["chunks"])
    hits = search(loaded, "multiply two numbers", k=2)
    assert any("math_utils.py" in h.chunk.path for h in hits)


def test_extractive_answer_cites(sample_repo):
    index = build_index(sample_repo)
    hits = search(index, "login", k=2)
    out = answer_extractive("login", hits)
    assert "auth.py:" in out


def test_empty_repo_raises(tmp_path):
    with pytest.raises(ValueError):
        build_index(tmp_path)
