"""Index a code repository into retrievable chunks.

Chunking is *symbol-aware* for Python (each function/class becomes a chunk via
the `ast` module, so citations land on real definitions) and a sliding window
of lines for everything else. Chunks are embedded with TF-IDF so retrieval
works fully offline with no model download.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# File types we index; everything else (binaries, images, locks) is skipped.
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".sh", ".sql",
    ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".ini",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
             "dist", "build", ".pytest_cache", ".mypy_cache", ".idea",
             "models", "data", ".ruff_cache"}

WINDOW = 45          # lines per generic chunk
OVERLAP = 10         # overlapping lines between generic chunks
MAX_FILE_BYTES = 1_000_000


@dataclass
class CodeChunk:
    path: str          # repo-relative path
    start_line: int    # 1-indexed inclusive
    end_line: int
    symbol: str        # function/class name, or "" for generic windows
    text: str

    @property
    def citation(self) -> str:
        loc = f"{self.path}:{self.start_line}"
        return f"{loc} ({self.symbol})" if self.symbol else loc


def chunk_python(rel_path: str, source: str) -> list[CodeChunk]:
    """Split Python source into one chunk per top-level + nested def/class."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_generic(rel_path, source)

    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            text = "\n".join(lines[start - 1:end])
            if text.strip():
                chunks.append(CodeChunk(rel_path, start, end, node.name, text))
    # Also capture module-level code (imports, constants) as one chunk.
    if not chunks and source.strip():
        return chunk_generic(rel_path, source)
    return chunks


def chunk_generic(rel_path: str, source: str) -> list[CodeChunk]:
    lines = source.splitlines()
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    step = max(1, WINDOW - OVERLAP)
    for start in range(0, len(lines), step):
        window = lines[start:start + WINDOW]
        text = "\n".join(window)
        if text.strip():
            chunks.append(CodeChunk(rel_path, start + 1, start + len(window), "", text))
        if start + WINDOW >= len(lines):
            break
    return chunks


def iter_repo_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def build_chunks(root: Path) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for path in iter_repo_files(root):
        rel = str(path.relative_to(root))
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix.lower() == ".py":
            chunks.extend(chunk_python(rel, source))
        else:
            chunks.extend(chunk_generic(rel, source))
    return chunks


def build_index(root: str | Path) -> dict:
    """Build a TF-IDF index over a repository's code chunks."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    chunks = build_chunks(root)
    if not chunks:
        raise ValueError(f"No indexable code files found under {root}")
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9,
        token_pattern=r"(?u)\b\w[\w/.\-]+\b",
    )
    matrix = vectorizer.fit_transform(c.text for c in chunks)
    return {"root": str(root), "vectorizer": vectorizer,
            "matrix": matrix, "chunks": chunks}


def save_index(index: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store chunks as dicts for a stable, inspectable artifact.
    payload = {**index, "chunks": [asdict(c) for c in index["chunks"]]}
    joblib.dump(payload, path)
    return path


def load_index(path: str | Path) -> dict:
    payload = joblib.load(path)
    payload["chunks"] = [CodeChunk(**d) for d in payload["chunks"]]
    return payload
