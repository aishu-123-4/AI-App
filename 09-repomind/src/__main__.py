"""RepoMind CLI.

    python -m src index <repo_path>      # build & save the index
    python -m src ask "how does X work?"  # retrieve + answer
    python -m src ask "..." --repo <path> --k 8 --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .answer import answer, llm_available
from .indexer import build_index, load_index, save_index
from .retriever import search

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "index" / "repo_index.joblib"


def cmd_index(args: argparse.Namespace) -> int:
    index = build_index(args.repo)
    out = save_index(index, args.out)
    print(f"Indexed {len(index['chunks'])} chunks from {index['root']}")
    print(f"Saved index -> {out}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        if args.repo:
            index = build_index(args.repo)
        else:
            print(f"No index at {index_path}. Run `index <repo>` first, "
                  f"or pass --repo.", file=sys.stderr)
            return 1
    else:
        index = load_index(index_path)

    hits = search(index, args.question, k=args.k)
    use_llm = False if args.no_llm else None
    mode = "extractive" if args.no_llm or not llm_available() else "LLM"
    print(f"[retrieved {len(hits)} chunks | {mode} mode]\n")
    print(answer(args.question, hits, use_llm=use_llm))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repomind", description="Ask questions about a code repo.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build and save an index")
    p_index.add_argument("repo", help="path to the repository to index")
    p_index.add_argument("--out", default=str(DEFAULT_INDEX))
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", default=str(DEFAULT_INDEX))
    p_ask.add_argument("--repo", default=None, help="index this repo on the fly if no saved index")
    p_ask.add_argument("--k", type=int, default=5)
    p_ask.add_argument("--no-llm", action="store_true", help="force extractive mode")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
