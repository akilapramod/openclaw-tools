"""
Book Search RAG — Unified CLI.

Usage:
    python main.py ingest  [--books-dir ./data]
    python main.py search  "your query"   [--top-k 5]
    python main.py ask     "your question" [--top-k 5]
"""

import _compat  # noqa: F401 — Python 3.14 + ChromaDB compatibility shim

import argparse
import sys

from config import DATA_DIR


def cmd_ingest(args):
    from ingest import ingest
    ingest(args.books_dir)


def cmd_search(args):
    from query import query

    result = query(args.query, top_k=args.top_k)

    print(f"\n🔍 Query: {result['query']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"📄 Results: {result['num_results']}\n")

    for i, r in enumerate(result["results"], 1):
        print(f"{'─' * 60}")
        print(f"  [{i}] {r['citation']}")
        print(f"  Relevance: {r['relevance_score']:.4f}")
        print(f"  {r['text'][:300]}{'…' if len(r['text']) > 300 else ''}\n")

    return result


def cmd_ask(args):
    from query import query
    from web_fallback import augmented_query
    from llm import generate_answer

    # Step 1: Retrieve from books
    rag_result = query(args.query, top_k=args.top_k)
    print(f"\n📊 Book confidence: {rag_result['confidence']:.2%}")

    # Step 2: Web fallback if low confidence
    augmented = augmented_query(
        args.query,
        rag_result["results"],
        rag_result["confidence"],
    )

    if augmented["web_fallback_used"]:
        print(f"🌐 Web fallback triggered — {augmented['num_web_results']} web results added")

    # Step 3: Generate answer via Groq LLM
    answer_result = generate_answer(args.query, augmented["results"])

    print(f"\n{'═' * 60}")
    if answer_result["success"]:
        print(f"🤖 [{answer_result['model']}]\n")
    else:
        print(f"⚠️  LLM unavailable ({answer_result.get('error', 'unknown')})")
        print(f"📝 Fallback answer:\n")

    print(answer_result["answer"])

    if answer_result["sources_used"]:
        print(f"\n{'─' * 60}")
        print("📚 Sources:")
        for src in answer_result["sources_used"]:
            print(f"   • {src}")


def main():
    parser = argparse.ArgumentParser(
        prog="book-rag",
        description="Book Search RAG — Ingest, Search, and Ask your PDF library.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest ───────────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="Parse and index PDF books.")
    p_ingest.add_argument(
        "--books-dir", type=str, default=str(DATA_DIR),
        help=f"Directory containing PDF files (default: {DATA_DIR})",
    )

    # ── search ───────────────────────────────────────────────────────────
    p_search = subparsers.add_parser("search", help="Search your books (retrieval only).")
    p_search.add_argument("query", type=str, help="Your search query.")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results.")

    # ── ask ───────────────────────────────────────────────────────────────
    p_ask = subparsers.add_parser(
        "ask", help="Ask a question (retrieval + LLM answer + web fallback)."
    )
    p_ask.add_argument("query", type=str, help="Your question.")
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of context chunks.")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
