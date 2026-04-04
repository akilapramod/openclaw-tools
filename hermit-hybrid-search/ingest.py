"""
Ingestion CLI — Parse PDFs, chunk semantically, and index into vector + BM25 stores.
Usage: python ingest.py --books-dir ./data
"""

import argparse
import sys
import time
from pathlib import Path

from config import DATA_DIR
from parser import parse_directory
from chunker import chunk_pages
from vector_store import VectorStore
from bm25_store import BM25Store


def ingest(books_dir: str | Path, skip_existing: bool = True) -> dict:
    """
    Full ingestion pipeline: parse → chunk → index.

    Args:
        books_dir: Directory containing PDF files.
        skip_existing: If True, skip books already in the index.

    Returns:
        Stats dict with counts.
    """
    books_dir = Path(books_dir)
    print(f"\n📚 Book Search RAG — Ingestion Pipeline")
    print(f"{'─' * 50}")
    print(f"📁 Books directory: {books_dir}")
    start_time = time.time()

    # Step 1: Parse PDFs
    print(f"\n🔍 Step 1/3: Parsing PDFs...")
    pages = parse_directory(books_dir)
    if not pages:
        print("❌ No pages extracted. Check your PDF files.")
        return {"pages": 0, "chunks": 0, "indexed": 0}
    print(f"   ✅ Extracted {len(pages)} pages")

    # Step 2: Semantic chunking
    print(f"\n✂️  Step 2/3: Semantic chunking...")
    chunks = chunk_pages(pages)
    print(f"   ✅ Created {len(chunks)} chunks")

    # Step 3: Index into stores
    print(f"\n📥 Step 3/3: Indexing into vector + BM25 stores...")

    # Vector store
    vs = VectorStore()
    added_vs = vs.add_documents(chunks)
    print(f"   ✅ Vector store: {added_vs} chunks indexed")

    # BM25 store
    bm25 = BM25Store()
    bm25.add_documents(chunks)
    bm25.save()
    print(f"   ✅ BM25 store: {bm25.count()} chunks indexed")

    elapsed = time.time() - start_time
    stats = vs.get_collection_stats()
    print(f"\n{'─' * 50}")
    print(f"✅ Ingestion complete in {elapsed:.1f}s")
    print(f"   Total chunks in store: {stats['total_chunks']}")
    print(f"   Persist dir: {stats['persist_dir']}")

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "indexed": added_vs,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF books into the RAG search system."
    )
    parser.add_argument(
        "--books-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Directory containing PDF files (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    if not Path(args.books_dir).is_dir():
        print(f"❌ Directory not found: {args.books_dir}")
        sys.exit(1)

    ingest(args.books_dir)


if __name__ == "__main__":
    main()
