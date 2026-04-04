"""
Unified CLI Entry Point — Hybrid RAG for Books.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src/ to path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from core.retriever import HybridRetriever
from ingestion.chunker import SemanticChunker
from ingestion.parser import DocumentIngestor
from storage.vector import VectorStore
from storage.keyword import KeywordStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("book-rag")

def init_components(args):
    base_path = os.path.dirname(__file__)
    db_path = os.path.join(base_path, "data/processed/chroma_db")
    kw_path = os.path.join(base_path, "data/processed/keyword_store.pkl")
    
    vector_store = VectorStore(db_path=db_path)
    keyword_store = KeywordStore(save_path=kw_path)
    chunker = SemanticChunker()
    retriever = HybridRetriever(vector_store=vector_store, bm25_store=keyword_store)
    
    return vector_store, keyword_store, chunker, retriever

def cmd_ingest(args):
    v, k, c, r = init_components(args)
    ingestor = DocumentIngestor(v, k, c, r.embedder, raw_dir=args.raw_dir or "./data/raw")
    ingestor.ingest_all(force_reindex=args.force)

def cmd_search(args):
    v, k, c, r = init_components(args)
    log.info(f"Searching for: '{args.query}' (sources={args.sources}, overfetch={args.overfetch})")
    results = r.retrieve(
        args.query, 
        top_k=args.top_k, 
        sources=args.sources,
        overfetch_factor=args.overfetch
    )
    
    print(f"\n🔍 Query: {args.query}")
    print(f"📄 Top Results ({len(results)}):\n")
    for i, res in enumerate(results, 1):
        print(f"{'─' * 60}")
        print(f"  [{i}] Source: {res['metadata']['source']} (Page {res['metadata']['page']})")
        print(f"  Rerank Score: {res['rerank_score']:.4f}")
        print(f"  {res['text'][:400]}...\n")

def main():
    parser = argparse.ArgumentParser(prog="book-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest")
    p_ingest.add_argument("--raw-dir", type=str)
    p_ingest.add_argument("--force", action="store_true")

    p_search = subparsers.add_parser("search")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--overfetch", type=int, default=3, help="Candidates to fetch per retriever")
    p_search.add_argument("--sources", type=str, nargs='+', help="Filter by source filenames")

    args = parser.parse_args()
    if args.command == "ingest": cmd_ingest(args)
    elif args.command == "search": cmd_search(args)

if __name__ == "__main__": main()
