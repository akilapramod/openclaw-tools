"""
Central configuration for the Book Search RAG system.
All tunable parameters and paths live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env file into os.environ

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"
CHROMA_DIR = DB_DIR / "chroma"
BM25_PATH = DB_DIR / "bm25_index.pkl"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ─── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"           # ~80MB, CPU-friendly, 384-dim
EMBEDDING_DIMENSION = 384

# ─── Chunking ───────────────────────────────────────────────────────────────
SEMANTIC_SIMILARITY_THRESHOLD = 0.75            # Split when cosine sim drops below
MAX_CHUNK_SIZE = 1500                           # Max chars per chunk (fallback split)
CHUNK_OVERLAP = 200                             # Overlap chars for fallback splits
MIN_CHUNK_SIZE = 100                            # Discard chunks smaller than this

# ─── Retrieval ──────────────────────────────────────────────────────────────
INITIAL_RETRIEVAL_K = 20                        # Candidates from each retriever
FINAL_TOP_K = 5                                 # Results after reranking
RRF_K = 60                                      # RRF constant (standard value)
MIN_RELEVANCE_SCORE = 100.0                     # NUCLEAR TEST: Threshold impossible to reach

# ─── Reranker ───────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~80MB, CPU-friendly

# ─── Confidence & Fallback ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3                      # Below this → trigger web fallback
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ─── Groq LLM ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"             # Fast, free-tier model
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.1                          # Low temp for factual answers

# ─── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "book_chunks"
