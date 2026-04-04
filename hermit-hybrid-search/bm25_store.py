"""
BM25 keyword search store.
Uses rank_bm25 for lexical retrieval, with pickle persistence.
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from config import BM25_PATH


class BM25Store:
    """
    BM25-based keyword search index.

    Stores tokenized documents and their metadata, supports
    keyword-based retrieval. Persists to disk via pickle.
    """

    def __init__(self, persist_path: str | Path = BM25_PATH):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._documents: List[str] = []       # Original text
        self._tokenized: List[List[str]] = [] # Tokenized text
        self._metadatas: List[Dict] = []      # Metadata per doc
        self._ids: List[str] = []             # Document IDs
        self._bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenizer: lowercase, split on non-alphanumeric,
        filter short tokens.
        """
        tokens = re.findall(r'\b[a-z0-9]+(?:\.[a-z0-9]+)*\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    def add_documents(self, chunks: list) -> int:
        """
        Add document chunks to the BM25 index.

        Args:
            chunks: List of Chunk objects (from chunker.py).

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        added = 0
        for c in chunks:
            doc_id = f"{c.book_title}_{c.page_number}_{c.chunk_index}"

            # Skip duplicates
            if doc_id in self._ids:
                continue

            tokens = self._tokenize(c.text)
            if not tokens:
                continue

            self._documents.append(c.text)
            self._tokenized.append(tokens)
            self._metadatas.append({
                "book_title": c.book_title,
                "chapter": c.chapter,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
            })
            self._ids.append(doc_id)
            added += 1

        # Rebuild BM25 index
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

        return added

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata, score, id.
        """
        if not self._bm25 or not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-K indices sorted by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "id": self._ids[idx],
                    "text": self._documents[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(scores[idx]),
                })

        return results

    def save(self):
        """Persist the BM25 index to disk."""
        data = {
            "documents": self._documents,
            "tokenized": self._tokenized,
            "metadatas": self._metadatas,
            "ids": self._ids,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """
        Load the BM25 index from disk.

        Returns:
            True if loaded successfully, False if file not found.
        """
        if not self.persist_path.exists():
            return False

        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)

        self._documents = data["documents"]
        self._tokenized = data["tokenized"]
        self._metadatas = data["metadatas"]
        self._ids = data["ids"]

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

        return True

    def count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._documents)

    def clear(self):
        """Clear the index."""
        self._documents = []
        self._tokenized = []
        self._metadatas = []
        self._ids = []
        self._bm25 = None
