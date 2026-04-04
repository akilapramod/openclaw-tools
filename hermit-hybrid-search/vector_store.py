"""
ChromaDB vector store wrapper.
Manages a persistent collection with sentence-transformer embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import _compat  # noqa: F401 — must be before chromadb (Python 3.14 fix)
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
)


class VectorStore:
    """
    Wrapper around ChromaDB for storing and searching document chunks.

    Uses sentence-transformers for embedding generation and ChromaDB
    for persistent vector storage.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        model: Optional[SentenceTransformer] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._model = model or SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Create embedding function wrapper for ChromaDB
        self._ef = _SentenceTransformerEF(self._model)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: list,
        batch_size: int = 100,
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of Chunk objects (from chunker.py).
            batch_size: Number of chunks to upsert at once.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [f"{c.book_title}_{c.page_number}_{c.chunk_index}" for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "book_title": c.book_title,
                    "chapter": c.chapter,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            added += len(batch)

        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            where: Optional ChromaDB where filter on metadata.

        Returns:
            List of dicts with keys: text, metadata, distance, id.
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count() or top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        if self._collection.count() == 0:
            return []

        results = self._collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return output

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return stats about the collection."""
        count = self._collection.count()
        return {
            "collection_name": self._collection.name,
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self._client.delete_collection(self._collection.name)

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()


class _SentenceTransformerEF(chromadb.EmbeddingFunction):
    """ChromaDB-compatible embedding function wrapping sentence-transformers."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def __call__(self, input: list) -> list:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()
