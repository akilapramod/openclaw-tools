"""
Vector Store Adapter — ChromaDB Persistent Storage with Filtering.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logging.error("Missing dependency: pip install chromadb")
    raise

class VectorStore:
    """Persistent ChromaDB Vector Store Adapter with Metadata Filtering."""

    def __init__(
        self,
        db_path: str = "./data/processed/chroma_db",
        collection_name: str = "book_rag_collection"
    ):
        self.log = logging.getLogger("book-rag.vector")
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.log.info(f"Vector Store initialised at {self.db_path}")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: List[List[float]]):
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size]
            )
        self.log.info(f"Indexed {len(documents)} chunks.")

    def search(self, query_embedding: List[List[float]], k: int = 5, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search with optional source filtering."""
        where = None
        if sources:
            if len(sources) == 1:
                where = {"source": sources[0]}
            else:
                where = {"source": {"$in": sources}}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
        return formatted

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        results = self.collection.get(ids=ids, include=["documents", "metadatas"])
        formatted = []
        for i in range(len(results["ids"])):
            formatted.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        return formatted

    def delete_all(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)
        self.log.warning("Vector Store collection wiped.")
