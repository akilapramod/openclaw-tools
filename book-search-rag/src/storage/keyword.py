"""
Keyword Store Adapter — BM25 Lexical Search with Persistence and Filtering.
"""

import logging
import pickle
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logging.error("Missing dependency: pip install rank-bm25")
    raise

class KeywordStore:
    """Persistent BM25 Keyword Search Store with Source Filtering."""

    def __init__(self, save_path: str = "./data/processed/keyword_store.pkl", k1: float = 1.5, b: float = 0.75):
        self.log = logging.getLogger("book-rag.keyword")
        self.save_path = Path(save_path)
        self.k1 = k1
        self.b = b
        
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.id_to_idx: Dict[str, int] = {}

        self.load()

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        self._build_index()
        self.save()
        self.log.info(f"Keyword Store indexed {len(self.documents)} total chunks.")

    def _build_index(self):
        if not self.documents:
            self.bm25 = None
            self.id_to_idx = {}
            return
            
        tokenized_corpus = [d.lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.ids)}

    def search(self, query: str, k: int = 5, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform lexical keyword search with optional source filtering."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Format and filter results
        formatted = []
        for i in range(len(scores)):
            meta = self.metadatas[i]
            # Apply filter
            if sources and meta.get("source") not in sources:
                continue
                
            formatted.append({
                "id": self.ids[i],
                "text": self.documents[i],
                "metadata": meta,
                "bm25_score": float(scores[i])
            })
            
        # Sort by score and take top K
        formatted = sorted(formatted, key=lambda x: x["bm25_score"], reverse=True)
        return formatted[:k]

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        formatted = []
        for cid in ids:
            if cid in self.id_to_idx:
                idx = self.id_to_idx[cid]
                formatted.append({
                    "id": self.ids[idx],
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx]
                })
        return formatted

    def save(self):
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'wb') as f:
                pickle.dump({'documents': self.documents, 'metadatas': self.metadatas, 'ids': self.ids}, f)
        except Exception as e:
            self.log.error(f"Failed to save: {str(e)}")

    def load(self):
        if not self.save_path.exists(): return
        try:
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])
            self._build_index()
        except Exception as e:
            self.log.error(f"Failed to load: {str(e)}")

    def clear(self):
        self.documents, self.metadatas, self.ids, self.bm25, self.id_to_idx = [], [], [], None, {}
        if self.save_path.exists(): os.remove(self.save_path)
