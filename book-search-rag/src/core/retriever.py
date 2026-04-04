"""
Hybrid Retriever Module — Precision-Enhanced Retrieval with Filtering.
"""

import numpy as np
import logging
import re
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    logging.error("Missing dependencies: pip install sentence-transformers")
    raise

class HybridRetriever:
    """Combines Vector + Keyword + Context-Expanded Reranking with Source Filtering."""

    def __init__(
        self,
        vector_store: Any,
        bm25_store: Any,
        embed_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60
    ):
        self.log = logging.getLogger("book-rag.retriever")
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k
        self.embedder = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(rerank_model)

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        overfetch_factor: int = 3, 
        min_score: float = 0.0,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform high-precision hybrid retrieval with optional source filtering."""
        fetch_k = max(top_k * overfetch_factor, 15)

        # 1. Vector Retrieval
        q_emb = self.embedder.encode([query])
        vector_results = self.vector_store.search(q_emb, k=fetch_k, sources=sources)

        # 2. Keyword Retrieval
        bm25_results = self.bm25_store.search(query, k=fetch_k, sources=sources)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, res in enumerate(vector_results):
            cid = res["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[cid] = res

        for rank, res in enumerate(bm25_results):
            cid = res["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank + 1)
            if cid not in doc_map:
                doc_map[cid] = res

        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [doc_map[cid] for cid, _ in ranked_ids[:fetch_k]]

        if not candidates:
            return []

        # 4. Context Expansion
        expanded_pairs = []
        for cand in candidates:
            expanded_text = self._expand_context(cand)
            expanded_pairs.append([query, expanded_text])

        # 5. Cross-Encoder Reranking
        ce_scores = self.reranker.predict(expanded_pairs)

        for doc, score in zip(candidates, ce_scores):
            doc["rerank_score"] = float(score)

        # 6. Threshold Filtering
        filtered_results = [c for c in candidates if c["rerank_score"] >= min_score]
        final_results = sorted(filtered_results, key=lambda x: x["rerank_score"], reverse=True)

        return final_results[:top_k]

    def _expand_context(self, candidate: Dict[str, Any]) -> str:
        cid = candidate["id"]
        match = re.match(r"(.+)::page_(\d+)::chunk_(\d+)", cid)
        if not match: return candidate["text"]

        doc, page, chunk_idx = match.groups()
        page, chunk_idx = int(page), int(chunk_idx)
        
        prev_id = f"{doc}::page_{page}::chunk_{chunk_idx - 1}"
        next_id = f"{doc}::page_{page}::chunk_{chunk_idx + 1}"
        
        neighbors = self.bm25_store.get_by_ids([prev_id, next_id])
        
        prev_text, next_text = "", ""
        for n in neighbors:
            if n["id"] == prev_id: prev_text = n["text"]
            elif n["id"] == next_id: next_text = n["text"]
        
        return f"{prev_text} {candidate['text']} {next_text}".strip()
