"""
Hybrid retriever with Reciprocal Rank Fusion (RRF) and cross-encoder reranking.
NUCLEAR TEST VERSION - WITH WARNING LOGGING.
"""

import logging
from typing import Any, Dict, List, Optional
from sentence_transformers import CrossEncoder

from config import (
    INITIAL_RETRIEVAL_K,
    FINAL_TOP_K,
    RRF_K,
    RERANKER_MODEL,
    MIN_RELEVANCE_SCORE,
)
from vector_store import VectorStore
from bm25_store import BM25Store

# Setup basic logging to stderr for systemd capture
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nuclear.retriever")

_reranker: Optional[CrossEncoder] = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def reciprocal_rank_fusion(result_lists: List[List[Dict[str, Any]]], k: int = RRF_K) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result.copy()
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    merged = []
    for doc_id in sorted_ids:
        entry = doc_map[doc_id].copy()
        entry["rrf_score"] = scores[doc_id]
        merged.append(entry)
    return merged

def rerank(query: str, results: List[Dict[str, Any]], top_k: int = FINAL_TOP_K, min_score: float = MIN_RELEVANCE_SCORE, reranker: Optional[CrossEncoder] = None) -> List[Dict[str, Any]]:
    if not results:
        return []
    model = reranker or _get_reranker()
    pairs = [[query, r["text"]] for r in results]
    scores = model.predict(pairs)
    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])

    # NUCLEAR FILTER
    filtered_results = [r for r in results if r["rerank_score"] >= min_score]
    
    # UNDENIABLE LOGGING
    logger.warning(f"!!! NUCLEAR RERANK !!! query='{query}' threshold={min_score} results_in={len(results)} results_out={len(filtered_results)}")

    sorted_results = sorted(filtered_results, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_results[:top_k]

class HybridRetriever:
    def __init__(self, vector_store=None, bm25_store=None, reranker=None, initial_k=INITIAL_RETRIEVAL_K, final_k=FINAL_TOP_K):
        self._vs = vector_store or VectorStore()
        self._bm25 = bm25_store or BM25Store()
        self._reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k
        if self._bm25.count() == 0:
            self._bm25.load()

    def retrieve(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        final_k = top_k or self.final_k
        threshold = min_score if min_score is not None else MIN_RELEVANCE_SCORE
        vector_results = self._vs.search(query, top_k=self.initial_k)
        bm25_results = self._bm25.search(query, top_k=self.initial_k)
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        if not fused:
            return []
        return rerank(query=query, results=fused, top_k=final_k, min_score=threshold, reranker=self._reranker)
