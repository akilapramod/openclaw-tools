"""
Query interface with citation formatting and confidence scoring.
Wraps the hybrid retriever to produce user-facing results.
"""

from typing import Any, Dict, List, Optional

from retriever import HybridRetriever


def format_citation(metadata: Dict[str, Any]) -> str:
    """
    Format a citation from chunk metadata.

    Returns:
        Citation string like: 📖 [Book Title] — Chapter X, Page Y
    """
    book_title = metadata.get("book_title", "Unknown Book")
    chapter = metadata.get("chapter", "Unknown")
    page = metadata.get("page_number", "?")
    return f"📖 [{book_title}] — {chapter}, Page {page}"


def calculate_confidence(results: List[Dict[str, Any]]) -> float:
    """
    Calculate confidence score from reranker scores.

    Uses the average reranker score of the top results,
    normalized to 0-1 range using sigmoid-like scaling.

    Args:
        results: List of result dicts with 'rerank_score' key.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    scores = [r.get("rerank_score", 0.0) for r in results]
    avg_score = sum(scores) / len(scores)

    # Cross-encoder scores are typically in [-10, 10] range
    # Normalize using a simple sigmoid: 1 / (1 + exp(-x))
    import math
    try:
        confidence = 1.0 / (1.0 + math.exp(-avg_score))
    except OverflowError:
        confidence = 0.0 if avg_score < 0 else 1.0

    return round(confidence, 4)


def query(
    query_text: str,
    retriever: Optional[HybridRetriever] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Execute a query against the RAG system.

    Args:
        query_text: The user's search query.
        retriever: Optional pre-configured retriever.
        top_k: Number of results to return.

    Returns:
        Dict with keys:
        - query: The original query
        - results: List of {text, citation, relevance_score}
        - confidence: Overall confidence score (0-1)
        - num_results: Number of results returned
    """
    _retriever = retriever or HybridRetriever()

    raw_results = _retriever.retrieve(query_text, top_k=top_k)

    formatted_results = []
    for r in raw_results:
        formatted_results.append({
            "text": r["text"],
            "citation": format_citation(r.get("metadata", {})),
            "relevance_score": round(r.get("rerank_score", 0.0), 4),
            "metadata": r.get("metadata", {}),
        })

    confidence = calculate_confidence(raw_results)

    return {
        "query": query_text,
        "results": formatted_results,
        "confidence": confidence,
        "num_results": len(formatted_results),
    }
