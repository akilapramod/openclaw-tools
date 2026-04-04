"""
Web search fallback using Tavily API.
Triggered when RAG confidence is below threshold.
"""

from typing import Any, Dict, List, Optional

from config import TAVILY_API_KEY, CONFIDENCE_THRESHOLD


def should_fallback(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Check if web fallback should be triggered."""
    return confidence < threshold


def web_search(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Perform a web search using Tavily API.

    Args:
        query: Search query.
        api_key: Optional Tavily API key override.
        max_results: Maximum number of results.

    Returns:
        List of result dicts with 'text', 'citation', 'url'.
    """
    _api_key = api_key or TAVILY_API_KEY

    if not _api_key:
        return [{
            "text": "Web search unavailable (TAVILY_API_KEY not set).",
            "citation": "🌐 [No API Key]",
            "url": "",
            "source": "web",
        }]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        results = []
        for item in response.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "Web Source")
            content = item.get("content", "")

            results.append({
                "text": content,
                "citation": f"🌐 [{title}]({url})",
                "url": url,
                "source": "web",
            })

        return results

    except Exception as e:
        return [{
            "text": f"Web search failed: {str(e)}",
            "citation": "🌐 [Error]",
            "url": "",
            "source": "web",
        }]


def merge_results(
    book_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge book RAG results with web search results.

    Book results come first (they're from the primary source),
    followed by web results clearly labeled.

    Args:
        book_results: Results from the book RAG pipeline.
        web_results: Results from web search.

    Returns:
        Combined list with source labels.
    """
    merged = []

    # Add book results with source marker
    for r in book_results:
        entry = r.copy()
        entry.setdefault("source", "book")
        merged.append(entry)

    # Add web results
    for r in web_results:
        entry = r.copy()
        entry.setdefault("source", "web")
        merged.append(entry)

    return merged


def augmented_query(
    query_text: str,
    book_results: List[Dict[str, Any]],
    confidence: float,
    tavily_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform augmented query with web fallback if confidence is low.

    Args:
        query_text: The original query.
        book_results: Results from the book RAG system.
        confidence: Confidence score from the book RAG.
        tavily_api_key: Optional API key.

    Returns:
        Dict with merged results and metadata.
    """
    web_used = False
    web_results = []

    if should_fallback(confidence):
        web_results = web_search(query_text, api_key=tavily_api_key)
        web_used = True

    merged = merge_results(book_results, web_results)

    return {
        "query": query_text,
        "results": merged,
        "book_confidence": confidence,
        "web_fallback_used": web_used,
        "num_book_results": len(book_results),
        "num_web_results": len(web_results),
    }
