"""
Groq LLM module for answer generation.
Takes retrieved context and generates grounded answers with citations.
"""

from typing import Any, Dict, List, Optional

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE


SYSTEM_PROMPT = """You are a precise research assistant that answers questions ONLY based on the provided context.

Rules:
1. Answer ONLY from the provided context. Do NOT use any external knowledge.
2. If the context doesn't contain enough information, say "I cannot find sufficient information in the provided books."
3. Include inline citations using the EXACT citation format provided with each context chunk.
4. Be concise but thorough.
5. When referencing specific facts, always include the citation immediately after.

Citation format examples:
- 📖 [Book Title] — Chapter X, Page Y (for book sources)
- 🌐 [URL] (for web sources)
"""


def build_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Build the user prompt with context chunks and citations.

    Args:
        query: The user's question.
        results: List of result dicts with 'text' and 'citation' keys.

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, r in enumerate(results, 1):
        citation = r.get("citation", "Unknown source")
        text = r.get("text", "")
        context_parts.append(f"[Source {i}] {citation}\n{text}")

    context_block = "\n\n---\n\n".join(context_parts)

    return (
        f"Context:\n{context_block}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above. Include citations."
    )


def generate_answer(
    query: str,
    results: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an answer using Groq LLM.

    Args:
        query: The user's question.
        results: Retrieved context results with 'text' and 'citation'.
        api_key: Optional Groq API key override.
        model: Optional model name override.

    Returns:
        Dict with keys: answer, sources_used, model, success.
    """
    _api_key = api_key or GROQ_API_KEY
    _model = model or GROQ_MODEL

    if not _api_key:
        return {
            "answer": _fallback_answer(results),
            "sources_used": [r.get("citation", "") for r in results],
            "model": "fallback (no API key)",
            "success": False,
            "error": "GROQ_API_KEY not set. Set it via environment variable.",
        }

    if not results:
        return {
            "answer": "No relevant context was found to answer this question.",
            "sources_used": [],
            "model": _model,
            "success": True,
        }

    try:
        from groq import Groq

        client = Groq(api_key=_api_key)
        user_prompt = build_prompt(query, results)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=_model,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )

        answer = chat_completion.choices[0].message.content

        return {
            "answer": answer,
            "sources_used": [r.get("citation", "") for r in results],
            "model": _model,
            "success": True,
        }

    except Exception as e:
        return {
            "answer": _fallback_answer(results),
            "sources_used": [r.get("citation", "") for r in results],
            "model": "fallback (error)",
            "success": False,
            "error": str(e),
        }


def _fallback_answer(results: List[Dict[str, Any]]) -> str:
    """
    Generate a simple fallback answer when Groq is unavailable.
    Just presents the retrieved context with citations.
    """
    if not results:
        return "No relevant information found."

    parts = ["Here are the most relevant passages found:\n"]
    for i, r in enumerate(results, 1):
        citation = r.get("citation", "Unknown source")
        text = r.get("text", "")
        parts.append(f"{i}. {citation}\n   {text}\n")

    return "\n".join(parts)
