"""
Semantic chunking module.
Splits text on meaning boundaries using sentence embeddings cosine similarity.
Falls back to character-based splitting for oversized chunks.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    SEMANTIC_SIMILARITY_THRESHOLD,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
)


@dataclass
class Chunk:
    """A semantically coherent text chunk with source metadata."""
    text: str
    book_title: str
    chapter: str
    page_number: int
    chunk_index: int = 0  # Index within the document


# Module-level model cache (lazy loaded)
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model to save memory."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations and decimal numbers.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter out very short fragments
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _recursive_char_split(text: str, max_size: int, overlap: int) -> List[str]:
    """
    Fallback: split text by character count with overlap.
    Tries to break on paragraph/sentence boundaries when possible.
    """
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size

        if end < len(text):
            # Try to break at paragraph boundary
            break_point = text.rfind("\n\n", start, end)
            if break_point == -1 or break_point <= start:
                # Try sentence boundary
                break_point = text.rfind(". ", start, end)
            if break_point == -1 or break_point <= start:
                # Try word boundary
                break_point = text.rfind(" ", start, end)
            if break_point > start:
                end = break_point + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Ensure forward progress: overlap must be less than chunk size
        effective_overlap = min(overlap, end - start - 1) if end < len(text) else 0
        start = end - effective_overlap if end < len(text) else len(text)

    return chunks


def semantic_chunk(
    text: str,
    book_title: str,
    chapter: str,
    page_number: int,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    model: Optional[SentenceTransformer] = None,
) -> List[Chunk]:
    """
    Split text into semantically coherent chunks.

    Algorithm:
    1. Split text into sentences.
    2. Compute embeddings for each sentence.
    3. Group consecutive sentences where cosine similarity > threshold.
    4. If a group exceeds max_chunk_size, apply recursive character splitting.
    5. Discard chunks smaller than min_chunk_size.

    Args:
        text: The full text to chunk.
        book_title: Title of the source book.
        chapter: Chapter name/number.
        page_number: Source page number.
        similarity_threshold: Cosine similarity threshold for splitting.
        max_chunk_size: Max characters per chunk before fallback split.
        chunk_overlap: Overlap characters for fallback splits.
        min_chunk_size: Discard chunks smaller than this.
        model: Optional pre-loaded SentenceTransformer model.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    sentences = _split_into_sentences(text)

    # If too few sentences, return as single chunk (with possible size split)
    if len(sentences) <= 1:
        chunk_texts = _recursive_char_split(text.strip(), max_chunk_size, chunk_overlap)
        return [
            Chunk(
                text=t, book_title=book_title,
                chapter=chapter, page_number=page_number,
            )
            for t in chunk_texts if len(t) >= min_chunk_size
        ]

    # Compute sentence embeddings
    _model = model or _get_model()
    embeddings = _model.encode(sentences, show_progress_bar=False)

    # Group sentences by semantic similarity
    groups: List[List[str]] = [[sentences[0]]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim >= similarity_threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    # Convert groups to chunks, applying size-based fallback if needed
    chunks: List[Chunk] = []
    for group in groups:
        group_text = " ".join(group)

        if len(group_text) > max_chunk_size:
            sub_texts = _recursive_char_split(group_text, max_chunk_size, chunk_overlap)
        else:
            sub_texts = [group_text]

        for t in sub_texts:
            if len(t) >= min_chunk_size:
                chunks.append(Chunk(
                    text=t, book_title=book_title,
                    chapter=chapter, page_number=page_number,
                ))

    # Assign chunk indices
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    return chunks


def chunk_pages(pages: list, model: Optional[SentenceTransformer] = None) -> List[Chunk]:
    """
    Chunk a list of PageDocument objects.

    Args:
        pages: List of PageDocument objects from parser.py.
        model: Optional pre-loaded SentenceTransformer model.

    Returns:
        List of Chunk objects with sequential chunk_index.
    """
    all_chunks: List[Chunk] = []
    _model = model or _get_model()

    for page in pages:
        page_chunks = semantic_chunk(
            text=page.text,
            book_title=page.book_title,
            chapter=page.chapter,
            page_number=page.page_number,
            model=_model,
        )
        all_chunks.extend(page_chunks)

    # Re-index globally
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    return all_chunks
