"""
ephemeral_rag.py
================
Lightweight, ephemeral Hybrid RAG pipeline for per-query web research.

Given a list of scraped documents and a research question, this module:
 1. Chunks raw text and injects source metadata into every chunk.
 2. Builds an in-memory hybrid index (BM25 + dense embeddings).
 3. Retrieves candidates from both indices and merges via Reciprocal Rank Fusion (RRF).
 4. Reranks the fused candidates with a Cross-Encoder.
 5. Expands winning chunks with their immediate neighbours from the same source.
 6. Returns a clean, citation-rich string ready to paste into an LLM prompt.

Dependencies
------------
 pip install sentence-transformers rank_bm25 numpy

All indices are kept entirely in-process and are garbage-collected when the
RAGPipeline instance goes out of scope — no persistent state, no external servers.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Centralised knobs for the pipeline. Override any field as needed."""

    # --- Chunking ---
    chunk_size: int = 400 # target tokens (approx. words) per chunk
    chunk_overlap: int = 60 # overlap between adjacent chunks

    # --- Retrieval ---
    bm25_top_k: int = 20 # candidates from BM25
    dense_top_k: int = 20 # candidates from dense ANN
    rrf_k: int = 60 # RRF smoothing constant (standard default)
    rrf_top_k: int = 20 # candidates forwarded to the reranker

    # --- Reranking ---
    rerank_top_k: int = 7 # final chunks returned after reranking

    # --- Context expansion ---
    context_window: int = 1 # neighbours ±N to include around winning chunk

    # --- Models ---
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Output ---
    max_output_chars: int = 12_000 # hard cap on concatenated output string


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One text chunk with its provenance metadata."""
    chunk_id: int # global index across all chunks
    url: str
    title: str
    text: str # raw chunk text (no source header)
    display_text: str # text with [Source: …] header prepended
    source_idx: int # index of the parent document in the input list
    chunk_local_idx: int # position of this chunk within its source document


@dataclass
class RetrievedResult:
    """A chunk decorated with retrieval scores for pipeline tracing."""
    chunk: Chunk
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    cross_encoder_score: float = 0.0


# ---------------------------------------------------------------------------
# Step 1 — Chunking
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str) -> list[str]:
    """Minimal whitespace tokenizer used only for BM25 term splitting."""
    return re.findall(r"\w+", text.lower())


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping word-level windows.

    Using words rather than sub-word tokens keeps the chunker dependency-free
    while remaining a close enough proxy for token count with MiniLM models.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def build_chunks(
    documents: list[dict],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """
    Convert a list of raw document dicts into a flat list of Chunk objects.

    Each chunk's display_text begins with a structured citation header so the
    LLM always sees provenance even when chunks are shuffled or truncated.
    """
    all_chunks: list[Chunk] = []
    chunk_id = 0

    for source_idx, doc in enumerate(documents):
        url = doc.get("url", "unknown")
        title = doc.get("title", "Untitled")
        raw_text = doc.get("raw_text", "")

        if not raw_text.strip():
            continue

        source_header = f"[Source: {url} | Title: {title}]"
        raw_chunks = _chunk_text(raw_text, chunk_size, overlap)

        for local_idx, chunk_text in enumerate(raw_chunks):
            display = f"{source_header}\n{chunk_text}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    url=url,
                    title=title,
                    text=chunk_text,
                    display_text=display,
                    source_idx=source_idx,
                    chunk_local_idx=local_idx,
                )
            )
            chunk_id += 1

    return all_chunks


# ---------------------------------------------------------------------------
# Step 2 — Build hybrid index
# ---------------------------------------------------------------------------

@dataclass
class HybridIndex:
    """Container holding both the BM25 index and the dense embedding matrix."""
    chunks: list[Chunk]
    bm25: BM25Okapi
    embeddings: np.ndarray # shape (N, D), float32, L2-normalised
    embedding_model: SentenceTransformer


def build_index(
    chunks: list[Chunk],
    embedding_model: SentenceTransformer,
) -> HybridIndex:
    """
    Build BM25 and dense indices over *chunks* in a single pass.

    Dense embeddings are L2-normalised so that inner-product == cosine similarity,
    enabling a fast matrix–vector multiply at query time without extra normalisation.
    """
    corpus_texts = [c.display_text for c in chunks]

    # --- BM25 ---
    tokenized_corpus = [_simple_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- Dense ---
    # encode_normalize_embeddings is not available in all ST versions, so
    # we normalise manually to keep compatibility.
    raw_embeddings = embedding_model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms) # avoid divide-by-zero
    embeddings = (raw_embeddings / norms).astype(np.float32)

    return HybridIndex(
        chunks=chunks,
        bm25=bm25,
        embeddings=embeddings,
        embedding_model=embedding_model,
    )


# ---------------------------------------------------------------------------
# Step 3 — Hybrid retrieval + RRF
# ---------------------------------------------------------------------------

def _retrieve_bm25(
    index: HybridIndex,
    query: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """Return (chunk_id, bm25_score) pairs ranked best-first."""
    tokens = _simple_tokenize(query)
    scores = index.bm25.get_scores(tokens)
    ranked_ids = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked_ids]


def _retrieve_dense(
    index: HybridIndex,
    query: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """Return (chunk_id, cosine_sim) pairs ranked best-first via brute-force dot product.

    For ephemeral corpora of a few thousand chunks this is fast enough on CPU
    and requires zero extra dependencies (no FAISS / hnswlib).
    """
    q_vec = index.embedding_model.encode(
        [query], show_progress_bar=False, convert_to_numpy=True
    )[0].astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec /= norm

    # shape: (N,) — efficient matrix–vector product
    similarities = index.embeddings @ q_vec
    ranked_ids = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in ranked_ids]


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    Score formula: Σ 1 / (k + rank) where rank is 1-based.

    Documents appearing in only one list still receive a partial score for
    that list and zero contribution from the other — no candidate is silently
    dropped.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Step 4 — Cross-Encoder reranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[tuple[int, float]],
    chunks: list[Chunk],
    cross_encoder: CrossEncoder,
    top_k: int,
) -> list[RetrievedResult]:
    """
    Score (query, chunk) pairs with a cross-encoder and return the best top_k.

    Cross-encoders see both query and passage jointly, giving substantially
    higher ranking quality than bi-encoder cosine similarity alone.
    """
    if not candidates:
        return []

    # Build (chunk_id → rrf_score) lookup before we shuffle the list
    rrf_lookup = {cid: score for cid, score in candidates}
    chunk_ids = [cid for cid, _ in candidates]
    selected_chunks = [chunks[cid] for cid in chunk_ids]

    pairs = [(query, c.display_text) for c in selected_chunks]
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)

    results: list[RetrievedResult] = []
    for chunk, ce_score, chunk_id in zip(selected_chunks, ce_scores, chunk_ids):
        results.append(
            RetrievedResult(
                chunk=chunk,
                rrf_score=rrf_lookup[chunk_id],
                cross_encoder_score=float(ce_score),
            )
        )

    results.sort(key=lambda r: r.cross_encoder_score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Step 5 — Context expansion
# ---------------------------------------------------------------------------

def expand_context(
    top_results: list[RetrievedResult],
    all_chunks: list[Chunk],
    window: int,
) -> list[Chunk]:
    """
    For each winning chunk, include the ±window neighbours from the *same URL*.

    Neighbours are collected in document order (local_idx) and deduplicated
    so overlapping expansions never double-count a chunk.
    """
    # Build a lookup: (source_idx, local_idx) → Chunk
    by_position: dict[tuple[int, int], Chunk] = {
        (c.source_idx, c.chunk_local_idx): c for c in all_chunks
    }

    seen_ids: set[int] = set()
    expanded: list[Chunk] = []

    for result in top_results:
        anchor = result.chunk
        neighbours: list[Chunk] = []

        for offset in range(-window, window + 1):
            key = (anchor.source_idx, anchor.chunk_local_idx + offset)
            neighbour = by_position.get(key)
            if neighbour and neighbour.chunk_id not in seen_ids:
                neighbours.append(neighbour)
                seen_ids.add(neighbour.chunk_id)

        # Sort neighbours by their local position so context reads naturally
        neighbours.sort(key=lambda c: c.chunk_local_idx)
        expanded.extend(neighbours)

    return expanded


# ---------------------------------------------------------------------------
# Step 6 — Format output
# ---------------------------------------------------------------------------

def format_output(chunks: list[Chunk], max_chars: int) -> str:
    """
    Concatenate chunks into a single prompt-ready string.

    Each chunk is separated by a clear divider. The whole string is hard-
    capped at max_chars to protect the LLM's context budget.
    """
    divider = "\n" + ("─" * 72) + "\n"
    parts: list[str] = []
    total = 0

    for chunk in chunks:
        segment = chunk.display_text.strip()
        if total + len(segment) > max_chars:
            remaining = max_chars - total
            if remaining > 200: # only append if there's meaningful space left
                parts.append(segment[:remaining] + " […truncated]")
            break
        parts.append(segment)
        total += len(segment)

    return divider.join(parts)


# ---------------------------------------------------------------------------
# Public API — RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Ephemeral hybrid RAG pipeline.

    Typical usage
    -------------
    pipeline = RAGPipeline() # loads models once
    context = pipeline.run(documents, question) # per-query call
    # pipeline goes out of scope → index memory is freed

    Models are loaded lazily on the first call to `run` and cached on the
    instance so that repeated queries in the same session are fast.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._embedder: Optional[SentenceTransformer] = None
        self._cross_encoder: Optional[CrossEncoder] = None

    # ------------------------------------------------------------------
    # Model loading (deferred, cached per-instance)
    # ------------------------------------------------------------------

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.config.cross_encoder_model)
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def run(
        self,
        documents: list[dict],
        research_question: str,
    ) -> str:
        """
        Execute the full pipeline and return an LLM-ready context string.

        Parameters
        ----------
        documents : list[dict]
            Each dict must contain 'url', 'title', and 'raw_text'.
        research_question : str
            The question the retrieved context should answer.

        Returns
        -------
        str
            Concatenated, citation-annotated chunks ordered by relevance.
            The string is capped at config.max_output_chars.
        """
        cfg = self.config

        if not documents:
            return ""
        if not research_question.strip():
            raise ValueError("research_question must not be empty.")

        # ── 1. Chunk ────────────────────────────────────────────────────────
        chunks = build_chunks(documents, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            return ""

        # ── 2. Index ────────────────────────────────────────────────────────
        index = build_index(chunks, self._get_embedder())

        # ── 3. Retrieve ─────────────────────────────────────────────────────
        bm25_hits = _retrieve_bm25(index, research_question, cfg.bm25_top_k)
        dense_hits = _retrieve_dense(index, research_question, cfg.dense_top_k)

        fused = reciprocal_rank_fusion(
            bm25_hits, dense_hits, k=cfg.rrf_k, top_k=cfg.rrf_top_k
        )

        # ── 4. Rerank ────────────────────────────────────────────────────────
        top_results = rerank(
            research_question,
            fused,
            chunks,
            self._get_cross_encoder(),
            cfg.rerank_top_k,
        )

        if not top_results:
            return ""

        # ── 5. Context expansion ─────────────────────────────────────────────
        expanded_chunks = expand_context(top_results, chunks, cfg.context_window)

        # ── 6. Format ────────────────────────────────────────────────────────
        return format_output(expanded_chunks, cfg.max_output_chars)


# ---------------------------------------------------------------------------
# Convenience wrapper — module-level functional interface
# ---------------------------------------------------------------------------

def run_ephemeral_rag(
    documents: list[dict],
    research_question: str,
    config: Optional[RAGConfig] = None,
) -> str:
    """
    One-shot convenience function that constructs a RAGPipeline, runs it,
    and lets the pipeline object be garbage-collected immediately.

    Use this when you need a single call with no model reuse across queries.
    For repeated queries, instantiate RAGPipeline directly to reuse the
    loaded models.
    """
    return RAGPipeline(config).run(documents, research_question)


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly to verify the setup)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEMO_DOCS = [
        {
            "url": "https://example.com/article-1",
            "title": "Introduction to Hybrid RAG",
            "raw_text": (
                "Retrieval-Augmented Generation combines a retrieval component with a "
                "language model to produce grounded answers. Hybrid RAG extends this "
                "by fusing lexical (BM25) and semantic (dense vector) retrieval signals. "
                "Reciprocal Rank Fusion is a simple but effective technique for combining "
                "ranked lists from multiple retrieval systems without requiring score "
                "normalisation. A cross-encoder reranker then refines the fused candidates "
                "by jointly encoding the query and each passage, yielding higher precision "
                "than bi-encoder cosine similarity alone. "
                * 10 # repeat to create a realistically-sized chunk corpus
            ),
        },
        {
            "url": "https://example.com/article-2",
            "title": "BM25 and Dense Embeddings",
            "raw_text": (
                "BM25 is a probabilistic term-weighting scheme that scores documents "
                "based on query-term frequency and inverse document frequency. Dense "
                "embedding models such as all-MiniLM-L6-v2 map sentences into a "
                "continuous vector space where semantically similar texts cluster "
                "together. Combining both signals captures complementary relevance "
                "cues: BM25 excels at exact-match keywords while dense retrieval "
                "handles paraphrase and semantic generalisation. "
                * 10
            ),
        },
    ]

    _QUESTION = "How does Reciprocal Rank Fusion improve hybrid retrieval?"

    print("Running smoke-test …")
    result = run_ephemeral_rag(_DEMO_DOCS, _QUESTION)
    print("\n=== Pipeline Output ===\n")
    print(textwrap.shorten(result, width=800, placeholder=" […]"))
    print("\nSmoke-test complete.")