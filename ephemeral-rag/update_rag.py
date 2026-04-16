import os

new_code = r"""from __future__ import annotations
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

@dataclass
class RAGConfig:
    chunk_size: int = 400
    chunk_overlap: int = 60
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60
    rrf_top_k: int = 20
    bi_encoder_threshold: float = 0.25
    max_rerank_candidates: int = 10
    rerank_top_k: int = 5 
    cross_encoder_threshold: float = 0.1
    context_window: int = 1
    embedding_model: str = 'all-MiniLM-L6-v2'
    cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    max_output_chars: int = 12_000

@dataclass
class Chunk:
    chunk_id: int
    url: str
    title: str
    text: str
    display_text: str
    source_idx: int
    chunk_local_idx: int

@dataclass
class RetrievedResult:
    chunk: Chunk
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    cross_encoder_score: float = 0.0
    bi_encoder_score: float = 0.0

def _simple_tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words: return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words): break
        start += chunk_size - overlap
    return chunks

def build_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    chunk_id = 0
    for source_idx, doc in enumerate(documents):
        url = doc.get('url', 'unknown')
        title = doc.get('title', 'Untitled')
        raw_text = doc.get('raw_text', '')
        if not raw_text.strip(): continue
        source_header = f'[Source: {url} | Title: {title}]'
        raw_chunks = _chunk_text(raw_text, chunk_size, overlap)
        for local_idx, chunk_text in enumerate(raw_chunks):
            display = f'{source_header}\n{chunk_text}'
            all_chunks.append(Chunk(chunk_id=chunk_id, url=url, title=title, text=chunk_text, display_text=display, source_idx=source_idx, chunk_local_idx=local_idx))
            chunk_id += 1
    return all_chunks

@dataclass
class HybridIndex:
    chunks: list[Chunk]
    bm25: BM25Okapi
    embeddings: np.ndarray
    embedding_model: SentenceTransformer

def build_index(chunks: list[Chunk], embedding_model: SentenceTransformer) -> HybridIndex:
    corpus_texts = [c.display_text for c in chunks]
    tokenized_corpus = [_simple_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    raw_embeddings = embedding_model.encode(corpus_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms) 
    embeddings = (raw_embeddings / norms).astype(np.float32)
    return HybridIndex(chunks=chunks, bm25=bm25, embeddings=embeddings, embedding_model=embedding_model)

def _retrieve_bm25(index: HybridIndex, query: str, top_k: int) -> list[tuple[int, float]]:
    tokens = _simple_tokenize(query)
    scores = index.bm25.get_scores(tokens)
    ranked_ids = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked_ids]

def _retrieve_dense(index: HybridIndex, query: str, top_k: int) -> list[tuple[int, float]]:
    q_vec = index.embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0].astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0: q_vec /= norm
    similarities = index.embeddings @ q_vec
    ranked_ids = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in ranked_ids]

def reciprocal_rank_fusion(bm25_results: list[tuple[int, float]], dense_results: list[tuple[int, float]], k: int, top_k: int) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def rerank(query: str, candidates: list[tuple[int, float]], chunks: list[Chunk], index: HybridIndex, cross_encoder: CrossEncoder, config: RAGConfig) -> list[RetrievedResult]:
    if not candidates: return []
    q_vec = index.embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0].astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0: q_vec /= norm
    pre_filtered = []
    for cid, _ in candidates:
        chunk_vec = index.embeddings[cid]
        sim = float(np.dot(q_vec, chunk_vec))
        if sim >= config.bi_encoder_threshold:
            pre_filtered.append((cid, sim))
    pre_filtered.sort(key=lambda x: x[1], reverse=True)
    pre_filtered = pre_filtered[:config.max_rerank_candidates]
    print(f'[RAG] Bi-encoder pre-filter: kept {len(pre_filtered)}/{initial_count} docs')
    if not pre_filtered: return []
    chunk_ids = [cid for cid, _ in pre_filtered]
    bi_scores = {cid: score for cid, score in pre_filtered}
    selected_chunks = [chunks[cid] for cid in chunk_ids]
    pairs = [(query, c.display_text) for c in selected_chunks]
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    results: list[RetrievedResult] = []
    for chunk, ce_score, cid in zip(selected_chunks, ce_scores, chunk_ids):
        if ce_score >= config.cross_encoder_threshold:
            results.append(RetrievedResult(chunk=chunk, cross_encoder_score=float(ce_score), bi_encoder_score=bi_scores[cid]))
    results.sort(key=lambda r: r.cross_encoder_score, reverse=True)
    return results[:config.rerank_top_k]

def expand_context(top_results: list[RetrievedResult], all_chunks: list[Chunk], window: int) -> list[Chunk]:
    by_position = {(c.source_idx, c.chunk_local_idx): c for c in all_chunks}
    seen_ids = set()
    expanded = []
    for result in top_results:
        anchor = result.chunk
        for offset in range(-window, window + 1):
            key = (anchor.source_idx, anchor.chunk_local_idx + offset)
            neighbour = by_position.get(key)
            if neighbour and neighbour.chunk_id not in seen_ids:
                expanded.append(neighbour)
                seen_ids.add(neighbour.chunk_id)
    expanded.sort(key=lambda c: (c.source_idx, c.chunk_local_idx))
    return expanded

def format_output(chunks: list[Chunk], max_chars: int) -> str:
    divider = '\n' + ('\u2500' * 72) + '\n'
    parts = []
    total = 0
    for chunk in chunks:
        segment = chunk.display_text.strip()
        if total + len(segment) > max_chars: break
        parts.append(segment)
        total += len(segment)
    return divider.join(parts)

class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._embedder = None
        self._cross_encoder = None
    def _get_embedder(self):
        if self._embedder is None: self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder
    def _get_cross_encoder(self):
        if self._cross_encoder is None: self._cross_encoder = CrossEncoder(self.config.cross_encoder_model)
        return self._cross_encoder
    def run(self, documents: list[dict], research_question: str) -> str:
        cfg = self.config
        if not documents: return ''
        chunks = build_chunks(documents, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks: return ''
        index = build_index(chunks, self._get_embedder())
        bm25_hits = _retrieve_bm25(index, research_question, cfg.bm25_top_k)
        dense_hits = _retrieve_dense(index, research_question, cfg.dense_top_k)
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits, k=cfg.rrf_k, top_k=cfg.rrf_top_k)
        top_results = rerank(research_question, fused, chunks, index, self._get_cross_encoder(), cfg)
        if not top_results: return ''
        expanded_chunks = expand_context(top_results, chunks, cfg.context_window)
        return format_output(expanded_chunks, cfg.max_output_chars)

def run_ephemeral_rag(documents, research_question, config=None):
    return RAGPipeline(config).run(documents, research_question)
"""

with open('/home/hermit/openclaw-tools/ephemeral-rag/ephemeral_rag.py', 'w') as f:
    f.write(new_code.strip() + '\n')
