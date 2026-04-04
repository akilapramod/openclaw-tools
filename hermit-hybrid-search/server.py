#!/usr/bin/env python3
"""
Hermit Hybrid RAG Memory Server - NUCLEAR FIXED
================================
"""

import os
import glob
import re
import json
import logging
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MEMORY_DIR = os.path.expanduser("~/memory_files")
CHROMA_DIR = os.path.expanduser("~/openclaw-tools/hermit-hybrid-search/chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "hermit_memory"
CHUNK_MAX_WORDS = 120
RRF_K = 60
HOST = "0.0.0.0"
PORT = 5055

# NUCLEAR THRESHOLD
MIN_RELEVANCE_SCORE = 0.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("hermit-rag")

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class HermitHybridRetriever:
    """Full hybrid retriever: Vector + BM25 + RRF + Cross-Encoder reranking."""

    def __init__(self):
        log.info("Loading embedding model: %s", EMBED_MODEL)
        self.embedder = SentenceTransformer(EMBED_MODEL)

        log.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        log.info("Initialising ChromaDB (persistent) at %s", CHROMA_DIR)
        self.chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)

        self.documents = []
        self.metadata = []
        self.bm25 = None

        self._build_index()

    def _chunk_text(self, text, max_words=CHUNK_MAX_WORDS):
        raw = [c.strip() for c in text.split("\n\n") if c.strip()]
        chunks = []
        for block in raw:
            words = block.split()
            if len(words) > max_words:
                for i in range(0, len(words), max_words):
                    chunks.append(" ".join(words[i : i + max_words]))
            else:
                chunks.append(block)
        return chunks

    def _build_index(self, force=False):
        files = sorted(glob.glob(os.path.join(MEMORY_DIR, "*.md")))
        if not files:
            log.warning("No .md files in %s — index is empty", MEMORY_DIR)
            return
        log.info("Scanning %d files in %s", len(files), MEMORY_DIR)
        all_chunks, all_meta, all_ids = [], [], []
        for fpath in files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            for i, chunk in enumerate(self._chunk_text(text)):
                cid = f"{fname}::chunk_{i}"
                boost = 1.5 if fname == "MEMORY.md" else 1.0
                all_chunks.append(chunk)
                all_meta.append({"source": fname, "chunk_index": i, "boost": boost})
                all_ids.append(cid)
        self.documents = all_chunks
        self.metadata = all_meta
        self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])
        log.info("BM25 index built: %d chunks", len(self.documents))
        existing = self.collection.count()
        if force or existing != len(all_chunks):
            log.info("Rebuilding ChromaDB (%d -> %d)", existing, len(all_chunks))
            self.chroma.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma.create_collection(COLLECTION_NAME)
            batch = 100
            for i in range(0, len(all_chunks), batch):
                b_chunks = all_chunks[i : i + batch]
                b_ids = all_ids[i : i + batch]
                b_meta = all_meta[i : i + batch]
                b_emb = self.embedder.encode(b_chunks).tolist()
                self.collection.add(documents=b_chunks, embeddings=b_emb, metadatas=b_meta, ids=b_ids)
            log.info("ChromaDB indexed %d chunks", len(all_chunks))
        else:
            log.info("ChromaDB up-to-date (%d chunks), skipping embed", existing)

    def search(self, query, top_k=5, use_reranker=True, min_score=MIN_RELEVANCE_SCORE):
        if not self.documents: return []
        fetch_k = max(top_k * 3, 15)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        for i, m in enumerate(self.metadata):
            bm25_scores[i] *= m["boost"]
        bm25_top = np.argsort(bm25_scores)[::-1][:fetch_k]
        qemb = self.embedder.encode([query]).tolist()
        vec = self.collection.query(query_embeddings=qemb, n_results=fetch_k, include=["documents", "metadatas", "distances"])
        rrf = {}
        idx_map = {}
        for rank, idx in enumerate(bm25_top):
            m = self.metadata[idx]
            cid = f"{m['source']}::chunk_{m['chunk_index']}"
            rrf[cid] = rrf.get(cid, 0) + 1 / (RRF_K + rank + 1)
            idx_map[cid] = idx
        if vec["ids"] and vec["ids"][0]:
            for rank, cid in enumerate(vec["ids"][0]):
                rrf[cid] = rrf.get(cid, 0) + 1 / (RRF_K + rank + 1)
                if cid not in idx_map:
                    parts = cid.split("::chunk_")
                    if len(parts) == 2:
                        src, ci = parts[0], int(parts[1])
                        for j, m in enumerate(self.metadata):
                            if m["source"] == src and m["chunk_index"] == ci:
                                idx_map[cid] = j
                                break
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for cid, score in ranked[: fetch_k]:
            idx = idx_map.get(cid)
            if idx is not None:
                candidates.append({"id": cid, "source": self.metadata[idx]["source"], "content": self.documents[idx]})
        if use_reranker and candidates:
            pairs = [(query, c["content"]) for c in candidates]
            ce_scores = self.reranker.predict(pairs).tolist()
            for c, s in zip(candidates, ce_scores):
                c["rerank_score"] = float(s)
            # FILTER BY MIN_SCORE
            candidates = [c for c in candidates if c.get("rerank_score", -99) >= min_score]
            log.warning("!!! NUCLEAR FILTER !!! query='%s' threshold=%f filtered_count=%d", query, min_score, len(candidates))
            candidates.sort(key=lambda c: c.get("rerank_score", -99), reverse=True)
        return candidates[:top_k]

    def stats(self):
        return {"total_chunks": len(self.documents), "chroma_count": self.collection.count(), "files_indexed": len(set(m["source"] for m in self.metadata))}

app = Flask(__name__)
retriever = None

@app.route("/health")
def health(): return jsonify({"status": "ok"})

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    results = retriever.search(query, top_k=top_k)
    return jsonify({"query": query, "results": results})

if __name__ == "__main__":
    retriever = HermitHybridRetriever()
    app.run(host=HOST, port=PORT, debug=False)
