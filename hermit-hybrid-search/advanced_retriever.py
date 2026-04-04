import os
import glob
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HermitAdvancedHybridRetriever:
    """
    Advanced Hybrid Retriever for Hermit.
    Combines Vector Search (ChromaDB + all-MiniLM) and Keyword Search (BM25Okapi).
    Uses Reciprocal Rank Fusion (RRF) for reranking.
    """
    def __init__(self, memory_dir="~/memory_files", db_dir="~/hermit-hybrid-search/chroma_db"):
        self.memory_dir = Path(os.path.expanduser(memory_dir))
        self.db_dir = Path(os.path.expanduser(db_dir))
        
        logger.info("Initializing models and databases...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.chroma_client.get_or_create_collection(name="hermit_memory")
        
        self.documents = []
        self.metadata = []
        self.bm25 = None
        self.tokenized_corpus = []
        
        self._index_documents()

    def _chunk_text(self, text, max_words=100):
        """Naive semantic chunker (splits by double newline, then chunks by word count if too large)"""
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        refined_chunks = []
        for chunk in raw_chunks:
            words = chunk.split()
            if len(words) > max_words:
                for i in range(0, len(words), max_words):
                    refined_chunks.append(" ".join(words[i:i+max_words]))
            else:
                refined_chunks.append(chunk)
        return refined_chunks

    def _index_documents(self):
        """Reads markdown files, chunks them, and builds Vector + BM25 indexes."""
        files = glob.glob(str(self.memory_dir / "*.md"))
        if not files:
            logger.warning(f"No markdown files found in {self.memory_dir}")
            return

        logger.info(f"Indexing {len(files)} files...")
        
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_chunk_{i}"
                all_chunks.append(chunk)
                
                # Boost MEMORY.md
                boost = 1.5 if file_name == "MEMORY.md" else 1.0
                all_metadata.append({"source": file_name, "boost": boost, "chunk_index": i})
                all_ids.append(chunk_id)

        # Update BM25 Index
        self.documents = all_chunks
        self.metadata = all_metadata
        self.tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Update Vector Index (Skip if already exists for simplicity in this prototype, or wipe and rebuild)
        existing_count = self.collection.count()
        if existing_count < len(all_chunks):
            logger.info(f"Updating ChromaDB. Generating embeddings for {len(all_chunks)} chunks...")
            
            # Wipe existing collection for clean state
            self.chroma_client.delete_collection("hermit_memory")
            self.collection = self.chroma_client.create_collection(name="hermit_memory")
            
            # Batch add to avoid memory spikes
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i+batch_size]
                batch_ids = all_ids[i:i+batch_size]
                batch_metadata = all_metadata[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch_chunks).tolist()
                
                self.collection.add(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            logger.info("ChromaDB indexing complete.")
        else:
            logger.info("ChromaDB index seems up-to-date. Skipping embedding generation.")

    def _rrf_score(self, rank, k=60):
        """Reciprocal Rank Fusion calculation."""
        return 1 / (k + rank)

    def retrieve(self, query, top_k=5):
        """Performs hybrid retrieval using RRF."""
        logger.info(f"Executing hybrid search for: {query}")
        
        # 1. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Apply metadata boosts to BM25 scores
        for i, meta in enumerate(self.metadata):
            bm25_scores[i] *= meta["boost"]
            
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k*2] # Fetch more for reranking
        
        # 2. Vector Search
        query_embedding = self.embedding_model.encode([query]).tolist()
        vector_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k*2,
            include=["documents", "metadatas", "distances"]
        )
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Add BM25 ranks
        for rank, idx in enumerate(bm25_top_indices):
            doc_id = f"{self.metadata[idx]["source"]}_chunk_{self.metadata[idx]["chunk_index"]}"
            rrf_scores[doc_id] = self._rrf_score(rank + 1)
            
        # Add Vector ranks
        if vector_results["ids"] and vector_results["ids"][0]:
            for rank, doc_id in enumerate(vector_results["ids"][0]):
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += self._rrf_score(rank + 1)
                else:
                    rrf_scores[doc_id] = self._rrf_score(rank + 1)
                    
        # Sort by RRF score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        final_results = []
        for doc_id, rrf_score in sorted_rrf:
            # Extract index from doc_id (e.g. MEMORY.md_chunk_5 -> find the matching metadata)
            source, chunk_idx_str = doc_id.split("_chunk_")
            chunk_idx = int(chunk_idx_str)
            
            # Find the actual document text
            for i, meta in enumerate(self.metadata):
                if meta["source"] == source and meta["chunk_index"] == chunk_idx:
                    final_results.append({
                        "source": source,
                        "score": round(rrf_score, 4),
                        "content": self.documents[i]
                    })
                    break
                    
        return final_results

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "lxc password"
    retriever = HermitAdvancedHybridRetriever()
    results = retriever.retrieve(query)
    
    print(f"\n🔍 Advanced Hybrid Retrieval Results for: {query}")
    print("="*70)
    for i, res in enumerate(results):
        print(f"[{i+1}] Source: {res["source"]} | RRF Score: {res["score"]}")
        print(f"    {res["content"][:300]}...\n")
