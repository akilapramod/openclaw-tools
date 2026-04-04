import os
import glob
from retriever import HybridRetriever
from vector_store import VectorStore
from bm25_store import BM25Store
from chunker import SemanticChunker
from config import Config

class HermitBrain:
    def __init__(self, workspace_path="/home/hermit/.openclaw/workspace"):
        self.workspace_path = workspace_path
        self.config = Config()
        self.vector_store = VectorStore(self.config)
        self.bm25_store = BM25Store(self.config)
        self.chunker = SemanticChunker(self.config)
        self.retriever = HybridRetriever(self.config, self.vector_store, self.bm25_store)

    def ingest_workspace(self):
        """Ingests MEMORY.md and all files in memory/ directory."""
        files = [os.path.join(self.workspace_path, "MEMORY.md")]
        files.extend(glob.glob(os.path.join(self.workspace_path, "memory/*.md")))
        
        all_chunks = []
        for file_path in files:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as f:
                text = f.read()
                # Simplified chunking for this prototype
                chunks = self.chunker.split_text(text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {"source": os.path.relpath(file_path, self.workspace_path), "chunk": i}
                    })
        
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            self.bm25_store.add_documents(all_chunks)
            print(f"Ingested {len(all_chunks)} chunks from {len(files)} files.")

    def ask(self, query):
        results = self.retriever.retrieve(query)
        return results

if __name__ == "__main__":
    brain = HermitBrain()
    brain.ingest_workspace()
    query = "What is the LXC container IP and credentials?"
    print(f"\nQuery: {query}")
    results = brain.ask(query)
    for i, res in enumerate(results):
        print(f"\n[{i+1}] (Score: {res.get('score', 'N/A')}) Source: {res['metadata']['source']}")
        print(f"Content: {res['text'][:200]}...")
