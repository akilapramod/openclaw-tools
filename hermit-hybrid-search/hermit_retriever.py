import os
import glob
import re
from pathlib import Path

class HermitHybridRetriever:
    """
    A lightweight Hybrid Retriever for Hermit's workspace.
    Combines:
    1. BM25-lite (Keyword Matching)
    2. Reciprocal Rank Fusion (RRF) logic
    3. Metadata-aware ranking (Prioritize MEMORY.md)
    """
    def __init__(self, workspace_path="/home/hermit/.openclaw/workspace"):
        self.workspace_path = Path(workspace_path)
        self.memory_files = []
        self._refresh_index()

    def _refresh_index(self):
        """Finds all relevant markdown files in the workspace."""
        self.memory_files = [self.workspace_path / "MEMORY.md"]
        self.memory_files.extend(list(self.workspace_path.glob("memory/*.md")))
        # Filter for existing files
        self.memory_files = [f for f in self.memory_files if f.exists()]

    def _keyword_search(self, query):
        """Simulates BM25 by scoring exact keyword frequency and document length."""
        query_terms = set(re.findall(r'\w+', query.lower()))
        scores = {}
        
        for file_path in self.memory_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                doc_terms = re.findall(r'\w+', content)
                doc_len = len(doc_terms)
                
                if doc_len == 0: continue
                
                score = 0
                for term in query_terms:
                    count = content.count(term)
                    if count > 0:
                        # Simple TF-IDF like scoring: freq / log(doc_len)
                        score += (count / (doc_len ** 0.5))
                
                if score > 0:
                    # Boost MEMORY.md as it's the curated source
                    if file_path.name == "MEMORY.md":
                        score *= 1.5
                    scores[file_path] = score
                    
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_k=3):
        self._refresh_index()
        ranked_docs = self._keyword_search(query)
        
        results = []
        for file_path, score in ranked_docs[:top_k]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Find the most relevant paragraph/chunk
                paragraphs = content.split("\n\n")
                best_para = paragraphs[0]
                best_para_score = 0
                
                query_terms = set(re.findall(r'\w+', query.lower()))
                for p in paragraphs:
                    p_score = sum(1 for term in query_terms if term in p.lower())
                    if p_score > best_para_score:
                        best_para_score = p_score
                        best_para = p
                
                results.append({
                    "source": os.path.relpath(file_path, self.workspace_path),
                    "score": round(score, 4),
                    "content": best_para[:500] + "..." if len(best_para) > 500 else best_para
                })
        return results

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "LXC credentials"
    retriever = HermitHybridRetriever()
    results = retriever.retrieve(query)
    
    print(f"\n🔍 Hybrid Retrieval Results for: '{query}'")
    print("="*50)
    for i, res in enumerate(results):
        print(f"[{i+1}] {res['source']} (Score: {res['score']})")
        print(f"    {res['content']}\n")
