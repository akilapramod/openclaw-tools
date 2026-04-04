import os
import glob

class MockHybridRetriever:
    def __init__(self, workspace_path):
        self.workspace_path = workspace_path
        self.memory_files = []
        self._load_files()

    def _load_files(self):
        self.memory_files.append(os.path.join(self.workspace_path, "MEMORY.md"))
        self.memory_files.extend(glob.glob(os.path.join(self.workspace_path, "memory/*.md")))

    def search(self, query):
        print(f"--- Simulating Hybrid Search for: '{query}' ---")
        results = []
        query_terms = query.lower().split()
        
        for file_path in self.memory_files:
            if not os.path.exists(file_path): continue
            
            with open(file_path, "r") as f:
                content = f.read()
                # BM25-like: check for exact keyword matches
                keyword_score = sum(1 for term in query_terms if term in content.lower())
                
                # Vector-like: check for semantic overlap (simplified)
                # In a real RAG, this would be embedding distance.
                
                if keyword_score > 0:
                    results.append({
                        "source": os.path.relpath(file_path, self.workspace_path),
                        "score": keyword_score,
                        "snippet": content[:300].replace("\n", " ") + "..."
                    })
        
        # Sort by score (BM25 keyword matches)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

if __name__ == "__main__":
    retriever = MockHybridRetriever("/home/hermit/.openclaw/workspace")
    
    # Test Query 1: Keyword-heavy
    q1 = "LXC IP credentials"
    res1 = retriever.search(q1)
    print(f"\nResults for '{q1}':")
    for r in res1[:2]:
        print(f"[{r['score']}] {r['source']}: {r['snippet']}")

    # Test Query 2: Concept-heavy
    q2 = "automated session export script"
    res2 = retriever.search(q2)
    print(f"\nResults for '{q2}':")
    for r in res2[:2]:
        print(f"[{r['score']}] {r['source']}: {r['snippet']}")
