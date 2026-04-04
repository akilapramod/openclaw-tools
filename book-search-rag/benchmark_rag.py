import sys
import os

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from core.retriever import HybridRetriever
from storage.vector import VectorStore
from storage.keyword import KeywordStore

# Verified Benchmark Set
BENCHMARK_SET = [
    {"q": "What are some no-cost cyber services mentioned on the government website?", "source": "devsecops_basics.pdf"},
    {"q": "In what languages other than English is the SANS poster available?", "source": "sans_devsecops_poster.pdf"},
    {"q": "What type of Creative Commons license does the OWASP site use?", "source": "owasp_devsecops_guideline.pdf"},
    {"q": "Which GitHub feature is used to find and fix vulnerabilities?", "source": "sans_devsecops_poster.pdf"},
    {"q": "How can a user filter their search results more quickly according to the GitHub interface?", "source": "sans_devsecops_poster.pdf"}
]

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_path, "data/processed/chroma_db")
    kw_path = os.path.join(base_path, "data/processed/keyword_store.pkl")

    print("🚀 Initializing Retriever (loading models)...\n")
    v = VectorStore(db_path=db_path)
    k = KeywordStore(save_path=kw_path)
    r = HybridRetriever(vector_store=v, bm25_store=k)

    factors = [1, 3, 5, 10]
    results = {}

    print("🚀 Starting In-Process Overfetch Sensitivity Test...\n")
    
    for f in factors:
        hits = 0
        total = len(BENCHMARK_SET)
        print(f"--- Testing Overfetch Factor: {f} ---")
        
        for item in BENCHMARK_SET:
            results_list = r.retrieve(item['q'], top_k=3, overfetch_factor=f)
            # Check if source is in any of the results
            if any(item['source'] == res['metadata']['source'] for res in results_list):
                hits += 1
        
        accuracy = (hits / total) * 100
        results[f] = accuracy
        print(f"Accuracy: {accuracy:.2f}% Hit Rate @ 3\n")

    print("📊 Sensitivity Results Summary:")
    for f, acc in results.items():
        print(f"  Factor {f}: {acc:.2f}%")

if __name__ == '__main__':
    main()
