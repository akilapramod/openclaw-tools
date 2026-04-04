import sys
import os
import logging
from flask import Flask, request, jsonify

# Add src/ to path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.retriever import HybridRetriever
from storage.vector import VectorStore
from storage.keyword import KeywordStore
from api.schemas import SearchRequest, SearchResponse

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("book-rag.api")

# Initialize global components
base_path = os.path.join(os.path.dirname(__file__), "../..")
db_path = os.path.join(base_path, "data/processed/chroma_db")
keyword_path = os.path.join(base_path, "data/processed/keyword_store.pkl")

vector_store = VectorStore(db_path=db_path)
keyword_store = KeywordStore(save_path=keyword_path)

retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_store=keyword_store
)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.json
        search_req = SearchRequest(**data)
        
        log.info(f"API Search: '{search_req.query}' (k={search_req.top_k}, sources={search_req.sources})")
        
        results = retriever.retrieve(
            search_req.query, 
            top_k=search_req.top_k,
            min_score=search_req.min_score,
            overfetch_factor=search_req.overfetch_factor,
            sources=search_req.sources
        )
        
        response_results = [{
            "id": r["id"],
            "text": r["text"],
            "metadata": r["metadata"],
            "rerank_score": r["rerank_score"]
        } for r in results]
            
        return jsonify(SearchResponse(
            query=search_req.query,
            results=response_results
        ).dict())

    except Exception as e:
        log.error(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
