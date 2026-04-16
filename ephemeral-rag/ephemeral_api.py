import sys
import logging
from flask import Flask, request, jsonify
from ephemeral_rag import run_ephemeral_rag, RAGConfig

# Force logs to stdout/stderr for journalctl visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models": ["all-MiniLM-L6-v2", "ms-marco-MiniLM-L-6-v2"]
    })

@app.route('/filter', methods=['POST'])
def filter_documents():
    data = request.json
    if not data or 'documents' not in data or 'research_question' not in data:
        return jsonify({"error": "Missing 'documents' or 'research_question' in payload"}), 400
    
    top_k = data.get('top_k', 5)
    
    try:
        config = RAGConfig(
            rerank_top_k=top_k
        )

        filtered_text = run_ephemeral_rag(
            documents=data['documents'],
            research_question=data['research_question'],
            config=config
        )
        return jsonify({"filtered_text": filtered_text})
    except Exception as e:
        print(f"[RAG] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5056)
