from flask import Flask, request, jsonify
from ephemeral_rag import run_ephemeral_rag

app = Flask(__name__)

@app.route('/filter', methods=['POST'])
def filter_documents():
    data = request.json
    if not data or 'documents' not in data or 'research_question' not in data:
        return jsonify({"error": "Missing 'documents' or 'research_question' in payload"}), 400
    
    try:
        filtered_text = run_ephemeral_rag(
            documents=data['documents'],
            research_question=data['research_question']
        )
        return jsonify({"filtered_text": filtered_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on port 5056 to avoid conflict with the Memory RAG currently on 5055
    app.run(host='0.0.0.0', port=5056)