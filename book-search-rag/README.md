# Book Search RAG — High-Precision Hybrid Retrieval

A modular, production-ready RAG framework designed for searching through PDF libraries using hybrid retrieval (Vector + BM25) and Cross-Encoder reranking.

## 🚀 Features

- **Hybrid Retrieval**: Combines semantic (ChromaDB) and lexical (BM25) search.
- **Precision Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to filter and re-order results.
- **REST API**: Built with Flask and Pydantic for easy integration.
- **Modular Design**: Clean separation of core logic, ingestion, and storage.
- **Dockerized**: Ready for deployment in any environment.
- **CI/CD**: Linting and testing pipeline via GitHub Actions.

## 📁 Project Structure

- `src/core/`: Retrieval and reranking logic.
- `src/storage/`: Database adapters for ChromaDB and BM25.
- `src/ingestion/`: PDF parsing and semantic chunking.
- `src/api/`: REST API service layer.
- `tests/`: Unit and integration test suite.
- `data/`: Raw PDF storage and processed indexes.

## 🛠️ Getting Started

### 1. Installation
```bash
make setup
```

### 2. Ingesting Books
Place your PDF files in `data/raw/` and run:
```bash
make ingest
```

### 3. Running the API
```bash
make run
```
The API will be available at `http://localhost:5000`.

### 4. Running Tests
```bash
make test
```

## 🐳 Docker Support

Build the image:
```bash
make docker-build
```

Run the container:
```bash
make docker-run
```

## 🧪 CI/CD
Automatically runs linting (`flake8`) and tests (`pytest`) on every push to `main` and `develop`.
