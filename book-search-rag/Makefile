.PHONY: setup test run ingest clean docker-build docker-run

setup:
	pip install -r requirements.txt

test:
	pytest tests/

run:
	python src/api/server.py

ingest:
	python main.py ingest --raw-dir ./data/raw

clean:
	rm -rf data/processed/chroma_db
	find . -type d -name "__pycache__" -exec rm -rf {} +

docker-build:
	docker build -t book-rag-api .

docker-run:
	docker run -p 5000:5000 book-rag-api
