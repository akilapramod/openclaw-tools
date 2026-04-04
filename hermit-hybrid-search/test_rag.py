import os
import glob
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import pandas as pd

print("Imports successful!")

# Initialize ChromaDB client
chroma_client = chromadb.Client()
print("ChromaDB Client Initialized")

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer Model Loaded")

# Create a test collection
collection = chroma_client.create_collection(name="hermit_memory_test")

# Read a test memory file
memory_dir = os.path.expanduser("~/memory_files")
files = glob.glob(os.path.join(memory_dir, "*.md"))

if not files:
    print("No memory files found!")
else:
    print(f"Found {len(files)} memory files. Processing the first one...")
    test_file = files[0]
    with open(test_file, "r") as f:
        content = f.read()

    # Create chunks (naive splitting by newlines for quick test)
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    # Generate embeddings and add to collection
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"doc1_chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Added {len(chunks)} chunks to ChromaDB collection.")

    # Test Query
    query = "What is the lxc ip address?"
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    
    print("\n--- Test Vector Search Results ---")
    print(f"Query: {query}")
    for idx, doc in enumerate(results["documents"][0]):
        print(f"\nResult {idx+1}:")
        print(f"{doc[:150]}...")

print("\nFull RAG pipeline test complete.")
