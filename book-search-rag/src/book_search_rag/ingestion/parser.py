"""
PDF Ingestion Module — Document Parsing with Context Injection.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import fitz
    import torch
    from tqdm import tqdm
except ImportError:
    logging.error("Missing dependencies: pip install pymupdf torch tqdm")
    raise

class DocumentIngestor:
    """PDF Document Parser with Source Context Injection."""

    def __init__(self, vector_store, keyword_store, chunker, embedder, raw_dir="./data/raw"):
        self.log = logging.getLogger("book-rag.ingestion")
        self.raw_dir = Path(raw_dir)
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.chunker = chunker
        self.embedder = embedder
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def ingest_all(self, force_reindex=False):
        pdf_files = sorted(list(self.raw_dir.glob("*.pdf")))
        if not pdf_files: return

        if force_reindex:
            self.vector_store.delete_all()
            self.keyword_store.clear()

        for pdf_path in tqdm(pdf_files, desc="Parsing"):
            self._process_file(pdf_path)

    def _process_file(self, file_path):
        doc_name = file_path.name
        try:
            doc = fitz.open(file_path)
            all_chunks, all_meta, all_ids = [], [], []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if not text: continue

                chunks = self.chunker.split_text(text)
                for i, chunk in enumerate(chunks):
                    # INJECTION: Prepend source context to improve semantic retrieval
                    context_chunk = f"[Document: {doc_name}] {chunk}"
                    
                    cid = f"{doc_name}::page_{page_num+1}::chunk_{i}"
                    meta = {
                        "source": doc_name,
                        "page": page_num + 1,
                        "chunk_index": i
                    }
                    all_chunks.append(context_chunk)
                    all_meta.append(meta)
                    all_ids.append(cid)
            
            if not all_chunks: return

            embeddings = self.embedder.encode(all_chunks).tolist()
            self.vector_store.add_documents(all_chunks, all_meta, all_ids, embeddings)
            self.keyword_store.add_documents(all_chunks, all_meta, all_ids)
            doc.close()
        except Exception as e:
            self.log.error(f"Failed {doc_name}: {str(e)}")
