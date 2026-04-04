import sys
import os
import unittest
from unittest.mock import MagicMock

# Add src/ to path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core.retriever import HybridRetriever

class TestHybridRetrieverIntegration(unittest.TestCase):
    """Integration-style tests for the HybridRetriever logic (using mocks for external stores)."""

    def setUp(self):
        self.mock_vector_store = MagicMock()
        self.mock_keyword_store = MagicMock()
        
        # Initialise retriever
        self.retriever = HybridRetriever(
            vector_store=self.mock_vector_store,
            bm25_store=self.mock_keyword_store
        )

    def test_retrieve_pipeline(self):
        # Mocking store search results
        self.mock_vector_store.search.return_value = [
            {"id": "doc1", "text": "I love books.", "metadata": {"source": "test.pdf", "page": 1}}
        ]
        self.mock_keyword_store.search.return_value = [
            {"id": "doc1", "text": "I love books.", "metadata": {"source": "test.pdf", "page": 1}}
        ]
        
        results = self.retriever.retrieve("love", top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1")
        self.assertIn("rerank_score", results[0])

if __name__ == "__main__":
    unittest.main()
