import sys
import os
import unittest

# Add src/ to path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from book_search_rag.ingestion.chunker import SemanticChunker

class TestSemanticChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    def test_split_by_paragraph(self):
        text = (
            "This is a paragraph that is quite long and easily exceeds the limit. It is the first paragraph.\n\n"
            "This is another paragraph that is also quite long and easily exceeds the limit. It is the second paragraph."
        )
        # With a large chunk size, it should remain as a single chunk
        self.chunker.chunk_size = 1000
        chunks_large = self.chunker.split_text(text)
        self.assertEqual(len(chunks_large), 1)
        
        # With a small chunk size, it should be split into multiple chunks
        self.chunker.chunk_size = 50
        chunks_small = self.chunker.split_text(text)
        self.assertGreater(len(chunks_small), 1)

    def test_recursive_split_by_sentence(self):
        text = "Sentence one is here and it is long. Sentence two is also very long and detailed. Sentence three is right here."
        # Verify it splits when chunk size is small
        self.chunker.chunk_size = 40
        chunks = self.chunker.split_text(text)
        self.assertGreater(len(chunks), 1)
        
        # Verify no information is lost (all sentences are present somewhere in the chunks)
        combined = " ".join(chunks)
        self.assertTrue("Sentence one" in combined)
        self.assertTrue("Sentence two" in combined)
        self.assertTrue("Sentence three" in combined)

if __name__ == "__main__":
    unittest.main()
