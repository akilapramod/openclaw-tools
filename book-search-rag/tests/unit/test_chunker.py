import sys
import os
import unittest

# Add src/ to path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from ingestion.chunker import SemanticChunker

class TestSemanticChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    def test_split_by_paragraph(self):
        text = "This is a paragraph.\n\nThis is another paragraph."
        chunks = self.chunker.split_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is a paragraph.")

    def test_recursive_split_by_sentence(self):
        # Long text that exceeds 100 characters should split by sentence
        text = "Sentence one is here. Sentence two is very long and detailed. Sentence three is right here."
        chunks = self.chunker.split_text(text)
        self.assertGreater(len(chunks), 1)
        # Verify sentences aren't mangled
        for chunk in chunks:
            self.assertTrue(any(s in chunk for s in ["Sentence one", "Sentence two", "Sentence three"]))

if __name__ == "__main__":
    unittest.main()
