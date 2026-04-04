"""
Recursive Character Chunker — Context-Aware Document Splitting.

Splits text into chunks while respecting paragraph, sentence, 
and word boundaries. Ensures context is preserved and chunks 
don't cut off in the middle of vital information.
"""

import re
import logging
from typing import List, Optional

class SemanticChunker:
    """Recursively splits text to maintain semantic coherence."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.log = logging.getLogger("book-rag.chunker")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into manageable, context-rich chunks."""
        final_chunks = []
        
        # Initial split on the largest separator (paragraphs)
        raw_chunks = self._recursive_split(text, self.separators)
        
        # Merge small chunks to reach target size with overlap
        current_chunk = ""
        for rc in raw_chunks:
            if len(current_chunk) + len(rc) < self.chunk_size:
                current_chunk += rc + " "
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + rc + " "
        
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Internal recursive splitting logic."""
        if not separators:
            return [text]
            
        sep = separators[0]
        new_separators = separators[1:]
        
        if sep == "":
            return list(text) # Character-level fallback
            
        # Split by separator while keeping it (for sentences)
        parts = text.split(sep)
        result = []
        
        for p in parts:
            if not p.strip():
                continue
            if len(p) <= self.chunk_size:
                result.append(p)
            else:
                # Recurse with finer separators
                result.extend(self._recursive_split(p, new_separators))
                
        return result
