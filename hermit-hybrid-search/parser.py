"""
PDF parser module using PyMuPDF (fitz).
Extracts text page-by-page with metadata: book title, chapter, page number.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class PageDocument:
    """Represents a single page extracted from a PDF."""
    text: str
    book_title: str
    page_number: int
    chapter: str = "Unknown"


# ─── Chapter Detection Heuristics ────────────────────────────────────────────

# Patterns that indicate a chapter heading
CHAPTER_PATTERNS = [
    re.compile(r"^chapter\s+(\d+|[ivxlcdm]+)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^CHAPTER\s+(\d+|[IVXLCDM]+)", re.MULTILINE),
    re.compile(r"^Part\s+(\d+|[IVXLCDM]+)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Section\s+(\d+\.?\d*)", re.IGNORECASE | re.MULTILINE),
]


def _detect_chapter(text: str, current_chapter: str) -> str:
    """
    Detect if the page text contains a chapter heading.
    Returns the new chapter name, or the current one if no heading found.
    """
    for pattern in CHAPTER_PATTERNS:
        match = pattern.search(text[:500])  # Only check start of page
        if match:
            # Extract the full line containing the match
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = min(match.end() + 100, len(text))
            chapter_line = text[line_start:line_end].strip()
            # Clean up and return
            if len(chapter_line) < 120:  # Reasonable chapter title length
                return chapter_line
    return current_chapter


def _extract_book_title(pdf_path: Path, doc: fitz.Document) -> str:
    """
    Extract book title from PDF metadata or filename.
    Prefers PDF metadata 'title' if available, falls back to filename.
    """
    metadata = doc.metadata
    if metadata and metadata.get("title") and metadata["title"].strip():
        return metadata["title"].strip()
    # Fallback to filename without extension
    return pdf_path.stem.replace("_", " ").replace("-", " ").title()


def parse_pdf(pdf_path: str | Path) -> List[PageDocument]:
    """
    Parse a PDF file and extract text page-by-page with metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageDocument objects, one per page with non-empty text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    book_title = _extract_book_title(pdf_path, doc)
    pages: List[PageDocument] = []
    current_chapter = "Introduction"

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if not text:
            continue

        # Detect chapter changes
        current_chapter = _detect_chapter(text, current_chapter)

        pages.append(PageDocument(
            text=text,
            book_title=book_title,
            page_number=page_num + 1,  # 1-indexed
            chapter=current_chapter,
        ))

    doc.close()
    return pages


def parse_directory(directory: str | Path) -> List[PageDocument]:
    """
    Parse all PDF files in a directory.

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        Combined list of PageDocument objects from all PDFs.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_pages: List[PageDocument] = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in {directory}")
        return all_pages

    for pdf_path in pdf_files:
        print(f"📄 Parsing: {pdf_path.name}")
        try:
            pages = parse_pdf(pdf_path)
            all_pages.extend(pages)
            print(f"   → Extracted {len(pages)} pages")
        except Exception as e:
            print(f"   ❌ Error parsing {pdf_path.name}: {e}")

    return all_pages
