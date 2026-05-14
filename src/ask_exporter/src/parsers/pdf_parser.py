"""PDF text extraction using PyMuPDF (fitz)."""

from pathlib import Path

import fitz  # PyMuPDF

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


def parse_pdf_file(pdf_path: str | Path, max_pages: int = 100) -> str:
    """Extract plain text from a PDF file path."""
    doc = fitz.open(str(pdf_path))
    n = min(len(doc), max_pages)
    pages = [doc[i].get_text("text") for i in range(n)]
    doc.close()
    logger.info("Parsed %d pages from %s", n, pdf_path)
    return "\n\n".join(pages)


def parse_pdf_bytes(pdf_bytes: bytes, max_pages: int = 100) -> str:
    """Extract plain text from PDF bytes (e.g. Streamlit upload)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = min(len(doc), max_pages)
    pages = [doc[i].get_text("text") for i in range(n)]
    doc.close()
    logger.info("Parsed %d pages from uploaded PDF", n)
    return "\n\n".join(pages)
