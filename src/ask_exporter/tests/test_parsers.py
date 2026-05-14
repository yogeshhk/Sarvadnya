"""Tests for arxiv_fetcher, pdf_parser, section_extractor — all external I/O mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.parsers.section_extractor import extract_relevant_sections
from src.utils.validators import extract_arxiv_id, parse_item_list, validate_pdf


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------
class TestArxivIdExtractor:
    def test_bare_id(self):
        assert extract_arxiv_id("2301.12345") == "2301.12345"

    def test_bare_id_with_version(self):
        assert extract_arxiv_id("2301.12345v2") == "2301.12345v2"

    def test_abs_url(self):
        assert extract_arxiv_id("https://arxiv.org/abs/2301.12345") == "2301.12345"

    def test_pdf_url(self):
        assert extract_arxiv_id("https://arxiv.org/pdf/2301.12345") == "2301.12345"

    def test_invalid(self):
        assert extract_arxiv_id("not-an-id") is None

    def test_empty(self):
        assert extract_arxiv_id("") is None


class TestValidatePdf:
    def test_valid_pdf(self):
        ok, err = validate_pdf(b"%PDF-1.4 fake content")
        assert ok is True
        assert err == ""

    def test_not_pdf(self):
        ok, err = validate_pdf(b"PK\x03\x04 zip content")
        assert ok is False
        assert "valid PDF" in err

    def test_too_large(self):
        ok, err = validate_pdf(b"%PDF" + b"x" * (51 * 1024 * 1024))
        assert ok is False
        assert "MB" in err


class TestParseItemList:
    def test_newline_separated(self):
        items = parse_item_list("Dilution refrigerator\nLaser system\nQubit chip")
        assert items == ["Dilution refrigerator", "Laser system", "Qubit chip"]

    def test_comma_separated(self):
        items = parse_item_list("Dilution refrigerator, Laser system, Qubit chip")
        assert len(items) == 3

    def test_strips_bullets(self):
        items = parse_item_list("- Dilution refrigerator\n• Laser system\n* Qubit chip")
        assert items == ["Dilution refrigerator", "Laser system", "Qubit chip"]

    def test_strips_numbers(self):
        items = parse_item_list("1. Item one\n2. Item two")
        assert items == ["Item one", "Item two"]

    def test_empty_input(self):
        assert parse_item_list("") == []

    def test_blank_lines_ignored(self):
        items = parse_item_list("Item A\n\n\nItem B\n")
        assert items == ["Item A", "Item B"]


# ---------------------------------------------------------------------------
# section_extractor
# ---------------------------------------------------------------------------
_PAPER_WITH_SECTIONS = """
Abstract
This paper describes a quantum optics experiment.

1. Introduction
We study photon entanglement in optical cavities.

2. Experimental Setup
The apparatus consists of a Ti:Sapphire laser operating at 780 nm.
A dilution refrigerator (Oxford Instruments) cooled the sample to 10 mK.
Superconducting qubits were fabricated on silicon substrate.

3. Results
The coherence time was measured to be 100 microseconds.

4. Conclusion
We demonstrated improved qubit performance.

References
[1] Smith et al. ...
"""

_PAPER_NO_SECTIONS = """
Abstract
This is a theoretical paper with no experimental section.
We derive equations for quantum error correction.
"""


class TestSectionExtractor:
    def test_extracts_setup_section(self):
        result = extract_relevant_sections(_PAPER_WITH_SECTIONS)
        assert "Ti:Sapphire" in result
        assert "dilution refrigerator" in result.lower()

    def test_stops_at_results(self):
        result = extract_relevant_sections(_PAPER_WITH_SECTIONS)
        # Should not include the Results section
        assert "coherence time" not in result

    def test_fallback_when_no_sections(self):
        result = extract_relevant_sections(_PAPER_NO_SECTIONS, max_chars=200)
        # Falls back to first 200 chars
        assert len(result) <= 200

    def test_respects_max_chars(self):
        long_paper = _PAPER_WITH_SECTIONS * 20
        result = extract_relevant_sections(long_paper, max_chars=500)
        assert len(result) <= 500


# ---------------------------------------------------------------------------
# arxiv_fetcher (mocked)
# ---------------------------------------------------------------------------
class TestArxivFetcher:
    @patch("src.parsers.arxiv_fetcher.arxiv.Client")
    def test_fetch_metadata_success(self, mock_client_cls):
        paper = MagicMock()
        paper.title = "Quantum Error Correction"
        paper.authors = [MagicMock(name="Alice"), MagicMock(name="Bob")]
        paper.summary = "We demonstrate quantum error correction."
        paper.entry_id = "https://arxiv.org/abs/2301.12345"
        paper.pdf_url = "https://arxiv.org/pdf/2301.12345"
        paper.published = MagicMock(isoformat=lambda: "2023-01-01T00:00:00")
        paper.categories = ["quant-ph"]

        mock_client = MagicMock()
        mock_client.results.return_value = iter([paper])
        mock_client_cls.return_value = mock_client

        from src.parsers.arxiv_fetcher import fetch_paper_metadata
        meta = fetch_paper_metadata("2301.12345")
        assert meta["arxiv_id"] == "2301.12345"
        assert meta["title"] == "Quantum Error Correction"
        assert "Alice" in meta["authors"]

    @patch("src.parsers.arxiv_fetcher.arxiv.Client")
    def test_fetch_metadata_not_found(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.results.return_value = iter([])
        mock_client_cls.return_value = mock_client

        from src.parsers.arxiv_fetcher import fetch_paper_metadata
        with pytest.raises(ValueError, match="not found"):
            fetch_paper_metadata("9999.99999")


# ---------------------------------------------------------------------------
# pdf_parser (mocked fitz)
# ---------------------------------------------------------------------------
class TestPdfParser:
    @patch("src.parsers.pdf_parser.fitz.open")
    def test_parse_pdf_bytes(self, mock_fitz_open):
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 2
        mock_doc.__getitem__ = lambda self, i: MagicMock(get_text=lambda mode: f"Page {i} text")
        mock_fitz_open.return_value = mock_doc

        from src.parsers.pdf_parser import parse_pdf_bytes
        result = parse_pdf_bytes(b"%PDF fake")
        assert "Page 0 text" in result
        assert "Page 1 text" in result

    @patch("src.parsers.pdf_parser.fitz.open")
    def test_respects_max_pages(self, mock_fitz_open):
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 200
        mock_doc.__getitem__ = lambda self, i: MagicMock(get_text=lambda mode: f"Page {i}")
        mock_fitz_open.return_value = mock_doc

        from src.parsers.pdf_parser import parse_pdf_bytes
        result = parse_pdf_bytes(b"%PDF fake", max_pages=5)
        assert "Page 4" in result
        assert "Page 5" not in result
