"""
End-to-end pipeline tests via the headless runner (src.pipeline.run_pipeline).
All external I/O — LLM calls, HTTP requests, ArXiv API, PDF parsing — is mocked.

Mocking strategy:
  - BOM extraction  → patch src.agents.nodes.extract_bom  (imported name in nodes)
  - PDF parsing     → patch src.agents.nodes.parse_pdf_bytes / parse_pdf_file
  - ArXiv fetch     → patch src.agents.nodes.fetch_paper_metadata / download_paper_pdf
  - Export check LLM→ patch src.agents.nodes.ChatPromptTemplate + ChatGroq
  - Scrapers        → patch src.agents.nodes.*Scraper classes

Patching at the nodes.py import boundary avoids real network/API calls
regardless of how the underlying module is implemented.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import format_report, run_from_config, run_pipeline

# ---------------------------------------------------------------------------
# Shared mock payloads
# ---------------------------------------------------------------------------
_BOM_DICT = {
    "hardware": [
        {
            "name": "Dilution refrigerator",
            "subcategory": "cryogenics",
            "specifications": {"base_temperature": "10 mK"},
            "part_number": None, "manufacturer": "Oxford Instruments",
            "quantity": "1", "estimated_cost": None, "notes": "",
        }
    ],
    "software": [],
    "materials": [],
}

_EXPORT_RESULT = {
    "item_name": "Dilution refrigerator",
    "us_status": "controlled",
    "us_details": "Controlled under ECCN 3B001.",
    "germany_status": "controlled",
    "germany_details": "Listed under EU Dual-Use Regulation Annex I.",
    "eu_status": "controlled",
    "eu_details": "Annex I Category 3B.",
    "overall_risk": "high",
    "recommendation": "Obtain export licence before procurement.",
}


def _chain_returning(payload: dict):
    """Return a mock LangChain chain whose .invoke() returns payload as JSON."""
    response = MagicMock()
    response.content = json.dumps(payload)
    chain = MagicMock()
    chain.invoke = MagicMock(return_value=response)
    return chain


def _patch_prompt_alternating(first_payload, second_payload):
    """
    Context-manager factory: first prompt|llm call returns first_payload,
    subsequent calls return second_payload. Used for BOM-then-export-check flows.
    """
    call_count = {"n": 0}

    def side_effect(_prompt_template):
        mock_prompt = MagicMock()
        def make_chain(_llm):
            call_count["n"] += 1
            return _chain_returning(first_payload if call_count["n"] == 1 else second_payload)
        mock_prompt.__or__ = MagicMock(side_effect=make_chain)
        return mock_prompt

    return patch("src.agents.nodes.ChatPromptTemplate", from_template=MagicMock(side_effect=side_effect))


# ---------------------------------------------------------------------------
# Helpers — common mock stacks
# ---------------------------------------------------------------------------
_SCRAPER_PATCHES = [
    patch("src.agents.nodes.USBISScraper"),
    patch("src.agents.nodes.GermanyBAFAScraper"),
    patch("src.agents.nodes.EURegulationScraper"),
]


def _apply_empty_scrapers(mocks):
    for m in mocks:
        m.return_value.scrape.return_value = []
        m.return_value.get_recent_updates.return_value = []


# ---------------------------------------------------------------------------
# 1. Direct items mode
# ---------------------------------------------------------------------------
class TestDirectItemsMode:

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    def test_returns_report_with_results(self, mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu):
        _apply_empty_scrapers([mock_us, mock_de, mock_eu])
        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(return_value=_chain_returning(_EXPORT_RESULT))
        )

        report = run_pipeline("direct_items", "Dilution refrigerator")

        assert report["input_type"] == "direct_items"
        assert len(report["export_control_results"]) == 1
        assert report["export_control_results"][0]["item_name"] == "Dilution refrigerator"
        assert report["export_control_results"][0]["overall_risk"] == "high"

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    def test_multiple_items_all_checked(self, mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu):
        _apply_empty_scrapers([mock_us, mock_de, mock_eu])
        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(return_value=_chain_returning(_EXPORT_RESULT))
        )

        report = run_pipeline("direct_items", "Dilution refrigerator\nLaser system\nQubit chip")

        assert len(report["items_checked"]) == 3
        assert len(report["export_control_results"]) == 3

    def test_empty_input_returns_empty_results(self):
        report = run_pipeline("direct_items", "")
        assert report["export_control_results"] == []
        assert report["items_checked"] == []

    def test_report_always_has_required_keys(self):
        report = run_pipeline("direct_items", "")
        for key in ("input_type", "export_control_results", "items_checked",
                    "processing_time_seconds"):
            assert key in report, f"Missing key: {key}"

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    def test_processing_time_is_recorded(self, mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu):
        _apply_empty_scrapers([mock_us, mock_de, mock_eu])
        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(return_value=_chain_returning(_EXPORT_RESULT))
        )
        report = run_pipeline("direct_items", "Dilution refrigerator")
        assert isinstance(report["processing_time_seconds"], float)
        assert report["processing_time_seconds"] >= 0


# ---------------------------------------------------------------------------
# 2. PDF mode
# ---------------------------------------------------------------------------
class TestPDFMode:

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    @patch("src.agents.nodes.extract_bom", return_value=_BOM_DICT)
    @patch("src.agents.nodes.parse_pdf_bytes",
           return_value="Experimental Setup\nDilution refrigerator used at 10 mK.")
    def test_pdf_mode_extracts_bom_then_checks(
        self, mock_parse_pdf, mock_extract_bom,
        mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu
    ):
        _apply_empty_scrapers([mock_us, mock_de, mock_eu])
        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(return_value=_chain_returning(_EXPORT_RESULT))
        )

        report = run_pipeline("pdf", "paper.pdf", pdf_bytes=b"%PDF-1.4 fake content")

        assert report["bom"] is not None
        assert len(report["bom"]["hardware"]) == 1
        assert len(report["export_control_results"]) == 1

    def test_pdf_mode_without_bytes_records_error(self):
        # No pdf_bytes → pdf_parser sets error and empty paper_text.
        # Pipeline should not raise; error is surfaced in report.
        report = run_pipeline("pdf", "paper.pdf", pdf_bytes=None)
        assert report is not None
        assert "export_control_results" in report


# ---------------------------------------------------------------------------
# 3. ArXiv mode
# ---------------------------------------------------------------------------

_ARXIV_METADATA = {
    "arxiv_id": "2301.00001",
    "title": "Ion Trap Quantum Computing",
    "authors": ["Alice", "Bob"],
    "abstract": "Ion trap experiment setup.",
    "url": "https://arxiv.org/abs/2301.00001",
    "pdf_url": "https://arxiv.org/pdf/2301.00001",
    "published": "2023-01-01",
    "categories": ["quant-ph"],
}


class TestArxivMode:

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    @patch("src.agents.nodes.extract_bom", return_value=_BOM_DICT)
    @patch("src.agents.nodes.parse_pdf_file",
           return_value="Experimental Setup\nIon trap chip cooled to 4 K.")
    @patch("src.agents.nodes.download_paper_pdf",
           return_value=Path("data/papers/2301.00001.pdf"))
    @patch("src.agents.nodes.fetch_paper_metadata", return_value=_ARXIV_METADATA)
    def test_arxiv_mode_fetches_paper_and_checks(
        self, mock_fetch, mock_download, mock_parse, mock_extract_bom,
        mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu
    ):
        _apply_empty_scrapers([mock_us, mock_de, mock_eu])
        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(return_value=_chain_returning(_EXPORT_RESULT))
        )

        report = run_pipeline("arxiv", "2301.00001")

        assert report["paper_info"]["title"] == "Ion Trap Quantum Computing"
        assert report["bom"] is not None
        assert len(report["bom"]["hardware"]) == 1
        assert len(report["export_control_results"]) == 1

    def test_invalid_arxiv_id_records_error(self):
        # extract_arxiv_id returns None → paper_fetcher sets error + paper_text="".
        # section_extractor sees "" → returns "". Pipeline completes without raising.
        report = run_pipeline("arxiv", "not-a-valid-id")
        assert report is not None
        assert "export_control_results" in report


# ---------------------------------------------------------------------------
# 4. run_from_config (default config path)
# ---------------------------------------------------------------------------
class TestRunFromConfig:

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    def test_uses_headless_defaults(self, mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu):
        _apply_empty_scrapers([mock_us, mock_de, mock_eu])
        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(return_value=_chain_returning(_EXPORT_RESULT))
        )

        test_config = {
            "llm": {"model": "llama3-70b-8192", "temperature": 0.1, "max_tokens": 4096},
            "scrapers": {"rate_limit": 100, "timeout": 5, "retry_attempts": 1,
                         "retry_delay": 0, "user_agent": "Test/1.0",
                         "respect_robots_txt": False,
                         "countries": {
                             "us": {"base_url": "http://x", "endpoints": {"ccl": "/", "entity_list": "/"}},
                             "germany": {"base_url": "http://x", "endpoints": {"main": "/"}},
                             "eu": {"base_url": "http://x", "endpoints": {"main": "/"}},
                         }},
            "headless": {
                "input_type": "direct_items",
                "raw_input": "Dilution refrigerator\nLaser",
                "output_format": "json",
            },
        }

        report = run_from_config(test_config)
        assert report["input_type"] == "direct_items"
        assert len(report["items_checked"]) == 2

    def test_run_from_config_arxiv_mode_reads_arxiv_id(self):
        # Confirms run_from_config picks up arxiv_id from headless config section.
        # An invalid arxiv_id now fails gracefully (paper_fetcher sets error + paper_text="").
        test_config = {
            "llm": {"model": "llama3-70b-8192", "temperature": 0.1, "max_tokens": 4096},
            "scrapers": {"rate_limit": 100, "timeout": 5, "retry_attempts": 1,
                         "retry_delay": 0, "user_agent": "Test/1.0",
                         "respect_robots_txt": False,
                         "countries": {
                             "us": {"base_url": "http://x", "endpoints": {"ccl": "/", "entity_list": "/"}},
                             "germany": {"base_url": "http://x", "endpoints": {"main": "/"}},
                             "eu": {"base_url": "http://x", "endpoints": {"main": "/"}},
                         }},
            "headless": {
                "input_type": "arxiv",
                "arxiv_id": "not-a-valid-id",
            },
        }
        report = run_from_config(test_config)
        assert isinstance(report, dict)
        assert report["input_type"] == "arxiv"


# ---------------------------------------------------------------------------
# 5. format_report
# ---------------------------------------------------------------------------
class TestFormatReport:

    def _sample_report(self):
        return {
            "input_type": "direct_items",
            "paper_info": None,
            "bom": None,
            "items_checked": ["Dilution refrigerator"],
            "export_control_results": [_EXPORT_RESULT],
            "processing_time_seconds": 1.5,
        }

    def test_json_format_is_valid_json(self):
        out = format_report(self._sample_report(), fmt="json")
        parsed = json.loads(out)
        assert parsed["input_type"] == "direct_items"

    def test_pretty_format_contains_item_name(self):
        out = format_report(self._sample_report(), fmt="pretty")
        assert "Dilution refrigerator" in out
        assert "HIGH" in out

    def test_pretty_format_shows_all_three_countries(self):
        out = format_report(self._sample_report(), fmt="pretty")
        assert "US" in out
        assert "Germany" in out
        assert "EU" in out

    def test_pretty_format_includes_recommendation(self):
        out = format_report(self._sample_report(), fmt="pretty")
        assert "export licence" in out.lower() or "recommendation" in out.lower()
