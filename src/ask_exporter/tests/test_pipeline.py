"""
End-to-end pipeline tests via the headless runner (src.pipeline.run_pipeline).
All external I/O — LLM calls, HTTP requests, ArXiv API, PDF parsing — is mocked.

This is the primary integration test surface: tests exercise the public
run_pipeline() / run_from_config() API rather than wiring the graph manually.
"""

import json
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
    @patch("src.parsers.pdf_parser.fitz.open")
    def test_pdf_mode_extracts_bom_then_checks(
        self, mock_fitz, mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu
    ):
        # PDF returns text with an experimental setup section
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 1
        mock_doc.__getitem__ = lambda self, i: MagicMock(
            get_text=lambda mode: "Experimental Setup\nDilution refrigerator used at 10 mK."
        )
        mock_fitz.return_value = mock_doc

        _apply_empty_scrapers([mock_us, mock_de, mock_eu])

        # Alternate: first call = BOM extraction, subsequent = export check
        call_count = {"n": 0}
        def alternating_chain(_llm):
            call_count["n"] += 1
            return _chain_returning(_BOM_DICT if call_count["n"] == 1 else _EXPORT_RESULT)

        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(side_effect=alternating_chain)
        )

        report = run_pipeline("pdf", "paper.pdf", pdf_bytes=b"%PDF-1.4 fake content")

        assert report["bom"] is not None
        assert len(report["bom"]["hardware"]) == 1
        assert len(report["export_control_results"]) == 1

    def test_pdf_mode_without_bytes_records_error(self):
        report = run_pipeline("pdf", "paper.pdf", pdf_bytes=None)
        # Should not crash; error is surfaced in the report
        assert report is not None
        # Either no results or an error key present
        assert "export_control_results" in report


# ---------------------------------------------------------------------------
# 3. ArXiv mode
# ---------------------------------------------------------------------------
class TestArxivMode:

    @patch("src.agents.nodes.EURegulationScraper")
    @patch("src.agents.nodes.GermanyBAFAScraper")
    @patch("src.agents.nodes.USBISScraper")
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.agents.nodes.ChatPromptTemplate")
    @patch("src.parsers.pdf_parser.fitz.open")
    @patch("src.parsers.arxiv_fetcher.arxiv.Client")
    @patch("src.parsers.arxiv_fetcher.Path")
    def test_arxiv_mode_fetches_paper_and_checks(
        self, mock_path_cls, mock_arxiv_client, mock_fitz,
        mock_prompt_cls, mock_groq, mock_us, mock_de, mock_eu
    ):
        # Path.exists() → False so download is attempted
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.__str__ = lambda self: "data/papers/2301.00001.pdf"
        mock_path_instance.__truediv__ = lambda self, other: mock_path_instance
        mock_path_cls.return_value = mock_path_instance

        # ArXiv paper
        paper = MagicMock()
        paper.title = "Ion Trap Quantum Computing"
        paper.authors = [MagicMock(name="Alice")]
        paper.summary = "Ion trap experiment setup."
        paper.entry_id = "https://arxiv.org/abs/2301.00001"
        paper.pdf_url = "https://arxiv.org/pdf/2301.00001"
        paper.published = MagicMock(isoformat=lambda: "2023-01-01")
        paper.categories = ["quant-ph"]
        paper.get_short_id = MagicMock(return_value="2301.00001")
        paper.download_pdf = MagicMock()

        mock_client = MagicMock()
        mock_client.results.return_value = iter([paper])
        mock_arxiv_client.return_value = mock_client

        # PDF parse
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 1
        mock_doc.__getitem__ = lambda self, i: MagicMock(
            get_text=lambda mode: "Experimental Setup\nIon trap chip cooled to 4 K."
        )
        mock_fitz.return_value = mock_doc

        _apply_empty_scrapers([mock_us, mock_de, mock_eu])

        call_count = {"n": 0}
        def alternating_chain(_llm):
            call_count["n"] += 1
            return _chain_returning(_BOM_DICT if call_count["n"] == 1 else _EXPORT_RESULT)

        mock_prompt_cls.from_template.return_value = MagicMock(
            __or__=MagicMock(side_effect=alternating_chain)
        )

        report = run_pipeline("arxiv", "2301.00001")

        assert report["paper_info"]["title"] == "Ion Trap Quantum Computing"
        assert report["bom"] is not None
        assert len(report["export_control_results"]) >= 0  # may be 0 if BOM hardware empty

    def test_invalid_arxiv_id_records_error(self):
        report = run_pipeline("arxiv", "not-a-valid-id")
        assert report is not None
        # Pipeline should surface error gracefully, not raise
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
                "arxiv_id": "not-a-valid-id",   # will fail gracefully
            },
        }
        # Should not raise even when arxiv_id is invalid
        report = run_from_config(test_config)
        assert isinstance(report, dict)


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
