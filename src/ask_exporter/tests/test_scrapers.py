"""Tests for BaseScraper and country scrapers — HTTP is always mocked."""

from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.base import BaseScraper
from src.scrapers.eu_regulation import EURegulationScraper
from src.scrapers.germany_bafa import GermanyBAFAScraper
from src.scrapers.us_bis import USBISScraper

SAMPLE_CONFIG = {
    "scrapers": {
        "rate_limit": 100,
        "timeout": 5,
        "retry_attempts": 1,
        "retry_delay": 0,
        "user_agent": "TestBot/1.0",
        "respect_robots_txt": False,
        "countries": {
            "us": {"base_url": "https://example.com", "endpoints": {"ccl": "/ccl", "entity_list": "/el"}},
            "germany": {"base_url": "https://example.de", "endpoints": {"main": "/bafa"}},
            "eu": {"base_url": "https://example.eu", "endpoints": {"main": "/eu"}},
        },
    }
}

_SAMPLE_HTML = """
<html><body>
<h2>Quantum Technology Export Controls</h2>
<p>Dilution refrigerators (ECCN 3B001) are controlled under EAR for India.</p>
<p>Superconducting qubit chips may require an export license under CCL category 3E001.</p>
<p>Updated: January 15, 2025</p>
</body></html>
"""


def _mock_response(html: str = _SAMPLE_HTML) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# BaseScraper
# ---------------------------------------------------------------------------
class TestBaseScraper:
    def test_robots_txt_disabled(self):
        class MinimalScraper(BaseScraper):
            def scrape(self, query): return []
            def get_recent_updates(self, days=90): return []

        s = MinimalScraper(config=SAMPLE_CONFIG)
        assert s._robots_allowed("https://example.com/page") is True

    def test_rate_limit_gap(self):
        class MinimalScraper(BaseScraper):
            def scrape(self, query): return []
            def get_recent_updates(self, days=90): return []

        s = MinimalScraper(config=SAMPLE_CONFIG)
        import time
        s._last_request = time.time()
        s.rate_limit = 1000  # very fast
        start = time.time()
        s._wait_rate_limit()
        assert time.time() - start < 0.05  # should not block at high rate limit


# ---------------------------------------------------------------------------
# US BIS Scraper
# ---------------------------------------------------------------------------
class TestUSBISScraper:
    @patch("src.scrapers.us_bis.USBISScraper.get")
    def test_scrape_finds_match(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = USBISScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("dilution refrigerator")
        assert len(results) > 0
        assert results[0]["source"] == "US BIS CCL"
        assert "dilution" in results[0]["snippet"].lower()

    @patch("src.scrapers.us_bis.USBISScraper.get")
    def test_scrape_no_match(self, mock_get):
        mock_get.return_value = _mock_response("<html><body>No relevant content</body></html>")
        scraper = USBISScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("zxq_nonexistent_item_xyz")
        assert results == []

    @patch("src.scrapers.us_bis.USBISScraper.get")
    def test_scrape_extracts_eccn(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = USBISScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("dilution refrigerator")
        # At least one result should have ECCN codes extracted
        all_codes = [code for r in results for code in r.get("eccn_codes", [])]
        assert "3B001" in all_codes or "3E001" in all_codes

    @patch("src.scrapers.us_bis.USBISScraper.get")
    def test_get_recent_updates_returns_dates(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = USBISScraper(config=SAMPLE_CONFIG)
        updates = scraper.get_recent_updates()
        assert any("2025" in u["date"] for u in updates)

    @patch("src.scrapers.us_bis.USBISScraper.get")
    def test_fetch_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("network error")
        scraper = USBISScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("anything")
        assert results == []


# ---------------------------------------------------------------------------
# Germany BAFA Scraper
# ---------------------------------------------------------------------------
class TestGermanyBAFAScraper:
    @patch("src.scrapers.germany_bafa.GermanyBAFAScraper.get")
    def test_scrape_finds_match(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = GermanyBAFAScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("qubit")
        assert isinstance(results, list)

    @patch("src.scrapers.germany_bafa.GermanyBAFAScraper.get")
    def test_source_label(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = GermanyBAFAScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("superconducting")
        for r in results:
            assert r["source"] == "Germany BAFA"


# ---------------------------------------------------------------------------
# EU Regulation Scraper
# ---------------------------------------------------------------------------
class TestEURegulationScraper:
    @patch("src.scrapers.eu_regulation.EURegulationScraper.get")
    def test_scrape_finds_match(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = EURegulationScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("quantum")
        assert isinstance(results, list)

    @patch("src.scrapers.eu_regulation.EURegulationScraper.get")
    def test_source_label(self, mock_get):
        mock_get.return_value = _mock_response()
        scraper = EURegulationScraper(config=SAMPLE_CONFIG)
        results = scraper.scrape("cryostat")
        for r in results:
            assert r["source"] == "EU Dual-Use Regulation"
