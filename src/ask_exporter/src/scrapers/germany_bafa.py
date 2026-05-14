"""Germany BAFA (Federal Office for Economic Affairs and Export Control) scraper."""

from ..utils.helpers import load_config, setup_logger
from ..utils.parsers import html_to_text, extract_control_dates
from .base import BaseScraper

logger = setup_logger(__name__)


class GermanyBAFAScraper(BaseScraper):
    """Scrapes BAFA export control pages."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = config or load_config()
        country_cfg = cfg.get("scrapers", {}).get("countries", {}).get("germany", {})
        self.base_url: str = country_cfg.get("base_url", "https://www.bafa.de")
        self.main_path: str = country_cfg.get("endpoints", {}).get(
            "main", "/EN/Foreign_Trade/Export_Control/export_control_node.html"
        )

    def _fetch_page_text(self, path: str) -> str:
        url = self.base_url + path
        try:
            resp = self.get(url)
            return html_to_text(resp.text)
        except Exception as exc:
            logger.warning("Failed to fetch BAFA page %s: %s", url, exc)
            return ""

    def scrape(self, query: str) -> list[dict]:
        """Search BAFA main page for query term and return matching context snippets."""
        text = self._fetch_page_text(self.main_path)
        if not text:
            return []

        query_lower = query.lower()
        results = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 5)
                snippet = "\n".join(lines[context_start:context_end])
                results.append({
                    "source": "Germany BAFA",
                    "url": self.base_url + self.main_path,
                    "snippet": snippet,
                    "query": query,
                })
        logger.info("Germany BAFA: %d result(s) for '%s'", len(results), query)
        return results

    def get_recent_updates(self, days: int = 90) -> list[dict]:
        text = self._fetch_page_text(self.main_path)
        dates = extract_control_dates(text)
        updates = []
        for date in dates:
            idx = text.find(date)
            snippet = text[max(0, idx - 100): idx + 200].strip()
            updates.append({
                "source": "Germany BAFA",
                "date": date,
                "snippet": snippet,
                "url": self.base_url + self.main_path,
            })
        return updates
