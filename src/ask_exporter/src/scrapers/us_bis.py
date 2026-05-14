"""US Bureau of Industry and Security (BIS) scraper."""

from ..utils.helpers import load_config, setup_logger
from ..utils.parsers import html_to_text, extract_eccn_codes, extract_control_dates
from .base import BaseScraper

logger = setup_logger(__name__)


class USBISScraper(BaseScraper):
    """Scrapes BIS Commerce Control List pages for export restriction information."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = config or load_config()
        country_cfg = cfg.get("scrapers", {}).get("countries", {}).get("us", {})
        self.base_url: str = country_cfg.get("base_url", "https://www.bis.doc.gov")
        self.ccl_path: str = country_cfg.get("endpoints", {}).get(
            "ccl", "/index.php/regulations/commerce-control-list-ccl"
        )
        self.entity_list_path: str = country_cfg.get("endpoints", {}).get(
            "entity_list",
            "/index.php/policy-guidance/lists-of-parties-of-concern/entity-list",
        )

    def _fetch_page_text(self, path: str) -> str:
        url = self.base_url + path
        try:
            resp = self.get(url)
            return html_to_text(resp.text)
        except Exception as exc:
            logger.warning("Failed to fetch BIS page %s: %s", url, exc)
            return ""

    def scrape(self, query: str) -> list[dict]:
        """Search CCL page text for query term and return matching context snippets."""
        text = self._fetch_page_text(self.ccl_path)
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
                    "source": "US BIS CCL",
                    "url": self.base_url + self.ccl_path,
                    "snippet": snippet,
                    "eccn_codes": extract_eccn_codes(snippet),
                    "query": query,
                })
        logger.info("US BIS: %d result(s) for '%s'", len(results), query)
        return results

    def get_recent_updates(self, days: int = 90) -> list[dict]:
        """Fetch CCL page and return lines that contain recent dates."""
        text = self._fetch_page_text(self.ccl_path)
        dates = extract_control_dates(text)
        updates = []
        for date in dates:
            # Collect surrounding context
            idx = text.find(date)
            snippet = text[max(0, idx - 100): idx + 200].strip()
            updates.append({
                "source": "US BIS",
                "date": date,
                "snippet": snippet,
                "url": self.base_url + self.ccl_path,
            })
        return updates
