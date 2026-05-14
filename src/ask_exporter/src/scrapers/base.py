"""Base scraper with rate-limiting, retry, and robots.txt compliance."""

import time
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.helpers import load_config, setup_logger

logger = setup_logger(__name__)


class BaseScraper(ABC):
    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        scraper_cfg = cfg.get("scrapers", {})
        self.rate_limit: float = scraper_cfg.get("rate_limit", 2)  # req/s
        self.timeout: int = scraper_cfg.get("timeout", 30)
        self.retry_attempts: int = scraper_cfg.get("retry_attempts", 3)
        self.retry_delay: int = scraper_cfg.get("retry_delay", 5)
        self.user_agent: str = scraper_cfg.get("user_agent", "ExportControlBot/1.0")
        self.respect_robots: bool = scraper_cfg.get("respect_robots_txt", True)
        self._last_request: float = 0.0
        self._robots_cache: dict[str, RobotFileParser] = {}

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        gap = 1.0 / self.rate_limit
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_request = time.time()

    def _robots_allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, allow by default
                rp = None
            self._robots_cache[base] = rp
        rp = self._robots_cache[base]
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    def get(self, url: str) -> requests.Response:
        """Fetch a URL respecting rate-limit and robots.txt."""
        if not self._robots_allowed(url):
            raise PermissionError(f"robots.txt disallows fetching: {url}")
        self._wait_rate_limit()

        @retry(stop=stop_after_attempt(self.retry_attempts), wait=wait_fixed(self.retry_delay))
        def _fetch():
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            resp.raise_for_status()
            return resp

        logger.debug("GET %s", url)
        return _fetch()

    @abstractmethod
    def scrape(self, query: str) -> list[dict]:
        """Scrape source for items related to *query*. Return list of result dicts."""

    @abstractmethod
    def get_recent_updates(self, days: int = 90) -> list[dict]:
        """Return items added or modified in the last *days* days."""
