"""Simple JSON file cache with TTL."""

import hashlib
import json
import time
from pathlib import Path

from ..utils.helpers import load_config, setup_logger, ensure_dirs

logger = setup_logger(__name__)


class JSONCache:
    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        cache_cfg = cfg.get("cache", {})
        self.enabled: bool = cache_cfg.get("enabled", True)
        self.directory: str = cache_cfg.get("directory", "data/cache")
        self.expiry_seconds: float = cache_cfg.get("expiry_hours", 24) * 3600
        if self.enabled:
            ensure_dirs(self.directory)

    def _key_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return Path(self.directory) / f"{hashed}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - data.get("_cached_at", 0)
            if age > self.expiry_seconds:
                path.unlink(missing_ok=True)
                return None
            return data.get("payload")
        except Exception as exc:
            logger.warning("Cache read error for key '%s': %s", key, exc)
            return None

    def set(self, key: str, value: dict | list) -> None:
        if not self.enabled:
            return
        path = self._key_path(key)
        try:
            path.write_text(
                json.dumps({"_cached_at": time.time(), "payload": value}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Cache write error for key '%s': %s", key, exc)

    def invalidate(self, key: str) -> None:
        self._key_path(key).unlink(missing_ok=True)

    def clear_all(self) -> int:
        count = 0
        for f in Path(self.directory).glob("*.json"):
            f.unlink()
            count += 1
        return count
