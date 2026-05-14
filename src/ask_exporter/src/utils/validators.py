import re
from typing import Optional


_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_URL = re.compile(r"https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")


def extract_arxiv_id(raw: str) -> Optional[str]:
    """Return bare ArXiv ID from an ID string or URL, or None if unrecognised."""
    raw = raw.strip()
    if _ARXIV_ID.match(raw):
        return raw
    m = _ARXIV_URL.search(raw)
    return m.group(1) if m else None


def validate_pdf(file_bytes: bytes, max_mb: int = 50) -> tuple[bool, str]:
    """Return (ok, error_message). error_message is empty string on success."""
    if len(file_bytes) > max_mb * 1024 * 1024:
        return False, f"File exceeds {max_mb} MB limit"
    if not file_bytes.startswith(b"%PDF"):
        return False, "Not a valid PDF file"
    return True, ""


def parse_item_list(text: str) -> list[str]:
    """Parse a free-text item list (newline or comma separated) into clean strings."""
    text = text.strip()
    if not text:
        return []
    lines = text.splitlines()
    # If single line with commas, split on commas
    if len(lines) == 1 and "," in lines[0]:
        lines = lines[0].split(",")
    # Strip bullets, numbers, leading punctuation
    cleaned = []
    for line in lines:
        item = re.sub(r"^[\s\-\*•\d\.\)]+", "", line).strip()
        if item:
            cleaned.append(item)
    return cleaned
