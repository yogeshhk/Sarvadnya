import re
from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_eccn_codes(text: str) -> list[str]:
    """Find ECCN-style control codes like 3E001, 2B350."""
    pattern = re.compile(r"\b[0-9][A-E]\d{3}(?:\.[a-z])?\b")
    return sorted(set(pattern.findall(text)))


def extract_control_dates(text: str) -> list[str]:
    """Find dates in common regulatory formats."""
    pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
    )
    return sorted(set(pattern.findall(text)))


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"
