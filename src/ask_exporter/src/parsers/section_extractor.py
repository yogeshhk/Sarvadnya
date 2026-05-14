"""Identify and extract Methods / Materials / Setup sections from paper text."""

import re

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)

# Patterns that mark the start of a relevant section
_START_PATTERNS = [
    re.compile(r"(?i)^\s*(?:\d+[\.\)]\s+)?(?:experimental\s+)?(?:methods?|procedures?|techniques?|setup|apparatus)\s*$"),
    re.compile(r"(?i)^\s*(?:\d+[\.\)]\s+)?experimental\s+(?:setup|details?|methods?)\s*$"),
    re.compile(r"(?i)^\s*(?:\d+[\.\)]\s+)?(?:materials?\s+and\s+methods?)\s*$"),
    re.compile(r"(?i)^\s*(?:\d+[\.\)]\s+)?(?:instruments?|equipment|components?|devices?)\s*$"),
    re.compile(r"(?i)^\s*(?:\d+[\.\)]\s+)?sample\s+(?:preparation|fabrication)\s*$"),
]

# Patterns that signal we should stop capturing
_STOP_PATTERNS = [
    re.compile(r"(?i)^\s*(?:\d+[\.\)]\s+)?(?:results?|discussion|conclusions?|references?|bibliography|acknowledgem)\s*$"),
]


def extract_relevant_sections(text: str, max_chars: int = 8000) -> str:
    """
    Return concatenated Methods/Materials/Setup sections.
    Falls back to the first max_chars of the text if no sections are detected.
    """
    lines = text.split("\n")
    blocks: list[list[str]] = []
    current: list[str] = []
    capturing = False

    for line in lines:
        stripped = line.strip()
        if any(pat.match(stripped) for pat in _START_PATTERNS):
            if current:
                blocks.append(current)
            current = [line]
            capturing = True
        elif capturing and any(pat.match(stripped) for pat in _STOP_PATTERNS):
            blocks.append(current)
            current = []
            capturing = False
        elif capturing:
            current.append(line)

    if current:
        blocks.append(current)

    if not blocks:
        logger.warning("No Methods/Materials sections found; using first %d chars", max_chars)
        return text[:max_chars]

    combined = "\n\n".join("\n".join(block) for block in blocks)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined
