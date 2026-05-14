"""ArXiv paper fetching via the arxiv library."""

import time
from pathlib import Path
from typing import Optional

import arxiv

from ..utils.helpers import setup_logger, ensure_dirs
from ..utils.validators import extract_arxiv_id

logger = setup_logger(__name__)


def fetch_paper_metadata(arxiv_id: str) -> dict:
    """Return paper metadata dict for a given ArXiv ID."""
    client = arxiv.Client(num_retries=3, delay_seconds=3)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"Paper not found on ArXiv: {arxiv_id}")
    paper = results[0]
    return {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "published": paper.published.isoformat() if paper.published else None,
        "categories": paper.categories,
    }


def download_paper_pdf(arxiv_id: str, save_dir: str = "data/papers") -> Path:
    """Download the PDF for an ArXiv paper, using cached file if available."""
    ensure_dirs(save_dir)
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = Path(save_dir) / f"{safe_id}.pdf"
    if pdf_path.exists():
        logger.info("Using cached PDF: %s", pdf_path)
        return pdf_path

    client = arxiv.Client(num_retries=3, delay_seconds=3)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"Paper not found: {arxiv_id}")
    paper = results[0]
    paper.download_pdf(dirpath=str(save_dir), filename=pdf_path.name)
    logger.info("Downloaded PDF: %s", pdf_path)
    return pdf_path


def search_papers(query: str, max_results: int = 5, category: str = "quant-ph") -> list[dict]:
    """Search ArXiv and return list of paper metadata dicts."""
    client = arxiv.Client(num_retries=3, delay_seconds=3)
    search = arxiv.Search(
        query=f"cat:{category} AND {query}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append({
            "arxiv_id": paper.get_short_id(),
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],
            "abstract": paper.summary[:300] + ("..." if len(paper.summary) > 300 else ""),
            "url": paper.entry_id,
            "published": paper.published.isoformat() if paper.published else None,
        })
        time.sleep(0.4)
    return results


def resolve_input_to_id(raw: str) -> Optional[str]:
    """Try to extract an ArXiv ID from free text; return None if not recognised."""
    return extract_arxiv_id(raw)
