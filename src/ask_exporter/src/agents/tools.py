"""LangChain tool definitions wrapping scrapers and knowledge base."""

from langchain_core.tools import tool

from ..knowledge.cache import JSONCache
from ..knowledge.vectorstore import VectorStore
from ..scrapers.eu_regulation import EURegulationScraper
from ..scrapers.germany_bafa import GermanyBAFAScraper
from ..scrapers.us_bis import USBISScraper

_cache = JSONCache()
_us = USBISScraper()
_de = GermanyBAFAScraper()
_eu = EURegulationScraper()
_vs = VectorStore()


@tool
def scrape_us_bis(query: str) -> str:
    """Search US BIS Commerce Control List for an item or keyword."""
    cached = _cache.get(f"us_bis:{query}")
    if cached:
        return str(cached)
    results = _us.scrape(query)
    _cache.set(f"us_bis:{query}", results)
    if not results:
        return "No US BIS results found."
    return "\n---\n".join(r["snippet"] for r in results[:3])


@tool
def scrape_germany_bafa(query: str) -> str:
    """Search Germany BAFA export control list for an item or keyword."""
    cached = _cache.get(f"de_bafa:{query}")
    if cached:
        return str(cached)
    results = _de.scrape(query)
    _cache.set(f"de_bafa:{query}", results)
    if not results:
        return "No Germany BAFA results found."
    return "\n---\n".join(r["snippet"] for r in results[:3])


@tool
def scrape_eu_regulation(query: str) -> str:
    """Search EU Dual-Use Regulation for an item or keyword."""
    cached = _cache.get(f"eu_reg:{query}")
    if cached:
        return str(cached)
    results = _eu.scrape(query)
    _cache.set(f"eu_reg:{query}", results)
    if not results:
        return "No EU regulation results found."
    return "\n---\n".join(r["snippet"] for r in results[:3])


@tool
def vector_search_export_docs(query: str) -> str:
    """Semantic search over previously scraped export control documents."""
    docs = _vs.search_export_docs(query)
    if not docs:
        return "No relevant export control documents found in knowledge base."
    return "\n---\n".join(d.page_content for d in docs[:3])


ALL_TOOLS = [scrape_us_bis, scrape_germany_bafa, scrape_eu_regulation, vector_search_export_docs]
