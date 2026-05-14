"""LangGraph node functions for the unified pipeline."""

import json
import re
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from ..extractors.bom import extract_bom, hardware_names_from_bom
from ..parsers.arxiv_fetcher import download_paper_pdf, fetch_paper_metadata
from ..parsers.pdf_parser import parse_pdf_bytes, parse_pdf_file
from ..parsers.section_extractor import extract_relevant_sections
from ..scrapers.eu_regulation import EURegulationScraper
from ..scrapers.germany_bafa import GermanyBAFAScraper
from ..scrapers.us_bis import USBISScraper
from ..utils.helpers import load_config, setup_logger
from ..utils.validators import extract_arxiv_id, parse_item_list
from .state import AgentState

logger = setup_logger(__name__)

_EXPORT_CHECK_PROMPT = """\
You are an export control expert. Determine whether the following item is export-controlled \
for export to India by the US, Germany, and EU.

Item: {item_name}

US BIS information:
{us_info}

Germany BAFA information:
{de_info}

EU Dual-Use Regulation information:
{eu_info}

Respond ONLY as a JSON object with this structure (no markdown):
{{
  "item_name": "{item_name}",
  "us_status": "controlled|not_controlled|unclear",
  "us_details": "brief explanation",
  "germany_status": "controlled|not_controlled|unclear",
  "germany_details": "brief explanation",
  "eu_status": "controlled|not_controlled|unclear",
  "eu_details": "brief explanation",
  "overall_risk": "high|medium|low|clear|unclear",
  "recommendation": "one-sentence action recommendation"
}}\
"""


def _get_llm(config: dict | None = None) -> ChatGroq:
    cfg = config or load_config()
    llm_cfg = cfg.get("llm", {})
    return ChatGroq(
        model=llm_cfg.get("model", "llama3-70b-8192"),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
    )


# ---------------------------------------------------------------------------
# Node: input_router  (no-op — routing is done via conditional edges in graph)
# ---------------------------------------------------------------------------
def input_router(state: AgentState) -> AgentState:
    logger.info("Input type: %s", state["input_type"])
    return state


# ---------------------------------------------------------------------------
# Node: paper_fetcher  (arxiv mode)
# ---------------------------------------------------------------------------
def paper_fetcher(state: AgentState) -> AgentState:
    arxiv_id = extract_arxiv_id(state["raw_input"])
    if not arxiv_id:
        return {**state, "error": f"Could not parse ArXiv ID from: {state['raw_input']}"}
    try:
        metadata = fetch_paper_metadata(arxiv_id)
        pdf_path = download_paper_pdf(arxiv_id)
        text = parse_pdf_file(pdf_path)
        return {**state, "paper_metadata": metadata, "paper_text": text}
    except Exception as exc:
        logger.error("paper_fetcher error: %s", exc)
        return {**state, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: pdf_parser  (pdf mode — bytes already in state)
# ---------------------------------------------------------------------------
def pdf_parser(state: AgentState) -> AgentState:
    pdf_bytes = state.get("pdf_bytes")
    if not pdf_bytes:
        return {**state, "error": "No PDF bytes in state"}
    try:
        text = parse_pdf_bytes(pdf_bytes)
        return {**state, "paper_text": text}
    except Exception as exc:
        logger.error("pdf_parser error: %s", exc)
        return {**state, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: section_extractor
# ---------------------------------------------------------------------------
def section_extractor(state: AgentState) -> AgentState:
    text = state.get("paper_text", "")
    sections = extract_relevant_sections(text)
    return {**state, "relevant_sections": sections}


# ---------------------------------------------------------------------------
# Node: bom_extractor
# ---------------------------------------------------------------------------
def bom_extractor(state: AgentState) -> AgentState:
    text = state.get("relevant_sections") or state.get("paper_text", "")
    if not text:
        return {**state, "bom": {"hardware": [], "software": [], "materials": []}}
    try:
        bom = extract_bom(text)
        return {**state, "bom": bom}
    except Exception as exc:
        logger.error("bom_extractor error: %s", exc)
        return {**state, "bom": {"hardware": [], "software": [], "materials": []}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: item_list_builder
# ---------------------------------------------------------------------------
def item_list_builder(state: AgentState) -> AgentState:
    if state["input_type"] == "direct_items":
        items = parse_item_list(state["raw_input"])
    else:
        bom = state.get("bom") or {}
        items = hardware_names_from_bom(bom)
    logger.info("Items to check: %s", items)
    return {**state, "items_to_check": items}


# ---------------------------------------------------------------------------
# Node: export_control_checker
# ---------------------------------------------------------------------------
def export_control_checker(state: AgentState) -> AgentState:
    items = state.get("items_to_check", [])
    if not items:
        return {**state, "export_control_results": []}

    us_scraper = USBISScraper()
    de_scraper = GermanyBAFAScraper()
    eu_scraper = EURegulationScraper()
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(_EXPORT_CHECK_PROMPT)
    chain = prompt | llm

    results = []
    for item in items:
        logger.info("Checking export control for: %s", item)
        us_info = _snippets(us_scraper.scrape(item))
        de_info = _snippets(de_scraper.scrape(item))
        eu_info = _snippets(eu_scraper.scrape(item))

        try:
            response = chain.invoke({
                "item_name": item,
                "us_info": us_info or "No data found.",
                "de_info": de_info or "No data found.",
                "eu_info": eu_info or "No data found.",
            })
            result = _parse_json_response(response.content, item)
        except Exception as exc:
            logger.error("Export check LLM error for '%s': %s", item, exc)
            result = _unclear_result(item)

        results.append(result)
        time.sleep(0.5)  # gentle rate-limiting between LLM calls

    return {**state, "export_control_results": results}


# ---------------------------------------------------------------------------
# Node: report_generator
# ---------------------------------------------------------------------------
def report_generator(state: AgentState) -> AgentState:
    report = {
        "input_type": state["input_type"],
        "paper_info": state.get("paper_metadata"),
        "bom": state.get("bom"),
        "items_checked": state.get("items_to_check", []),
        "export_control_results": state.get("export_control_results", []),
        "error": state.get("error"),
    }
    return {**state, "final_report": report}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _snippets(results: list[dict], max_results: int = 2) -> str:
    return "\n---\n".join(r.get("snippet", "") for r in results[:max_results])


def _parse_json_response(raw: str, item_name: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return _unclear_result(item_name)


def _unclear_result(item_name: str) -> dict:
    return {
        "item_name": item_name,
        "us_status": "unclear",
        "us_details": "Could not retrieve information.",
        "germany_status": "unclear",
        "germany_details": "Could not retrieve information.",
        "eu_status": "unclear",
        "eu_details": "Could not retrieve information.",
        "overall_risk": "unclear",
        "recommendation": "Manual verification required.",
    }
