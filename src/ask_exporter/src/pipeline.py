"""
Headless pipeline runner — no UI, no argparse.

Public API:
    run_pipeline(input_type, raw_input, pdf_bytes, config) -> dict
    run_from_config(config)                                -> dict

All defaults come from the `headless` section of config.yaml.
"""

import json
import time
from pathlib import Path
from typing import Optional

from .agents.graph import compile_graph
from .utils.helpers import load_config, setup_logger

logger = setup_logger(__name__)


def run_pipeline(
    input_type: str,
    raw_input: str,
    pdf_bytes: Optional[bytes] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Run the full LangGraph pipeline and return the final report dict.

    Parameters
    ----------
    input_type : "direct_items" | "arxiv" | "pdf"
    raw_input  : item list text, ArXiv ID/URL, or filename hint (pdf mode)
    pdf_bytes  : raw PDF bytes (pdf mode only)
    config     : override config dict; loads from config.yaml if None
    """
    cfg = config or load_config()
    graph = compile_graph()

    initial_state = {
        "input_type": input_type,
        "raw_input": raw_input.strip(),
        "pdf_bytes": pdf_bytes,
        "paper_metadata": None,
        "paper_text": None,
        "relevant_sections": None,
        "bom": None,
        "items_to_check": [],
        "export_control_results": [],
        "final_report": None,
        "error": None,
    }

    logger.info("Pipeline start — input_type=%s", input_type)
    t0 = time.time()
    result = graph.invoke(initial_state)
    elapsed = round(time.time() - t0, 1)

    report = result.get("final_report") or {}
    report["processing_time_seconds"] = elapsed
    if result.get("error"):
        report["error"] = result["error"]

    logger.info("Pipeline done in %.1fs — %d item(s) checked", elapsed,
                len(report.get("export_control_results", [])))
    return report


def run_from_config(config: Optional[dict] = None) -> dict:
    """
    Run the pipeline using defaults from the `headless` section of config.yaml.
    Entry point for cmd_main.py and the test suite's default smoke-test path.
    """
    cfg = config or load_config()
    h = cfg.get("headless", {})
    input_type = h.get("input_type", "direct_items")

    if input_type == "arxiv":
        raw_input = h.get("arxiv_id", "")
    else:
        raw_input = h.get("raw_input", "")

    return run_pipeline(input_type=input_type, raw_input=raw_input, config=cfg)


def format_report(report: dict, fmt: str = "json") -> str:
    """Render a report dict as a string for stdout or file output."""
    if fmt == "pretty":
        lines = [f"=== Ask Exporter Report ===",
                 f"Input type : {report.get('input_type', 'N/A')}",
                 f"Time       : {report.get('processing_time_seconds', 0)}s",
                 ""]
        paper = report.get("paper_info")
        if paper:
            lines += [f"Paper      : {paper.get('title', '')}",
                      f"ArXiv ID   : {paper.get('arxiv_id', '')}",
                      ""]
        results = report.get("export_control_results", [])
        lines.append(f"Items checked: {len(results)}")
        for r in results:
            risk = r.get("overall_risk", "unclear").upper()
            lines.append(f"  [{risk}] {r.get('item_name', '')}")
            lines.append(f"    US      : {r.get('us_status', '')} — {r.get('us_details', '')}")
            lines.append(f"    Germany : {r.get('germany_status', '')} — {r.get('germany_details', '')}")
            lines.append(f"    EU      : {r.get('eu_status', '')} — {r.get('eu_details', '')}")
            lines.append(f"    >> {r.get('recommendation', '')}")
            lines.append("")
        if report.get("error"):
            lines.append(f"WARNING: {report['error']}")
        return "\n".join(lines)

    return json.dumps(report, indent=2, ensure_ascii=False)
