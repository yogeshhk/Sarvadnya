"""Unified Streamlit UI — three input modes feeding the LangGraph pipeline."""

import json
import sys
import time
from pathlib import Path

# When Streamlit runs this file directly it has no parent package, so relative
# imports fail. Adding the project root (ask_exporter/) to sys.path lets Python
# find the `src` package and all relative imports inside it still resolve normally.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.agents.graph import compile_graph
from src.utils.helpers import load_config
from src.utils.validators import extract_arxiv_id, parse_item_list, validate_pdf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask Exporter",
    page_icon="🛂",
    layout="wide",
)

config = load_config()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {"report": None, "running": False, "error": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar — mode selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛂 Ask Exporter")
    st.caption("Export Control Checker + Research BOM Generator")
    st.divider()
    mode = st.radio(
        "Input mode",
        options=["ArXiv Paper", "Upload PDF", "Direct Items"],
        index=0,
    )
    st.divider()
    st.markdown("**Checks:** US BIS CCL · Germany BAFA · EU Dual-Use")
    st.markdown("**Target country:** India")


# ---------------------------------------------------------------------------
# Main — input panel
# ---------------------------------------------------------------------------
st.title("Ask Exporter")
st.subheader("Export Control Checker + Research BOM Generator")

if mode == "ArXiv Paper":
    col1, col2 = st.columns([3, 1])
    with col1:
        arxiv_input = st.text_input(
            "ArXiv ID or URL",
            placeholder="e.g. 2301.12345 or https://arxiv.org/abs/2301.12345",
        )
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("Analyze", type="primary", use_container_width=True)
    input_type = "arxiv"
    raw_input = arxiv_input.strip()
    pdf_bytes = None

elif mode == "Upload PDF":
    uploaded = st.file_uploader("Upload a research paper PDF", type=["pdf"])
    run_btn = st.button("Analyze", type="primary")
    input_type = "pdf"
    raw_input = uploaded.name if uploaded else ""
    pdf_bytes = uploaded.read() if uploaded else None

else:  # Direct Items
    st.markdown("Enter items to check, one per line (or comma-separated):")
    items_text = st.text_area(
        "Items",
        height=150,
        placeholder="Dilution refrigerator\nSuperconducting qubit chip\nCryogenic probe station",
    )
    run_btn = st.button("Check Items", type="primary")
    input_type = "direct_items"
    raw_input = items_text.strip()
    pdf_bytes = None


# ---------------------------------------------------------------------------
# Validation + run
# ---------------------------------------------------------------------------
def _validate_inputs() -> str | None:
    """Return an error string, or None if inputs are valid."""
    if input_type == "arxiv":
        if not raw_input:
            return "Please enter an ArXiv ID or URL."
        if not extract_arxiv_id(raw_input):
            return "Could not parse a valid ArXiv ID. Expected format: 2301.12345"
    elif input_type == "pdf":
        if pdf_bytes is None:
            return "Please upload a PDF file."
        ok, err = validate_pdf(pdf_bytes)
        if not ok:
            return err
    else:
        if not raw_input:
            return "Please enter at least one item."
        if not parse_item_list(raw_input):
            return "No recognisable items found in the input."
    return None


if run_btn:
    err = _validate_inputs()
    if err:
        st.error(err)
    else:
        st.session_state["report"] = None
        st.session_state["error"] = None
        st.session_state["running"] = True

        with st.spinner("Running pipeline... (this may take 30-90 seconds)"):
            try:
                graph = compile_graph()
                initial_state = {
                    "input_type": input_type,
                    "raw_input": raw_input,
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
                t0 = time.time()
                result = graph.invoke(initial_state)
                elapsed = time.time() - t0
                report = result.get("final_report") or {}
                report["processing_time_seconds"] = round(elapsed, 1)
                st.session_state["report"] = report
                if result.get("error"):
                    st.session_state["error"] = result["error"]
            except Exception as exc:
                st.session_state["error"] = str(exc)
            finally:
                st.session_state["running"] = False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
report = st.session_state.get("report")
error_msg = st.session_state.get("error")

if error_msg:
    st.warning(f"Warning: {error_msg}")

if report:
    st.divider()
    proc_time = report.get("processing_time_seconds", 0)
    st.caption(f"Completed in {proc_time}s")

    # --- Paper metadata (arxiv / pdf modes) ---
    paper_info = report.get("paper_info")
    if paper_info:
        with st.expander("Paper Information", expanded=True):
            st.markdown(f"**{paper_info.get('title', 'N/A')}**")
            authors = paper_info.get("authors", [])
            st.markdown(f"*{', '.join(authors[:5])}{'...' if len(authors) > 5 else ''}*")
            st.markdown(f"ArXiv: [{paper_info.get('arxiv_id', '')}]({paper_info.get('url', '')})")
            st.markdown(paper_info.get("abstract", ""))

    # --- BOM table (arxiv / pdf modes) ---
    bom = report.get("bom")
    if bom and any(bom.get(k) for k in ("hardware", "software", "materials")):
        with st.expander("Bill of Materials", expanded=True):
            hw = bom.get("hardware", [])
            sw = bom.get("software", [])
            mat = bom.get("materials", [])

            if hw:
                st.markdown("**Hardware / Equipment**")
                import pandas as pd
                hw_df = pd.DataFrame([
                    {
                        "Name": item.get("name", ""),
                        "Subcategory": item.get("subcategory", ""),
                        "Specifications": str(item.get("specifications", {})),
                        "Manufacturer": item.get("manufacturer") or "",
                        "Part No.": item.get("part_number") or "",
                        "Est. Cost": item.get("estimated_cost") or "",
                    }
                    for item in hw
                ])
                st.dataframe(hw_df, use_container_width=True)

            if sw:
                st.markdown("**Software / Frameworks**")
                sw_df = pd.DataFrame([
                    {"Name": i.get("name", ""), "Version": i.get("version") or "",
                     "Purpose": i.get("purpose", "")}
                    for i in sw
                ])
                st.dataframe(sw_df, use_container_width=True)

            if mat:
                st.markdown("**Materials / Consumables**")
                mat_df = pd.DataFrame([
                    {"Name": i.get("name", ""), "Subcategory": i.get("subcategory", ""),
                     "Specification": i.get("specification") or "",
                     "Supplier": i.get("supplier") or ""}
                    for i in mat
                ])
                st.dataframe(mat_df, use_container_width=True)

    # --- Export control results ---
    ec_results = report.get("export_control_results", [])
    if ec_results:
        st.markdown("### Export Control Results")
        st.caption("Checking against: US BIS CCL · Germany BAFA · EU Dual-Use Regulation (target: India)")

        _RISK_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡", "clear": "🟢", "unclear": "⚪"}
        _STATUS_EMOJI = {"controlled": "🔴", "not_controlled": "🟢", "unclear": "⚪"}

        for res in ec_results:
            risk = res.get("overall_risk", "unclear")
            emoji = _RISK_EMOJI.get(risk, "⚪")
            with st.expander(f"{emoji} {res.get('item_name', 'Unknown')} — Risk: {risk.upper()}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    us_s = res.get("us_status", "unclear")
                    st.metric("US BIS", f"{_STATUS_EMOJI.get(us_s, '⚪')} {us_s.replace('_', ' ').title()}")
                    st.caption(res.get("us_details", ""))
                with c2:
                    de_s = res.get("germany_status", "unclear")
                    st.metric("Germany BAFA", f"{_STATUS_EMOJI.get(de_s, '⚪')} {de_s.replace('_', ' ').title()}")
                    st.caption(res.get("germany_details", ""))
                with c3:
                    eu_s = res.get("eu_status", "unclear")
                    st.metric("EU Dual-Use", f"{_STATUS_EMOJI.get(eu_s, '⚪')} {eu_s.replace('_', ' ').title()}")
                    st.caption(res.get("eu_details", ""))
                st.info(f"**Recommendation:** {res.get('recommendation', '')}")

    # --- JSON download ---
    st.divider()
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="export_control_report.json",
        mime="application/json",
    )
