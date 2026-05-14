"""LangGraph agent state definition."""

from typing import Literal, Optional, TypedDict


class AgentState(TypedDict):
    # Input
    input_type: Literal["direct_items", "arxiv", "pdf"]
    raw_input: str              # ArXiv ID/URL or free-text item list
    pdf_bytes: Optional[bytes]  # uploaded PDF bytes (pdf mode only)

    # Paper processing (arxiv / pdf modes)
    paper_metadata: Optional[dict]
    paper_text: Optional[str]
    relevant_sections: Optional[str]

    # BOM (arxiv / pdf modes)
    bom: Optional[dict]          # raw BOM dict {hardware: [...], software: [...], materials: [...]}

    # Items passed to export control checker
    items_to_check: list[str]

    # Export control results — one dict per item
    export_control_results: list[dict]

    # Final structured report
    final_report: Optional[dict]

    # Error message (non-fatal errors stored here, fatal ones raise)
    error: Optional[str]
