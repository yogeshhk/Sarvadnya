"""LangGraph StateGraph — unified paper→BOM→export-control pipeline."""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    bom_extractor,
    export_control_checker,
    input_router,
    item_list_builder,
    paper_fetcher,
    pdf_parser,
    report_generator,
    section_extractor,
)
from .state import AgentState


def _route_input(state: AgentState) -> str:
    """Conditional edge: decide first node after input_router."""
    input_type = state.get("input_type", "direct_items")
    if input_type == "arxiv":
        return "paper_fetcher"
    if input_type == "pdf":
        return "pdf_parser"
    return "item_list_builder"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("input_router", input_router)
    g.add_node("paper_fetcher", paper_fetcher)
    g.add_node("pdf_parser", pdf_parser)
    g.add_node("section_extractor", section_extractor)
    g.add_node("bom_extractor", bom_extractor)
    g.add_node("item_list_builder", item_list_builder)
    g.add_node("export_control_checker", export_control_checker)
    g.add_node("report_generator", report_generator)

    # Entry
    g.add_edge(START, "input_router")

    # Conditional routing based on input_type
    g.add_conditional_edges(
        "input_router",
        _route_input,
        {
            "paper_fetcher": "paper_fetcher",
            "pdf_parser": "pdf_parser",
            "item_list_builder": "item_list_builder",
        },
    )

    # Paper pipeline (arxiv path)
    g.add_edge("paper_fetcher", "section_extractor")

    # Paper pipeline (pdf path)
    g.add_edge("pdf_parser", "section_extractor")

    # Shared paper pipeline continuation
    g.add_edge("section_extractor", "bom_extractor")
    g.add_edge("bom_extractor", "item_list_builder")

    # Export control pipeline (all paths converge here)
    g.add_edge("item_list_builder", "export_control_checker")
    g.add_edge("export_control_checker", "report_generator")
    g.add_edge("report_generator", END)

    return g


def compile_graph():
    """Return a compiled, runnable LangGraph."""
    return build_graph().compile()
