# ask_exporter — Unified Implementation Plan

## Confirmed Decisions
- Export control check: **hardware/equipment only** by default; software/materials opt-in
- Tests: **always mock** HTTP + LLM — no live network calls in test suite
- Dependencies: bumped to **langchain 0.3.x / langgraph 0.2.x** (consistent with rest of repo)

## What This System Does

Two-stage pipeline, three entry points:

| Mode | Input | Stages |
|------|-------|--------|
| Paper (ArXiv) | ArXiv ID, URL, or keyword | BOM extraction → Export control check |
| Paper (PDF upload) | Upload a PDF | BOM extraction → Export control check |
| Direct Items | Paste/type item list | Export control check only |

## Final File Structure

```
ask_exporter/
├── README.md                      (update for unified system)
├── requirements.txt               (merged + version-bumped)
├── .env.example                   (merged)
├── config.yaml                    (merged)
├── PLAN.md                        (this file)
├── src/
│   ├── __init__.py
│   ├── parsers/                   [NEW dir]
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py       fetch by ID/URL/keyword via arxiv lib
│   │   ├── pdf_parser.py          PyMuPDF full text extraction
│   │   └── section_extractor.py  find Methods/Setup/Materials sections
│   ├── extractors/                [NEW dir]
│   │   ├── __init__.py
│   │   └── bom.py                 single LLM call → structured BOM JSON
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base.py                BaseScraper: rate-limit, retry, robots.txt
│   │   ├── us_bis.py              US Bureau of Industry and Security
│   │   ├── germany_bafa.py        Germany BAFA
│   │   └── eu_regulation.py      EU Dual-Use Regulation
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── schemas.py             [NEW] Pydantic: HardwareItem, SoftwareItem,
│   │   │                          MaterialItem, BOM, ExportControlResult, UnifiedReport
│   │   ├── cache.py               JSON file cache with TTL
│   │   ├── database.py            SQLite: papers, boms, export_results tables
│   │   └── vectorstore.py        ChromaDB: export_control_docs + paper_chunks collections
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py               [NEW] TypedDict AgentState
│   │   ├── tools.py               LangChain tools wrapping all modules
│   │   ├── nodes.py               All LangGraph node functions
│   │   └── graph.py               StateGraph with conditional edges
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py       Unified 3-mode Streamlit UI
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py             load_config(), setup_logger()
│       ├── validators.py          ArXiv ID regex, PDF type/size, item list parse
│       └── parsers.py             html_to_text(), extract_eccn_codes()
├── data/
│   ├── papers/
│   ├── boms/
│   ├── export_control/
│   └── cache/
└── tests/
    ├── __init__.py
    ├── test_parsers.py            arxiv_fetcher, pdf_parser, section_extractor (mocked)
    ├── test_extractors.py         bom.py with mocked LLM + mocked paper text
    ├── test_scrapers.py           base + 3 country scrapers (mocked HTTP)
    └── test_pipeline.py           end-to-end: synthetic paper → BOM → export check (all mocked)
```

## LangGraph State

```python
class AgentState(TypedDict):
    input_type: Literal["direct_items", "arxiv", "pdf"]
    raw_input: str              # ArXiv ID/URL or item list text
    pdf_bytes: Optional[bytes]
    paper_metadata: Optional[dict]
    paper_text: Optional[str]
    relevant_sections: Optional[str]
    bom: Optional[dict]         # serialized BOM
    items_to_check: List[str]   # hardware names from BOM, or direct input
    export_control_results: List[dict]
    final_report: Optional[dict]
    error: Optional[str]
```

## LangGraph Flow

```
START → [input_router]
  ├─ "arxiv"        → [paper_fetcher] → [pdf_parser] → [section_extractor] → [bom_extractor]
  ├─ "pdf"          ─────────────────→ [pdf_parser] → [section_extractor] → [bom_extractor]
  └─ "direct_items" ──────────────────────────────────────────────────────→ (skip to below)
                                                                              ↓
                                                                    [item_list_builder]
                                                                              ↓
                                                                  [export_control_checker]
                                                                  (scrape + RAG + LLM per item)
                                                                              ↓
                                                                    [report_generator]
                                                                              ↓
                                                                             END
```

## Data Models (schemas.py)

```python
HardwareItem: name, subcategory, specifications, part_number, manufacturer, quantity, cost
SoftwareItem: name, version, purpose, license, url
MaterialItem: name, subcategory, specification, quantity, supplier, cost
BOM: hardware: List[HardwareItem], software: List[SoftwareItem], materials: List[MaterialItem]
     + .hardware_names() → List[str]  (used for export check by default)

ExportControlResult: item_name, us_status, us_details, germany_status, germany_details,
                     eu_status, eu_details, overall_risk (high/medium/low/clear/unclear),
                     recommendation

UnifiedReport: input_type, paper_info, bom, export_control_results, timestamp, processing_time
```

## Streamlit UI Layout

```
Sidebar: Mode radio [ArXiv Paper | Upload PDF | Direct Items]

Main:
  Input panel (mode-dependent)
  [Analyze Button]
  ---
  (paper modes) Paper metadata
  (paper modes) BOM table — expandable by category (hardware / software / materials)
  Export Control Results table:
    Item | US | Germany | EU | Risk | Recommendation
    (color: red=high, yellow=medium, green=clear)
  [Download Full JSON Report]
```

## Implementation Order (code file by file)

### Phase 1 — Foundation
1. `requirements.txt` (merged + bumped)
2. `config.yaml` (merged)
3. `.env.example` (merged)
4. `src/utils/helpers.py`
5. `src/utils/validators.py`
6. `src/utils/parsers.py`

### Phase 2 — Paper Pipeline
7. `src/parsers/__init__.py`
8. `src/parsers/arxiv_fetcher.py`
9. `src/parsers/pdf_parser.py`
10. `src/parsers/section_extractor.py`
11. `src/extractors/__init__.py`
12. `src/extractors/bom.py`

### Phase 3 — Export Control Pipeline
13. `src/scrapers/base.py`
14. `src/scrapers/us_bis.py`
15. `src/scrapers/germany_bafa.py`
16. `src/scrapers/eu_regulation.py`

### Phase 4 — Knowledge Layer
17. `src/knowledge/schemas.py`
18. `src/knowledge/cache.py`
19. `src/knowledge/database.py`
20. `src/knowledge/vectorstore.py`

### Phase 5 — Agent
21. `src/agents/state.py`
22. `src/agents/tools.py`
23. `src/agents/nodes.py`
24. `src/agents/graph.py`

### Phase 6 — UI
25. `src/ui/streamlit_app.py`

### Phase 7 — Tests
26. `tests/test_scrapers.py`
27. `tests/test_parsers.py`
28. `tests/test_extractors.py`
29. `tests/test_pipeline.py`

### Phase 8 — Cleanup
30. Delete `src/ask_papers/` entirely
31. Update `CLAUDE.md` to remove ask_papers section

## Status
- [x] Phase 1 — Foundation
- [x] Phase 2 — Paper Pipeline
- [x] Phase 3 — Export Control Pipeline
- [x] Phase 4 — Knowledge Layer
- [x] Phase 5 — Agent
- [x] Phase 6 — UI
- [x] Phase 7 — Tests
- [x] Phase 8 — Cleanup — ask_papers deleted, CLAUDE.md updated
