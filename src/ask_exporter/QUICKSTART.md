# Quick Start Guide - Ask Exporter

## Initial Setup (5 minutes)

### 1. Clone and Setup
```bash
cd ask_exporter
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API Key
```powershell
# Windows (PowerShell)
$env:GROQ_API_KEY = "your_key_here"
```
```bash
# Linux / macOS
export GROQ_API_KEY="your_key_here"
```

### 3. Run headless (no UI)
```bash
# Uses defaults from config.yaml [headless] section
python cmd_main.py

# Override input type via env vars
$env:HEADLESS_INPUT_TYPE = "arxiv"
$env:HEADLESS_RAW_INPUT  = "2301.07715"
python cmd_main.py
```

### 4. Launch UI
```bash
# From src/ask_exporter/, either command works:
streamlit run streamlit_main.py --server.fileWatcherType none
# or
streamlit run src/ui/streamlit_app.py --server.fileWatcherType none
```

## First Query

Try these queries in the chatbot:
1. "Is a helium-3 dilution refrigerator export-controlled to India?"
2. "What quantum-related items were added to US CCL in last 3 months?"
3. "Check if superconducting qubit chips are restricted"

## Development Workflow

1. **Scrapers First**: Implement US BIS scraper
2. **Vector Store**: Index scraped data in ChromaDB
3. **Agent**: Build LangGraph query agent
4. **UI**: Connect Streamlit to agent

## Troubleshooting

**Import errors**: Ensure venv is activated
**API errors**: Check that GROQ_API_KEY is set in your shell environment
**Scraping errors**: Check network connectivity to government sites

## Next Steps

- Read ARCHITECTURE.md (to be created)
- Check out notebooks/exploration.ipynb (to be created)
- Run tests: `pytest tests/`
