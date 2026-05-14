# Ask Exporter - Export Control List Tracker

## Overview
An AI-powered system to track and query export control lists from multiple countries, helping researchers and procurement teams understand restrictions on quantum technology and dual-use items being exported to India.

## Problem Statement
Export control lists are constantly updated by various countries (US, Germany, EU) making it difficult to:
- Track which items are restricted for export to India
- Monitor recent additions to controlled lists
- Validate purchase orders against current restrictions
- Stay updated with changing regulations

## Solution
An agentic LLM system using LangGraph that can:
- Scrape and index export control lists from official sources
- Answer natural language queries about restrictions
- Track changes and updates to control lists
- Provide interactive chatbot interface via Streamlit

## Project Phases

### Phase I (PoC) - Current Scope
- Scrape official government export control websites (US, Germany, EU)
- Use public APIs where available
- Interactive query interface via Streamlit chatbot
- On-demand data retrieval (invoked when needed)
- Answer queries about:
  - Specific item restrictions to India
  - Recent additions to lists (last 3 months)
  - Purchase order validation

### Phase II (Future)
- Monitor PDF updates from regulatory bodies
- RSS/news feed tracking for real-time updates
- Automated report generation
- Historical trend analysis
- Email/notification alerts

## Architecture

### Tech Stack
- **Language**: Python 3.10+
- **LLM Framework**: LangChain + LangGraph
- **LLM Provider**: Groq (fast inference)
- **UI**: Streamlit
- **Web Scraping**: BeautifulSoup4, Scrapy, requests
- **Data Storage**: SQLite (local), JSON cache
- **Vector DB**: ChromaDB (for semantic search)

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (Chatbot)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Agent Orchestrator               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Query Understanding → Tool Selection → Response │  │
│  └──────────────────────────────────────────────────┘  │
└────────┬────────────────────────────────┬───────────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐          ┌──────────────────────────┐
│  Data Scrapers   │          │    Knowledge Base        │
│                  │          │                          │
│ • US Commerce    │          │ • Vector Store (ChromaDB)│
│ • Germany BAFA   │◄────────►│ • SQLite Database        │
│ • EU Regulation  │          │ • JSON Cache             │
└──────────────────┘          └──────────────────────────┘
```

### Agent Workflow (LangGraph)

```
User Query
    │
    ▼
[Query Classifier]
    │
    ├─→ [Item Check] → Scrape/Search Lists → Extract Info
    │
    ├─→ [Recent Updates] → Time-filtered Search → Summarize
    │
    └─→ [Purchase Order] → Parse Items → Batch Check → Report
    │
    ▼
[Response Generator]
    │
    ▼
User Response
```

## Data Sources

### United States
- **Source**: Bureau of Industry and Security (BIS) - Commerce Control List (CCL)
- **URL**: https://www.bis.doc.gov/index.php/regulations/commerce-control-list-ccl
- **Format**: Web pages, downloadable PDFs, potential API
- **Key Lists**: EAR99, ECCN categories

### Germany
- **Source**: Federal Office for Economic Affairs and Export Control (BAFA)
- **URL**: https://www.bafa.de/EN/Foreign_Trade/Export_Control/export_control_node.html
- **Format**: Web pages, PDF documents
- **Key Lists**: Dual-use goods, Ausfuhrliste

### European Union
- **Source**: EU Dual-Use Regulation
- **URL**: https://policy.trade.ec.europa.eu/development-and-sustainability/sustainable-trade/export-control-dual-use-items_en
- **Format**: Official regulations, annexes
- **Key Lists**: Regulation (EU) 2021/821

## Features

### Core Functionality
1. **Item Restriction Query**
   - Input: Item name/description, target country (India)
   - Output: Restriction status, ECCN/control code, restrictions details
   
2. **Recent Updates Query**
   - Input: Time period (e.g., last 3 months), category (e.g., quantum tech)
   - Output: List of newly added items with dates and details

3. **Purchase Order Validation**
   - Input: List of items or PO document
   - Output: Flagged items, restriction details, recommendations

### Query Examples
```
"Is a helium-3 dilution refrigerator export-controlled to India?"
"What quantum-related items were added to US CCL in last 3 months?"
"Check if these items are restricted: superconducting qubit chip, cryogenic probe station"
"Show me all recent Germany BAFA updates on quantum sensors"
```

## Project Structure
```
ask_exporter/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── config.yaml                  # Configuration settings
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py            # LangGraph agent definition
│   │   ├── nodes.py            # Agent node functions
│   │   └── tools.py            # LangChain tools
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base.py             # Base scraper class
│   │   ├── us_bis.py           # US BIS scraper
│   │   ├── germany_bafa.py     # Germany BAFA scraper
│   │   └── eu_regulation.py    # EU regulation scraper
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── vectorstore.py      # ChromaDB integration
│   │   ├── database.py         # SQLite operations
│   │   └── cache.py            # JSON caching
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py    # Streamlit chatbot UI
│   └── utils/
│       ├── __init__.py
│       ├── parsers.py          # Text/document parsers
│       └── helpers.py          # Utility functions
├── data/
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Processed/cleaned data
│   └── cache/                  # Cached queries
├── tests/
│   ├── __init__.py
│   ├── test_scrapers.py
│   ├── test_agents.py
│   └── test_queries.py
└── notebooks/
    └── exploration.ipynb       # Data exploration
```

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Git
- Groq API key (free tier available)

### Installation Steps
```bash
# Clone repository
git clone <your-repo-url>
cd ask_exporter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your API keys

# Run the application
streamlit run src/ui/streamlit_app.py
```

## Configuration

See `config.yaml` for:
- Scraper settings (rate limits, timeouts)
- LLM parameters (model, temperature)
- Database paths
- Country-specific source URLs

## Usage

### Command Line (Future)
```bash
python -m src.cli query "Is item X restricted?"
python -m src.cli update --country US
```

### Streamlit UI
```bash
streamlit run src/ui/streamlit_app.py
```

### API (Future)
```python
from src.agents import ExportControlAgent

agent = ExportControlAgent()
response = agent.query("Is a quantum computer export-controlled to India?")
print(response)
```

## Development Roadmap

### Phase I - PoC (Current)
- [ ] Setup project structure
- [ ] Implement US BIS scraper
- [ ] Implement Germany BAFA scraper
- [ ] Implement EU regulation scraper
- [ ] Build LangGraph agent
- [ ] Create ChromaDB vector store
- [ ] Develop Streamlit chatbot UI
- [ ] Test core query types
- [ ] Documentation

### Phase II - Enhancement
- [ ] PDF monitoring system
- [ ] RSS feed integration
- [ ] Automated report generation
- [ ] Email notifications
- [ ] Historical tracking & analytics
- [ ] Multi-language support
- [ ] REST API
- [ ] Enhanced caching strategy

## Known Limitations (Phase I)
- On-demand scraping (no continuous monitoring)
- Limited to three countries (US, Germany, EU)
- No automated alerts
- No PDF parsing (web sources only)
- Local deployment only

## Contributing
(Add contribution guidelines if this becomes collaborative)

## License
(Specify license)

## Contact
Yogesh - [your-email]

## Acknowledgments
- Quantum education hub team
- Export control regulatory bodies for public data access
