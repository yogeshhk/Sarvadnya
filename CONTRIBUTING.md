# Contribution Guidelines

Thank you for contributing to Sarvadnya! This is a collection of independent RAG/fine-tuning PoC chatbots. Each subdirectory under `src/` is self-contained.

## Adding a New PoC Project

1. Create a new `src/ask_<topic>/` directory.
2. Include a `README.md` describing what the project does and how to run it.
3. Include a `requirements.txt` with pinned or minimum-version dependencies.
   - Do **not** list stdlib modules (`os`, `json`, `sqlite3`, `asyncio`, etc.) in `requirements.txt`.
4. Follow the standard pipeline pattern where applicable:
   - `ingest.py` — loads data, embeds, and saves vectorstore
   - `streamlit_main.py` (or `chainlit_main.py`) — UI
   - `config.py` — constants (model names, paths, API key reads)

## Code Style

- Python 3.9+.
- Format with **black** (`pip install black && black .`).
- Use `logging` (not `print`) for diagnostic output in library code; `print` is fine in scripts/demos.
- Catch specific exception types (`except ValueError`, `except ImportError`) rather than bare `except:`.
- Read API keys from environment variables (`os.getenv`); never hardcode placeholders or real keys.

## Commit Messages

Use the imperative mood and keep the subject line under 72 characters:

```
Add ask_finance agentic RAG with LangGraph
Fix FAISS deserialization comment in ask_almanack
Remove deprecated Rasa engine from ask_dataframe
```

## Pull Requests

- One logical change per PR.
- Run the project you modified manually end-to-end before opening the PR.
- If the maintainer requests changes, push to the same branch — do not open a new PR.

Thank you for your contributions!
