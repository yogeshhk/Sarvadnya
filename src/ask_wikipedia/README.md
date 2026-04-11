# Wikipedia Bot

QnA bot on Wikipedia using LangChain, HuggingFace embeddings, ChromaDB, and Groq LLaMA3.

## Features

- Loads `GROQ_API_KEY` from a `.env` file.
- Uses **LangChain’s WikipediaLoader** to fetch articles for configurable topics.
- Chunks articles with `RecursiveCharacterTextSplitter` (800 tokens, 400 overlap).
- Embeds chunks with **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2`.
- Persists embeddings in **ChromaDB** for fast semantic search across runs.
- Retrieves top candidates using **MMR** (Maximal Marginal Relevance) to reduce redundancy.
- Answers questions with **Groq’s LLaMA3-70b** via `langchain-groq`.
- Prints up to 2 unique Wikipedia source links per answer.

## Installation

```bash
pip install -r requirements.txt
```

`.env`

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Run

```bash
python main_driver.py
```

## Reference
https://python.langchain.com/docs/integrations/document_loaders/wikipedia/
