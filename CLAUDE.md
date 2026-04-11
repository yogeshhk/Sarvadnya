# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Sarvadnya (सर्वज्ञ)** is a collection of independent PoC (Proof-of-Concept) chatbot projects demonstrating RAG (Retrieval-Augmented Generation) and fine-tuning approaches for domain adaptation of LLMs. Each subdirectory under `src/` is a self-contained project with its own `requirements.txt`.

## Project Structure

```
src/
├── ask_almanack/       # Naval Ravikant's Almanack chatbot (FAISS + Groq/LLaMA3)
├── ask_bharat/         # RAG on historical Indian content (Streamlit + Chainlit, Groq, reranking, citations)
├── ask_biomedical/     # BioGPT / Falcon on medical text
├── ask_career_transition/  # Career transition guidance chatbot
├── ask_dataframe/      # Natural language queries on tabular/CSV data
├── ask_faqs/           # FAQ bot generator
├── ask_faqs_chatbot/   # FAQ chatbot with multiple vectorizer backends
├── ask_finance/        # Agentic RAG for financial documents (LangGraph + ChromaDB)
├── ask_floorplans/     # AI-powered floor plan retrieval and generation
├── ask_gandharva/      # Indian music chatbot (chainlit UI)
├── ask_graph/          # GraphRAG experiments
├── ask_gst/            # GST FAQs chatbot (Streamlit + Langchain + VertexAI)
├── ask_kautilya/       # Arthashastra chatbot (LlamaIndex + LangChain)
├── ask_manahprarupe/   # Marathi-language RAG/fine-tuning (BLOOM, mT5, MuRIL, Gemma3)
├── ask_manim/          # Text-to-Manim animation generator
├── ask_medicine/       # Medical PDF RAG
├── ask_paulgraham/     # Paul Graham essays chatbot
├── ask_suntzu/         # Art of War chatbot
├── ask_text2star/      # Text to star schema
├── ask_wikipedia/      # Wikipedia QnA bot (Groq/LLaMA3 + HuggingFace embeddings + ChromaDB + MMR)
├── ask_yhk/            # Personal AMA testbed for different LLMs
├── ask_yogasutra/      # Patanjali Yogasutra with GraphRAG + LinearRAG + benchmarks
models/                 # Local LLM model files (lmstudio-community, Qwen, etc.)
```

## Running Projects

Each project is independent. The general pattern is:

```bash
# Navigate to the project directory
cd src/<project_name>

# Install dependencies
pip install -r requirements.txt

# Step 1: Run ingestion to build the vectorstore
python ingest.py

# Step 2: Run the Streamlit UI
streamlit run streamlit_main.py --server.fileWatcherType none

# Or run chainlit UI (for projects using chainlit)
chainlit run chainlit_main.py

# Or run command-line version
python cmd_main.py
```

## Key Patterns

### RAG Pipeline (most projects follow this)
1. **Ingest**: Load text/PDF files → chunk → embed (HuggingFace `all-MiniLM-L6-v2`) → store in FAISS vectorstore
2. **Query**: Load FAISS vectorstore → retrieve candidates → (optional) cross-encoder rerank → pass to LLM → return answer

Advanced projects (ask_bharat) add a **reranking** step between retrieval and generation using a `CrossEncoder` from `sentence-transformers`.

### LLM Providers Used
- **Groq API** (`GROQ_API_KEY`): LLaMA3-70b, Mistral — used in ask_almanack, ask_bharat, ask_wikipedia, ask_yogasutra
- **OpenAI API** (`OPENAI_API_KEY`): Used in ask_finance, ask_floorplans
- **HuggingFace** (local/free): BioGPT, Falcon, sentence-transformers
- **Google VertexAI / PaLM**: Used in ask_gst

### Environment Variables
Set API keys as environment variables before running:
```bash
export GROQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

Projects typically read keys via `os.getenv()`.

## ask_bharat — Most Feature-Rich RAG UI

Has both a Streamlit UI (full-featured) and a Chainlit UI (Ollama/local):
```bash
cd src/ask_bharat

# Build FAISS vectorstore from PDFs
python ingest.py

# Streamlit UI — Groq + reranking + citations (recommended)
streamlit run streamlit_main.py --server.fileWatcherType none

# Chainlit UI — Ollama/llama2 (local LLM)
chainlit run chainlit_main.py
```

Key features of the Streamlit UI:
- **Conversational memory**: full multi-turn history sent to the LLM
- **Cross-encoder reranking**: FAISS retrieves 10 candidates → `ms-marco-MiniLM-L-6-v2` reranks → top 3 used
- **Hyperlinked citations**: LLM emits `[1]`, `[2]` markers; rendered as anchor links to reference cards
- **PDF download buttons**: each reference card has a download button for the source PDF

## ask_yogasutra — Most Feature-Complete Project

Has two RAG backends and a benchmark suite:
```bash
cd src/ask_yogasutra

# GraphRAG chatbot
streamlit run graphrag/streamlit_main_graphrag.py

# LinearRAG chatbot
streamlit run linearrag/streamlit_main_linearrag.py

# Benchmark testing
python tests/run_benchmark.py --list-configs
python tests/run_benchmark.py --config baseline_fast
python tests/run_benchmark.py --all

# Index management
python utils/manage_indices.py list
python utils/manage_indices.py clear all
```

Persisted indices are stored in `models/linearrag/` and `models/graphrag/` and auto-detected on subsequent runs (15–40x speedup).

## ask_manahprarupe — Fine-Tuning Project

Handles Marathi-language fine-tuning with multiple model options:
```bash
cd src/ask_manahprarupe

# RAG with Marathi models
streamlit run streamlit_muril.py     # MuRIL embeddings
streamlit run streamlit_i3cube.py    # i3Cube model
streamlit run streamlit_app.py       # Gemma3

# Fine-tuning scripts (require GPU/substantial compute)
python fine_tune_gemma3.py
python fine_tune_muril.py
python fine_tune_mt5.py
```

## Tech Stack Summary

| Layer | Technologies |
|-------|-------------|
| UI | Streamlit, Chainlit |
| LLM Orchestration | LangChain, LlamaIndex |
| Embeddings | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) |
| Reranking | sentence-transformers `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) |
| Vector Store | FAISS (local), ChromaDB, Pinecone |
| Graph Store | NetworkX, Neo4j (optional) |
| Fine-tuning | PEFT/LoRA, Unsloth, HuggingFace Transformers |
| Data formats | TXT, PDF, JSON |
