# Ask Kautilya

Ask questions to Artha Shastra by Kautilya

**LangChain**

**Files:**

- ask_kautilya/
- data/ # Arthashastra documents (TXT, PDF, etc.)
- models/ # FAISS index, embedding model files, etc.
  - faiss_index
  - index.pkl
- bot_config.json # Configuration for embedding model and paths
- langchain_streamlit_main.py # Main Streamlit app

**Flow:**
User Query → MiniLM Embedding → FAISS Similarity Search → Top Chunks →
→ Model → Final Answer

**Models:**  
"gemma2-9b-it", "llama3-8b-8192", "llama3-70b-8192"

**Run:** `python -m streamlit run langchain_streamlit_main.py`

---

**LLamaIndex**

- fast Q&A powered by sentence-transformers/all-MiniLM-L6-v2
- Local embedding & vector search using LlamaIndex

**Files:**

- ask_kautilya/

  - data/ # Text files (Arthashastra)
  - model/ # Persistent vector index files
  - `llamaindex_cmd_main.py` # Main chatbot script

- Embeddings- sentence-transformers/all-MiniLM-L6-v2
- Terminal UI rich

Dependencies:
`pip install llama-index rich langchain-community sentence-transformers`

Run:
`python llamaindex_cmd_main.py`
