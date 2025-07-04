### AskGraph: Intelligent Chatbot over Excel Data

- Dense Embeddings (MiniLM-L6-v2)
- Graph Search (NetworkX)
- Vector DB (ChromaDB)
- Groq's LLaMA 3 for natural language understanding
- Streamlit UI for interactive usage

## Technology Stack

| Component        | Library / Tool                   |
| ---------------- | -------------------------------- |
| Vector Embedding | Sentence Transformers (`MiniLM`) |
| Vector DB        | `ChromaDB`                       |
| Graph Structure  | `NetworkX`                       |
| UI               | `Streamlit`                      |
| LLM              | `Groq` (LLaMA 3)                 |
| Backend Model    | `GraphRAG` Class                 |

## File Structure

├── app.py # Streamlit UI
├── graphrag.py # Core logic: Graph building, RAG retrieval, querying
├── requirements.txt # Python dependencies
├── .env # Contains your GROQ_API_KEY
├── chroma_db/ # Auto-generated ChromaDB store
└── sample_data.xlsx # Test data (auto-created)

### How It Works

Data Upload & Indexing
Excel rows are joined into one sentence
Each row is encoded to a dense vector via MiniLM (all-MiniLM-L6-v2)
Graph is built with semantic similarity links (> 0.3 cosine similarity)
Vectors are stored in ChromaDB for fast retrieval

### Run

```bash
pip install -r requirements.txt

python graphrag_main.py

streamlit run app.py
```

**Video:**
https://github.com/user-attachments/assets/40d668f3-35af-4127-9a7c-018186ef5640

---

![WhatsApp Image 2025-06-29 at 01 26 33_f364f8d5](https://github.com/user-attachments/assets/8ae95e51-cc32-464d-abb8-d61c3bc5b10f)

### UI Features:

- File upload for Excel data
- Real-time chat interface
- Query history with detailed results
- Sample questions
- Graph statistics display

## Testing

The `graphrag.py` file includes comprehensive tests in the `__main__` section that:

- Creates sample Excel data
- Tests all major functions
- Validates graph creation and retrieval
- Runs sample queries

## Troubleshooting

1. **API Key Issues**: Ensure your Groq API key is valid and has sufficient credits
2. **File Upload**: Make sure Excel files have proper column headers
3. **Dependencies**: Run `pip install -r requirements.txt` to install all required packages
4. **ChromaDB**: The database is created automatically in `./chroma_db/` directory
