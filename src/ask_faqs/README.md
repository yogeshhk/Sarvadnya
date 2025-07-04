# FAQs Bot Generator

A custom FAQs Bot Generator, driven by `config.json`

```
{
  "APP_NAME": "MyApp",
  "DOCS_INDEX":"/fullpath/to/docs.index",
  "FAISS_STORE_PKL":"/fullpath/to/faiss_store.pkl",
  "FILES_PATHS": [
    "/fullpath/to/file1.csv",
    "/fullpath/to/file2.txt",
    "/fullpath/to/file3.pdf"
  ]
}
```

**Files:**
`streamlit_main.py` - Main chatbot UI and logic
`bot_config_wizard.py` - GUI to create/update bot_config.json
`bot_config.json` - Stores bot settings and paths
`models/docs.index` - FAISS vector index (auto-generated)
`models/faiss_store.pkl` - Metadata for vector index (auto-generated)
`data/*.html, *.csv, etc.` - Your source knowledge documents

## Flow (in short)

Load config.json → Load/Create FAISS Index from docs → Embed with MiniLM → Load Groq LLM → Build RetrievalQA → Streamlit UI → Ask questions → Get answers

- Run `bot_config_wizard.py` → Groq model, data files, and model storage path.
- Run streamlit_main.py:
  - Bot ingests files, builds index
  - Starts chatbot using **Groq** model

[Demo Video](./demo_video.mp4)

<!-- https://github.com/user-attachments/assets/6b7664a7-22f7-4836-a2d1-c1cea08dd0cd -->
