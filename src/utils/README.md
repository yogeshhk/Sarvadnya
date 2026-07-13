# utils

Shared helpers for `ask_*` projects that follow the repo's standard RAG pattern
(load documents → split into chunks → embed → store in FAISS).

## rag_loader.py

Two functions:

- `build_faiss_vectorstore(data_path, vectorstore_path, glob="*.txt", ...)` —
  loads `.txt` or `.pdf` files from `data_path`, chunks them, embeds them with
  `sentence-transformers/all-MiniLM-L6-v2`, and saves a FAISS index to
  `vectorstore_path`.
- `load_faiss_vectorstore(vectorstore_path, ...)` — loads a previously built
  FAISS index for querying.

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from rag_loader import build_faiss_vectorstore, load_faiss_vectorstore

build_faiss_vectorstore(data_path="data/", vectorstore_path="vectorstore/db_faiss", glob="*.txt")
store = load_faiss_vectorstore("vectorstore/db_faiss")
docs = store.similarity_search("What is wealth?", k=3)
```

No project currently imports this module — it was added to consolidate the
ingest boilerplate duplicated across projects. New RAG projects (or ones being
refactored) should call it instead of re-implementing the same load/chunk/embed
steps.
