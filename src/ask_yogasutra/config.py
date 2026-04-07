"""
Shared configuration constants for all ask_yogasutra backends.

Import from here instead of defining these in each backend file so that
a model change only needs to be made in one place.
"""

import os

# Embedding model — lightweight 384-dim model, good balance of speed and quality.
# Must match the model used when indices were originally built; changing this
# requires clearing models/ and rebuilding all persisted indices.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq LLM — alternatives: "mistral-saba-24b", "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Base directory for persisted LlamaIndex indices (relative to the project root)
PERSIST_BASE_DIR = "models"
