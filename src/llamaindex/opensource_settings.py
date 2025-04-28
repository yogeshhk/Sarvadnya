# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.__version__)          # e.g., 2.3.0+cu121

# --- Import necessary LlamaIndex components ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.llms.openai import OpenAI # For the LM Studio LLM endpoint
from llama_index.llms.lmstudio import LMStudio
# >>> Using HuggingFace embeddings locally is generally recommended with LM Studio <<<
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import nest_asyncio

nest_asyncio.apply()

# --- Configuration for LM Studio ---
# !! IMPORTANT: Verify these values with your LM Studio setup !!

# 1. API Base URL (Check LM Studio Server tab)
LM_STUDIO_API_BASE = "http://localhost:1234/v1" # Default for LM Studio

# 2. Model Identifier (CRITICAL: Get this from LM Studio after loading the model)
#    This is NOT just "llama3". Check the LM Studio server logs or UI for the exact ID.
#    Replace the placeholder below with the actual identifier.
LM_STUDIO_MODEL_NAME = "gemma-3-1b-it" # <<< REPLACE THIS PLACEHOLDER

# 3. API Key (LM Studio usually doesn't require one)
LM_STUDIO_API_KEY = "lm-studio" # Placeholder, often ignored by LM Studio

print(f"Configuring for LM Studio:")
print(f" - API Base: {LM_STUDIO_API_BASE}")
print(f" - Model Name: {LM_STUDIO_MODEL_NAME}")
print(f" - Using local HuggingFace embeddings.")

# --- Create the LLM instance pointing to LM Studio ---
llm = LMStudio(
    model_name=LM_STUDIO_MODEL_NAME, # Use the specific model identifier from LM Studio
    base_url=LM_STUDIO_API_BASE,
    # api_key=LM_STUDIO_API_KEY,
    # is_chat_model=True, # Llama 3 instruct models are chat models
    # LM Studio server might have timeout issues with complex tasks, increase if needed
    # timeout=600, # Example: 10 minutes, default is often 120 seconds
    temperature=0.7, # Adjust creativity/determinism
)

# --- Create a local Embedding model using HuggingFace ---
# LM Studio's focus is LLM serving, relying on its endpoint for embeddings might not work or be ideal.
# Using a separate local model is more robust.
# Requires: pip install llama-index-embeddings-huggingface sentence-transformers torch
print("Initializing local HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", # A good default, small and fast
    # device="cuda" # Uncomment this if you have a GPU and PyTorch with CUDA installed
    # device="cpu" # Explicitly use CPU if needed
)
print("Embedding model initialized.")

# --- Apply the configuration globally using Settings ---
Settings.llm = llm
Settings.embed_model = embed_model
print("LlamaIndex Settings configured.")

# --- Load documents (no change here) ---
print("Loading documents from 'pdf/' directory...")
try:
    documents = SimpleDirectoryReader("pdf/").load_data()
    print(f"Loaded {len(documents)} documents.")
    if not documents:
        print("Warning: No documents found in 'pdf/' directory.")
        # Decide if you want to exit or continue without documents
        # exit()
except Exception as e:
    print(f"Error loading documents: {e}")
    exit()


# --- Create index using the configured settings (no change here) ---
# This step will use the HuggingFace model to create embeddings locally
# and store them.
print("Creating vector store index...")
try:
    index = VectorStoreIndex.from_documents(documents)
    print("Index created successfully.")
except Exception as e:
    print(f"Error creating index: {e}")
    # This could be due to issues with embedding model, memory, etc.
    exit()


# --- Create query engine (no change here) ---
# The query engine will use the HuggingFace embeddings for retrieval
# and the LM Studio LLM for synthesizing the final answer.
print("Creating query engine...")
query_engine = index.as_query_engine()
print("Query engine created.")

# --- Query the engine (no change here) ---
query = "What are the design goals and give details about it please."
print(f"\nQuerying index: '{query}'")
try:
    response = query_engine.query(query)
    # --- Print response (no change here) ---
    print("\nResponse from LLM:")
    print(response)
except Exception as e:
    print(f"Error during query: {e}")
    # This could be a timeout from LM Studio, OOM error, etc.

print("\n--- Script Finished ---")