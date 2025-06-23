import os

# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# # Paths for FAISS index and metadata storage
# DOCS_INDEX = os.path.join(VECTORSTORE_DIR, "index.faiss")
# FAISS_STORE_PKL = os.path.join(VECTORSTORE_DIR, "index.pkl")

# Create folders if they don't exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
