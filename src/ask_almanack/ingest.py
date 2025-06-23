import os
import pickle
from config import DATA_FOLDER, VECTORSTORE_DIR
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load all .txt files from the data folder and its subfolders
def load_documents(data_folder):
    docs = []
    for root, _, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".txt"):
                path = os.path.join(root, filename)
                try:
                    loader = TextLoader(path, encoding="utf-8")  # Specify UTF-8 encoding
                    loaded = loader.load()
                    docs.extend(loaded)
                except UnicodeDecodeError as e:
                    print(f"Error decoding file {path}: {e}")
                    print("Trying with latin-1 encoding...")
                    try:
                        loader = TextLoader(path, encoding="latin-1") # Try latin-1
                        loaded = loader.load()
                        docs.extend(loaded)
                    except UnicodeDecodeError as e2:
                        print(f"Error decoding file {path} with latin-1: {e2}")
                        print(f"Skipping file: {path}") # Skip the file if both encodings fail
                        continue
    return docs

# 2. Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# 3. Embed and create FAISS index using local model
def create_vectorstore(chunks):
    # Check if chunks is empty
    if not chunks:
        print("[ERROR] No chunks provided to create vectorstore!")
        return None
    
    # Extract texts and metadata separately from chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    print(f"[INFO] Processing {len(texts)} text chunks with metadata...")
    
    # Check if texts are empty
    if not texts or all(not text.strip() for text in texts):
        print("[ERROR] No valid text content found in chunks!")
        return None
    
    try:
        # Use HuggingFaceEmbeddings for Langchain compatibility
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("[INFO] Embeddings model loaded successfully.")

        # Create FAISS store using texts and metadatas separately
        faiss_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        print(f"[INFO] FAISS vectorstore created successfully with {len(texts)} vectors.")
        return faiss_store
        
    except Exception as e:
        print(f"[ERROR] Failed to create FAISS vectorstore: {e}")
        return None

# 4. Save index and metadata
def save_vectorstore(store):
    if store is None:
        print("[ERROR] Cannot save vectorstore - store is None!")
        return False
    
    try:
        # Ensure the vectorstore directory exists
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        print(f"[INFO] Saving vectorstore to {VECTORSTORE_DIR}")
        
        # Use FAISS's built-in save method
        store.save_local(VECTORSTORE_DIR)
        
        # Verify files were created
        index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
        pkl_file = os.path.join(VECTORSTORE_DIR, "index.pkl")
        
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            print(f"[SUCCESS] Vectorstore saved successfully.")
            print(f"[INFO] Created files:")
            print(f"  - {index_file}")
            print(f"  - {pkl_file}")
            return True
        else:
            print("[ERROR] Expected files were not created!")
            return False
        
    except Exception as e:
        print(f"[ERROR] Failed to save vectorstore: {e}")
        return False

# 5. Load vectorstore (helper function for later use)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    print("[INFO] Loading documents...")
    print(f"Data folder = {DATA_FOLDER}")
    print(f"VectorStore folder = {VECTORSTORE_DIR}")

    documents = load_documents(DATA_FOLDER)
    print(f"[INFO] Loaded {len(documents)} documents.")
    
    if not documents:
        print("[ERROR] No documents loaded! Check your data folder path and file contents.")
        exit(1)

    print("[INFO] Splitting documents...")
    chunks = split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    
    if not chunks:
        print("[ERROR] No chunks created! Check your document content.")
        exit(1)
    
    # Debug: Print first few chunks
    print(f"[DEBUG] Sample chunk content (first 100 chars): {chunks[0].page_content[:100]}...")
    print(f"[DEBUG] Sample chunk metadata: {chunks[0].metadata}")

    print("[INFO] Creating FAISS vectorstore with local embeddings...")
    store = create_vectorstore(chunks)
    
    if store is None:
        print("[ERROR] Failed to create vectorstore!")
        exit(1)

    print("[INFO] Saving vectorstore...")
    success = save_vectorstore(store)
    
    if success:
        print("[SUCCESS] Ingestion complete. Vectorstore saved.")
    else:
        print("[ERROR] Failed to save vectorstore!")
        exit(1)