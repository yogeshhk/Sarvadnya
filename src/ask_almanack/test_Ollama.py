
import os
import pickle
from config import DATA_FOLDER, VECTORSTORE_DIR, DOCS_INDEX, FAISS_STORE_PKL

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss

# 1. Load all .txt files from the data folder
def load_documents(data_folder):
    docs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            path = os.path.join(data_folder, filename)
            loader = TextLoader(path)
            loaded = loader.load()
            docs.extend(loaded)
    return docs

# 2. Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# 3. Embed and create FAISS index using LangChain's embedding wrapper
def create_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding)

# 4. Save index and metadata
def save_vectorstore(store):
    faiss.write_index(store.index, DOCS_INDEX)
    with open(FAISS_STORE_PKL, "wb") as f:
        pickle.dump(store, f)

if __name__ == "__main__":
    print("[INFO] Loading documents...")
    documents = load_documents(DATA_FOLDER)
    print(f"[INFO] Loaded {len(documents)} documents.")

    print("[INFO] Splitting documents...")
    chunks = split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    print("[INFO] Creating FAISS vectorstore with local embeddings...")
    store = create_vectorstore(chunks)

    print("[INFO] Saving vectorstore...")
    save_vectorstore(store)
    print("[SUCCESS] Ingestion complete. Vectorstore saved.")
