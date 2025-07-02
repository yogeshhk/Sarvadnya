import os
import pickle
import faiss

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from config import DATA_FOLDER, DOCS_INDEX, FAISS_STORE_PKL
from llama_index.core import Settings
from embedding import MiniLMEmbedding  

# ---- Load documents ----
def load_documents(data_folder):
    return SimpleDirectoryReader(data_folder).load_data()

# ---- Configure LlamaIndex globally ----
def configure_llamaindex():
    Settings.embed_model = MiniLMEmbedding()
    Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)
    Settings.llm = None  # Disable OpenAI

# ---- Build FAISS-based index ----
def create_vectorstore_index(documents, service_context):
    dim = 384
    raw_faiss_index = faiss.IndexFlatL2(dim)
    faiss_store = FaissVectorStore(raw_faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=faiss_store)

    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context
    )
    return index, raw_faiss_index

# ---- Save FAISS index and store ----
def save_faiss_index(faiss_index, index):
    faiss.write_index(faiss_index, DOCS_INDEX)
    with open(FAISS_STORE_PKL, "wb") as f:
        pickle.dump(index, f)

# ---- Main ----
if __name__ == "__main__":
    print("[INFO] Loading documents...")
    docs = load_documents(DATA_FOLDER)
    print(f"[INFO] Loaded {len(docs)} documents.")

    print("[INFO] Configuring llama-index settings...")
    configure_llamaindex()

    print("[INFO] Building FAISS index...")
    index, faiss_index = create_vectorstore_index(docs, service_context=None)

    print("[INFO] Saving FAISS index...")
    save_faiss_index(faiss_index, index)

    print("[SUCCESS] Ingestion complete using LlamaIndex with FAISS.")
