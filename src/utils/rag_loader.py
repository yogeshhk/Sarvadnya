"""
Shared RAG ingestion utilities used across multiple ask_* projects.

The standard pipeline in this repository is:
  load documents → split into chunks → embed → store in FAISS

Every project that follows this pattern can call build_faiss_vectorstore()
instead of re-implementing the same ~20 lines.

Usage example (from any ask_* project):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from rag_loader import build_faiss_vectorstore, load_faiss_vectorstore

    # Build once
    build_faiss_vectorstore(
        data_path="data/",              # directory of .txt/.pdf files
        vectorstore_path="vectorstore/db_faiss",
        glob="*.txt",                   # or "*.pdf"
    )

    # Load for querying
    store = load_faiss_vectorstore("vectorstore/db_faiss")
    docs = store.similarity_search("What is wealth?", k=3)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Default embedding model — shared by all projects in this repo.
# Changing this constant requires rebuilding all existing vectorstores.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return an embeddings object, always on CPU (no GPU dependency)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def build_faiss_vectorstore(
    data_path: str,
    vectorstore_path: str,
    glob: str = "*.txt",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Load documents from *data_path*, embed them, and save a FAISS index.

    Args:
        data_path: Directory containing source documents.
        vectorstore_path: Directory where the FAISS index will be saved.
        glob: File pattern to load (e.g. "*.txt", "*.pdf"). Defaults to "*.txt".
        chunk_size: Token/character size of each chunk. Defaults to 500.
        chunk_overlap: Overlap between consecutive chunks. Defaults to 50.
        embedding_model: HuggingFace model name for embeddings.

    Returns:
        The created FAISS vectorstore object.

    Raises:
        ValueError: If no documents are found at *data_path*.
    """
    loader_cls = PyPDFLoader if glob.endswith(".pdf") else TextLoader

    loader = DirectoryLoader(
        data_path,
        glob=glob,
        loader_cls=loader_cls,
        loader_kwargs={"encoding": "utf-8"} if loader_cls is TextLoader else {},
    )
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No documents found in '{data_path}' matching glob '{glob}'. "
            "Check the path and file extension."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)

    Path(vectorstore_path).mkdir(parents=True, exist_ok=True)
    store.save_local(vectorstore_path)

    print(f"Vectorstore saved to '{vectorstore_path}' ({len(chunks)} chunks from {len(documents)} documents).")
    return store


def load_faiss_vectorstore(
    vectorstore_path: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Load a previously built FAISS index from disk.

    The index is generated locally by build_faiss_vectorstore(), so
    allow_dangerous_deserialization=True is safe here — the pickle file
    is never received from an untrusted source.

    Args:
        vectorstore_path: Directory containing index.faiss + index.pkl.
        embedding_model: Must match the model used when the index was built.

    Returns:
        The loaded FAISS vectorstore.

    Raises:
        FileNotFoundError: If the vectorstore directory or index files are missing.
    """
    index_file = Path(vectorstore_path) / "index.faiss"
    pkl_file = Path(vectorstore_path) / "index.pkl"

    if not index_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at '{vectorstore_path}'. "
            "Run build_faiss_vectorstore() first."
        )

    embeddings = _get_embeddings(embedding_model)
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: locally generated index
    )
