"""ChromaDB vector store — two collections: export_control_docs and paper_chunks."""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ..utils.helpers import load_config, setup_logger, ensure_dirs

logger = setup_logger(__name__)


class VectorStore:
    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        vs_cfg = cfg.get("vectorstore", {})
        self.persist_dir: str = vs_cfg.get("path", "data/chroma_db")
        self.export_collection: str = vs_cfg.get("export_collection", "export_control_docs")
        self.paper_collection: str = vs_cfg.get("paper_collection", "paper_chunks")
        self.top_k: int = vs_cfg.get("top_k", 5)
        embed_model: str = cfg.get("embeddings", {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        ensure_dirs(self.persist_dir)
        self._embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self._export_store: Chroma | None = None
        self._paper_store: Chroma | None = None

    def _get_export_store(self) -> Chroma:
        if self._export_store is None:
            self._export_store = Chroma(
                collection_name=self.export_collection,
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )
        return self._export_store

    def _get_paper_store(self) -> Chroma:
        if self._paper_store is None:
            self._paper_store = Chroma(
                collection_name=self.paper_collection,
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )
        return self._paper_store

    # --- Export control docs ---
    def add_export_docs(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        self._get_export_store().add_documents(docs)
        logger.info("Added %d doc(s) to export_control_docs", len(docs))

    def search_export_docs(self, query: str, k: int | None = None) -> list[Document]:
        return self._get_export_store().similarity_search(query, k=k or self.top_k)

    # --- Paper chunks ---
    def add_paper_chunks(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        self._get_paper_store().add_documents(docs)
        logger.info("Added %d chunk(s) to paper_chunks", len(docs))

    def search_paper_chunks(self, query: str, k: int | None = None) -> list[Document]:
        return self._get_paper_store().similarity_search(query, k=k or self.top_k)
