"""SQLite persistence via SQLAlchemy — papers, BOMs, and export control results."""

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..utils.helpers import load_config, setup_logger, ensure_dirs

logger = setup_logger(__name__)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"
    arxiv_id = Column(String, primary_key=True)
    title = Column(Text)
    authors = Column(Text)     # JSON list
    abstract = Column(Text)
    url = Column(String)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class BOMRecord(Base):
    __tablename__ = "boms"
    id = Column(String, primary_key=True)   # arxiv_id or filename hash
    source_type = Column(String)            # arxiv / pdf / direct
    bom_json = Column(Text)                 # full BOM JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class ExportControlRecord(Base):
    __tablename__ = "export_results"
    id = Column(String, primary_key=True)   # item_name hash
    item_name = Column(String)
    result_json = Column(Text)              # ExportControlResult JSON
    checked_at = Column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        db_path = cfg.get("database", {}).get("path", "data/export_control.db")
        ensure_dirs(str(db_path).rsplit("/", 1)[0] if "/" in db_path else "data")
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)

    def _session(self) -> Session:
        return self._Session()

    # --- Papers ---
    def upsert_paper(self, metadata: dict) -> None:
        with self._session() as s:
            paper = Paper(
                arxiv_id=metadata["arxiv_id"],
                title=metadata.get("title", ""),
                authors=json.dumps(metadata.get("authors", [])),
                abstract=metadata.get("abstract", ""),
                url=metadata.get("url", ""),
            )
            s.merge(paper)
            s.commit()

    def get_paper(self, arxiv_id: str) -> dict | None:
        with self._session() as s:
            paper = s.get(Paper, arxiv_id)
            if paper is None:
                return None
            return {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": json.loads(paper.authors or "[]"),
                "abstract": paper.abstract,
                "url": paper.url,
            }

    # --- BOMs ---
    def save_bom(self, record_id: str, source_type: str, bom: dict) -> None:
        with self._session() as s:
            rec = BOMRecord(
                id=record_id,
                source_type=source_type,
                bom_json=json.dumps(bom, ensure_ascii=False),
            )
            s.merge(rec)
            s.commit()

    def get_bom(self, record_id: str) -> dict | None:
        with self._session() as s:
            rec = s.get(BOMRecord, record_id)
            return json.loads(rec.bom_json) if rec else None

    # --- Export Control Results ---
    def save_export_result(self, item_name: str, result: dict) -> None:
        import hashlib
        key = hashlib.sha256(item_name.encode()).hexdigest()[:16]
        with self._session() as s:
            rec = ExportControlRecord(
                id=key,
                item_name=item_name,
                result_json=json.dumps(result, ensure_ascii=False),
            )
            s.merge(rec)
            s.commit()

    def get_export_result(self, item_name: str) -> dict | None:
        import hashlib
        key = hashlib.sha256(item_name.encode()).hexdigest()[:16]
        with self._session() as s:
            rec = s.get(ExportControlRecord, key)
            return json.loads(rec.result_json) if rec else None
