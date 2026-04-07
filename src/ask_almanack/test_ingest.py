"""
Unit tests for ask_almanack ingestion pipeline.

Run with: python -m pytest test_ingest.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestLoadDocuments(unittest.TestCase):
    """Tests for the document loading step."""

    def setUp(self):
        # Create a temporary directory tree with .txt files
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, rel_path: str, content: str):
        full = Path(self.tmpdir) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return str(full)

    def test_loads_txt_files(self):
        from ingest import load_documents
        self._write("a.txt", "Naval on wealth.")
        self._write("sub/b.txt", "Naval on happiness.")
        docs = load_documents(self.tmpdir)
        self.assertEqual(len(docs), 2)

    def test_ignores_non_txt_files(self):
        from ingest import load_documents
        self._write("notes.md", "# Notes")
        self._write("data.csv", "col1,col2")
        self._write("valid.txt", "real content")
        docs = load_documents(self.tmpdir)
        self.assertEqual(len(docs), 1)

    def test_empty_directory(self):
        from ingest import load_documents
        docs = load_documents(self.tmpdir)
        self.assertEqual(docs, [])

    def test_nested_directories(self):
        from ingest import load_documents
        self._write("a/b/c/deep.txt", "deep content")
        docs = load_documents(self.tmpdir)
        self.assertEqual(len(docs), 1)
        self.assertIn("deep content", docs[0].page_content)


class TestSplitDocuments(unittest.TestCase):
    """Tests for the text splitting step."""

    def _make_doc(self, text: str):
        from langchain_community.document_loaders import TextLoader
        from langchain.schema import Document
        return Document(page_content=text, metadata={"source": "test"})

    def test_splits_long_text(self):
        from ingest import split_documents
        long_text = "word " * 600  # well over chunk_size=500
        docs = [self._make_doc(long_text)]
        chunks = split_documents(docs)
        self.assertGreater(len(chunks), 1)

    def test_short_text_stays_single_chunk(self):
        from ingest import split_documents
        short_text = "This is a short sentence."
        docs = [self._make_doc(short_text)]
        chunks = split_documents(docs)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, short_text)

    def test_empty_documents_list(self):
        from ingest import split_documents
        chunks = split_documents([])
        self.assertEqual(chunks, [])


class TestCreateVectorstore(unittest.TestCase):
    """Tests for vectorstore creation — mocks embeddings to avoid network calls."""

    def _make_chunk(self, text: str):
        from langchain.schema import Document
        return Document(page_content=text, metadata={"source": "test"})

    @patch("ingest.HuggingFaceEmbeddings")
    @patch("ingest.FAISS")
    def test_returns_none_on_empty_chunks(self, mock_faiss, mock_embeddings):
        from ingest import create_vectorstore
        result = create_vectorstore([])
        self.assertIsNone(result)
        mock_faiss.from_texts.assert_not_called()

    @patch("ingest.HuggingFaceEmbeddings")
    @patch("ingest.FAISS")
    def test_calls_faiss_from_texts(self, mock_faiss, mock_embeddings):
        from ingest import create_vectorstore
        chunks = [self._make_chunk("content one"), self._make_chunk("content two")]
        mock_faiss.from_texts.return_value = MagicMock()
        create_vectorstore(chunks)
        mock_faiss.from_texts.assert_called_once()
        call_kwargs = mock_faiss.from_texts.call_args
        texts_passed = call_kwargs[1].get("texts") or call_kwargs[0][0]
        self.assertEqual(len(texts_passed), 2)


class TestSaveVectorstore(unittest.TestCase):
    """Tests for the vectorstore save step."""

    def test_returns_false_on_none_store(self):
        from ingest import save_vectorstore
        result = save_vectorstore(None)
        self.assertFalse(result)

    def test_saves_to_directory(self):
        from ingest import save_vectorstore
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_store = MagicMock()
            # Simulate FAISS save_local creating the expected files
            def fake_save(path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "index.faiss").touch()
                (Path(path) / "index.pkl").touch()
            mock_store.save_local.side_effect = fake_save

            with patch("ingest.VECTORSTORE_DIR", tmpdir):
                result = save_vectorstore(mock_store)

            self.assertTrue(result)
            mock_store.save_local.assert_called_once_with(tmpdir)


if __name__ == "__main__":
    unittest.main()
