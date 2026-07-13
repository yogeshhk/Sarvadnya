"""
Unit tests for ask_manahprarupe's RAGChatbot (rag_gemma.py).

RAGChatbot.__init__() eagerly calls out to Groq, HuggingFace embeddings, and
ChromaDB, so tests construct the object via __new__() (skipping __init__),
set only the attributes each method under test needs, and mock the external
clients. This exercises the real method bodies (including the
get_collection/create_collection fallback and the file-scanning /
error-handling logic) without hitting any network or model weights.

Run with: python -m pytest test_rag_gemma.py -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_gemma import RAGChatbot


def _bare_bot():
    """A RAGChatbot instance with __init__ skipped."""
    return RAGChatbot.__new__(RAGChatbot)


class TestSetupVectorStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    @patch("rag_gemma.StorageContext")
    @patch("rag_gemma.ChromaVectorStore")
    @patch("rag_gemma.chromadb.PersistentClient")
    def test_creates_collection_when_it_does_not_exist(self, mock_client_cls, mock_vs, mock_sc):
        bot = _bare_bot()
        bot.db_path = self.tmpdir
        bot.collection_name = "mental_models_marathi"

        mock_client = mock_client_cls.return_value
        mock_client.get_collection.side_effect = Exception("collection not found")

        bot._setup_vector_store()

        mock_client.get_collection.assert_called_once_with("mental_models_marathi")
        mock_client.create_collection.assert_called_once_with("mental_models_marathi")

    @patch("rag_gemma.StorageContext")
    @patch("rag_gemma.ChromaVectorStore")
    @patch("rag_gemma.chromadb.PersistentClient")
    def test_reuses_existing_collection(self, mock_client_cls, mock_vs, mock_sc):
        bot = _bare_bot()
        bot.db_path = self.tmpdir
        bot.collection_name = "mental_models_marathi"

        mock_client = mock_client_cls.return_value
        mock_client.get_collection.return_value = MagicMock()

        bot._setup_vector_store()

        mock_client.get_collection.assert_called_once_with("mental_models_marathi")
        mock_client.create_collection.assert_not_called()


class TestLoadDocuments(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write(self, rel_path: str, content: str):
        full = Path(self.tmpdir) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    def test_loads_supported_extensions_only(self):
        self._write("a.txt", "Marathi text content.")
        self._write("b.md", "# Heading content")
        self._write("c.tex", "\\section{content}")
        self._write("d.pdf", "not actually parsed as pdf")
        self._write("e.py", "print('code, not content')")

        bot = _bare_bot()
        bot.data_directory = self.tmpdir
        bot._load_documents()

        loaded_names = {doc.metadata["filename"] for doc in bot.documents}
        self.assertEqual(loaded_names, {"a.txt", "b.md", "c.tex"})

    def test_skips_empty_files(self):
        self._write("empty.txt", "   ")
        self._write("real.txt", "content")

        bot = _bare_bot()
        bot.data_directory = self.tmpdir
        bot._load_documents()

        self.assertEqual(len(bot.documents), 1)
        self.assertEqual(bot.documents[0].metadata["filename"], "real.txt")

    def test_raises_when_directory_missing(self):
        bot = _bare_bot()
        bot.data_directory = str(Path(self.tmpdir) / "does_not_exist")
        with self.assertRaises(FileNotFoundError):
            bot._load_documents()

    def test_raises_when_no_documents_found(self):
        self._write("notes.csv", "col1,col2")
        bot = _bare_bot()
        bot.data_directory = self.tmpdir
        with self.assertRaises(ValueError):
            bot._load_documents()


class TestGetResponse(unittest.TestCase):
    def test_returns_answer_and_context_on_success(self):
        bot = _bare_bot()
        node = MagicMock()
        node.text = "Relevant passage."
        retriever = MagicMock()
        retriever.retrieve.return_value = [node]
        bot.index = MagicMock()
        bot.query_engine = MagicMock()
        bot.query_engine.query.return_value = "The answer."

        with patch("rag_gemma.VectorIndexRetriever", return_value=retriever):
            result = bot.get_response("Inversion म्हणजे काय?")

        self.assertEqual(result["answer"], "The answer.")
        self.assertEqual(result["context"], "Relevant passage.")

    def test_returns_error_message_on_exception(self):
        bot = _bare_bot()
        bot.index = MagicMock()
        bot.query_engine = MagicMock()

        with patch("rag_gemma.VectorIndexRetriever", side_effect=RuntimeError("index unavailable")):
            result = bot.get_response("any query")

        self.assertIn("Error while generating answer", result["answer"])
        self.assertEqual(result["context"], "")


if __name__ == "__main__":
    unittest.main()
