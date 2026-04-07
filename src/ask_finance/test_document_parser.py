"""
Unit tests for ask_finance document_parser module.

Tests focus on the pure-Python logic that does not require docling or a real PDF.
Run with: python -m pytest test_document_parser.py -v
"""

import base64
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from document_parser import (
    Chunk,
    TextChunk,
    TableChunk,
    ImageChunk,
    MultiModalChunker,
    save_chunks,
    load_chunks,
)


class TestChunkDataclasses(unittest.TestCase):
    """Test that chunk constructors set fields correctly."""

    def test_text_chunk_fields(self):
        chunk = TextChunk("hello world", {"source": "test.pdf"})
        self.assertEqual(chunk.content_type, "text")
        self.assertEqual(chunk.content, "hello world")
        self.assertEqual(chunk.metadata["source"], "test.pdf")
        self.assertIsNotNone(chunk.chunk_id)

    def test_table_chunk_fields(self):
        df = pd.DataFrame({"revenue": [100, 200], "year": [2022, 2023]})
        chunk = TableChunk(df, {"source": "report.pdf"})
        self.assertEqual(chunk.content_type, "table")
        self.assertIn("data", chunk.content)
        self.assertIn("columns", chunk.content)
        self.assertEqual(chunk.content["columns"], ["revenue", "year"])

    def test_table_chunk_sql_schema_generated(self):
        df = pd.DataFrame({"name": ["A"], "amount": [1.5], "count": [3]})
        chunk = TableChunk(df, {})
        schema = chunk.content["sql_schema"]
        self.assertIn("CREATE TABLE", schema)
        self.assertIn("name TEXT", schema)
        self.assertIn("amount REAL", schema)
        self.assertIn("count INTEGER", schema)

    def test_image_chunk_base64_roundtrip(self):
        raw_bytes = b"fake-image-data"
        chunk = ImageChunk(raw_bytes, {"source": "chart.png"})
        self.assertEqual(chunk.content_type, "image")
        decoded = base64.b64decode(chunk.content["image_base64"])
        self.assertEqual(decoded, raw_bytes)

    def test_image_chunk_default_description(self):
        chunk = ImageChunk(b"img", {})
        self.assertEqual(chunk.content["description"], "Financial chart or diagram")

    def test_to_dict_contains_required_keys(self):
        chunk = TextChunk("text", {"page": 1})
        d = chunk.to_dict()
        for key in ("chunk_id", "content_type", "content", "metadata", "embedding"):
            self.assertIn(key, d)


class TestGenerateSqlSchema(unittest.TestCase):
    """Test the SQL schema generation helper on TableChunk."""

    def test_integer_column(self):
        df = pd.DataFrame({"qty": pd.array([1, 2], dtype="int64")})
        schema = TableChunk._generate_sql_schema(df)
        self.assertIn("qty INTEGER", schema)

    def test_float_column(self):
        df = pd.DataFrame({"price": [1.1, 2.2]})
        schema = TableChunk._generate_sql_schema(df)
        self.assertIn("price REAL", schema)

    def test_text_column(self):
        df = pd.DataFrame({"name": ["Alpha", "Beta"]})
        schema = TableChunk._generate_sql_schema(df)
        self.assertIn("name TEXT", schema)


class TestChunkText(unittest.TestCase):
    """Test MultiModalChunker.chunk_text without touching docling."""

    def setUp(self):
        # Patch out the docling DocumentConverter so we don't need it installed
        from unittest.mock import patch, MagicMock
        patcher = patch("document_parser.DocumentConverter", return_value=MagicMock())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chunker = MultiModalChunker(text_chunk_size=100, text_overlap=20)

    def test_empty_string_returns_no_chunks(self):
        chunks = self.chunker.chunk_text("", {"source": "x"})
        self.assertEqual(chunks, [])

    def test_whitespace_only_returns_no_chunks(self):
        chunks = self.chunker.chunk_text("   \n  ", {"source": "x"})
        self.assertEqual(chunks, [])

    def test_short_text_single_chunk(self):
        chunks = self.chunker.chunk_text("Short.", {"source": "x"})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content_type, "text")

    def test_long_text_multiple_chunks(self):
        text = "word " * 100  # 500 chars, chunk_size=100
        chunks = self.chunker.chunk_text(text, {"source": "x"})
        self.assertGreater(len(chunks), 1)

    def test_chunk_metadata_contains_source(self):
        chunks = self.chunker.chunk_text("Hello world.", {"source": "myfile.pdf"})
        self.assertEqual(chunks[0].metadata["source"], "myfile.pdf")

    def test_chunk_metadata_contains_position_info(self):
        chunks = self.chunker.chunk_text("Hello world.", {"source": "x"})
        self.assertIn("start_char", chunks[0].metadata)
        self.assertIn("end_char", chunks[0].metadata)


class TestSaveLoadChunks(unittest.TestCase):
    """Test round-trip serialization of chunks to/from JSON."""

    def test_text_chunk_roundtrip(self):
        original = [TextChunk("financial report text", {"page": 1})]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_chunks(original, path)
            loaded = load_chunks(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].content, "financial report text")
            self.assertEqual(loaded[0].content_type, "text")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_table_chunk_roundtrip(self):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        original = [TableChunk(df, {"table_index": 0})]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_chunks(original, path)
            loaded = load_chunks(path)
            self.assertEqual(loaded[0].content_type, "table")
            self.assertEqual(loaded[0].content["columns"], ["col_a", "col_b"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_image_chunk_roundtrip(self):
        raw = b"\x89PNG\r\n\x1a\n"
        original = [ImageChunk(raw, {"figure_index": 0}, description="Revenue chart")]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_chunks(original, path)
            loaded = load_chunks(path)
            self.assertEqual(loaded[0].content_type, "image")
            self.assertEqual(loaded[0].content["description"], "Revenue chart")
            decoded = base64.b64decode(loaded[0].content["image_base64"])
            self.assertEqual(decoded, raw)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_chunk_id_preserved_across_roundtrip(self):
        chunk = TextChunk("preserved id test", {})
        original_id = chunk.chunk_id
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_chunks([chunk], path)
            loaded = load_chunks(path)
            self.assertEqual(loaded[0].chunk_id, original_id)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
