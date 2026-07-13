"""
Unit tests for ask_bharat's pure helper functions (retrieval-query building,
cross-encoder reranking, citation formatting/linkification).

streamlit_main.py is a Streamlit script: importing it normally executes
st.set_page_config(), loads a real FAISS vectorstore, and requires
GROQ_API_KEY, none of which are available in a test environment. To avoid
mocking the entire Streamlit runtime, this test extracts just the standalone
helper functions via ast and execs them in an isolated namespace, so the
*real* function bodies are exercised without running the rest of the script.

Run with: python -m pytest test_streamlit_main.py -v
"""

import ast
import os
import re
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent / "streamlit_main.py"
_WANTED_FUNCS = {
    "build_retrieval_query",
    "rerank",
    "format_context",
    "extract_citation_meta",
    "linkify_citations",
}


def _load_pure_functions():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {"os": os, "re": re, "DATA_DIR": "/fake/data", "CrossEncoder": object}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, filename=str(MODULE_PATH), mode="exec"), namespace)
    missing = _WANTED_FUNCS - namespace.keys()
    if missing:
        raise RuntimeError(f"Could not find expected function(s) in streamlit_main.py: {missing}")
    return namespace


_ns = _load_pure_functions()
build_retrieval_query = _ns["build_retrieval_query"]
rerank = _ns["rerank"]
format_context = _ns["format_context"]
extract_citation_meta = _ns["extract_citation_meta"]
linkify_citations = _ns["linkify_citations"]


class FakeDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class StubReranker:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores


class TestBuildRetrievalQuery(unittest.TestCase):
    def test_no_history_returns_message_unchanged(self):
        result = build_retrieval_query("What is the capital?", [])
        self.assertEqual(result, "What is the capital?")

    def test_includes_last_turn_when_history_present(self):
        history = [
            {"role": "user", "content": "Tell me about archaeological sites."},
            {"role": "assistant", "content": "Sure, here are a few sites."},
        ]
        result = build_retrieval_query("What are their locations?", history)
        self.assertIn("archaeological sites", result)
        self.assertIn("What are their locations?", result)

    def test_only_uses_last_two_messages(self):
        history = [
            {"role": "user", "content": "OLDEST MESSAGE"},
            {"role": "assistant", "content": "OLD REPLY"},
            {"role": "user", "content": "Recent question"},
            {"role": "assistant", "content": "Recent answer"},
        ]
        result = build_retrieval_query("Follow-up", history)
        self.assertNotIn("OLDEST MESSAGE", result)
        self.assertIn("Recent answer", result)


class TestRerank(unittest.TestCase):
    def test_empty_docs_returned_unchanged(self):
        result = rerank("query", [], StubReranker([]), top_k=3)
        self.assertEqual(result, [])

    def test_sorts_by_score_descending_and_truncates(self):
        docs = [FakeDoc("a"), FakeDoc("b"), FakeDoc("c")]
        reranker = StubReranker([0.1, 0.9, 0.5])
        result = rerank("query", docs, reranker, top_k=2)
        self.assertEqual([d.page_content for d in result], ["b", "c"])


class TestFormatContext(unittest.TestCase):
    def test_numbers_passages_from_one(self):
        docs = [FakeDoc("Hello"), FakeDoc("World")]
        result = format_context(docs)
        self.assertEqual(result, "[1] Hello\n\n[2] World")

    def test_empty_docs_returns_empty_string(self):
        self.assertEqual(format_context([]), "")


class TestExtractCitationMeta(unittest.TestCase):
    def test_builds_expected_fields(self):
        doc = FakeDoc("Some excerpt text.", {"source": "dir/file.pdf", "page": 3})
        cards = extract_citation_meta([doc])
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card["num"], 1)
        self.assertEqual(card["src_name"], "file.pdf")
        self.assertEqual(card["page"], 3)
        self.assertEqual(card["excerpt"], "Some excerpt text.")

    def test_truncates_long_excerpt_with_ellipsis(self):
        long_text = "x" * 400
        doc = FakeDoc(long_text, {"source": "file.pdf", "page": 1})
        card = extract_citation_meta([doc])[0]
        self.assertTrue(card["excerpt"].endswith("…"))
        self.assertEqual(len(card["excerpt"]), 351)  # 350 chars + ellipsis

    def test_missing_source_reports_unknown(self):
        doc = FakeDoc("text", {})
        card = extract_citation_meta([doc])[0]
        self.assertEqual(card["src_name"], "Unknown source")
        self.assertEqual(card["page"], "?")


class TestLinkifyCitations(unittest.TestCase):
    def test_replaces_markers_with_anchor_links(self):
        result = linkify_citations("Claim one [1] and claim two [2].")
        self.assertIn('href="#cite-1"', result)
        self.assertIn('href="#cite-2"', result)
        self.assertIn(">[1]</a>", result)

    def test_leaves_text_without_markers_unchanged(self):
        result = linkify_citations("No citations here.")
        self.assertEqual(result, "No citations here.")


if __name__ == "__main__":
    unittest.main()
