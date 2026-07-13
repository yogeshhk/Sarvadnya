"""
Unit tests for ask_dataframe DfEngine query logic.

Run with: python -m pytest test_dfengine.py -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dfengine import DfEngine


class TestDfEngineQueryRasa(unittest.TestCase):
    """Tests for the HTTP call to the local Rasa NLU server."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        csv_path = Path(self.tmpdir) / "countries.csv"
        pd.DataFrame({
            "Country": ["China", "Congo"],
            "Population": ["1.4 billion", "5.5 million"],
        }).to_csv(csv_path, index=False, encoding="ISO-8859-1")
        self.engine = DfEngine(str(csv_path), "Country")

    @patch("dfengine.requests.post")
    def test_query_rasa_returns_parsed_json(self, mock_post):
        mock_post.return_value.json.return_value = {"intent": {"name": "query"}, "entities": []}
        result = self.engine.query_rasa("What is Population for China?")
        self.assertEqual(result["intent"]["name"], "query")

    @patch("dfengine.requests.post", side_effect=ConnectionError("Rasa server not running"))
    def test_query_rasa_handles_connection_error(self, mock_post):
        result = self.engine.query_rasa("anything")
        self.assertEqual(result, {"intent": {"name": "none"}, "entities": []})


class TestDfEngineProcessQueryIntent(unittest.TestCase):
    """Tests for the row/column lookup logic (also exercises the KeyError/IndexError fix)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        csv_path = Path(self.tmpdir) / "countries.csv"
        pd.DataFrame({
            "Country": ["China", "Congo"],
            "Population": ["1.4 billion", "5.5 million"],
        }).to_csv(csv_path, index=False, encoding="ISO-8859-1")
        self.engine = DfEngine(str(csv_path), "Country")

    def test_returns_value_for_known_row_and_column(self):
        entities = [{"entity": "row", "value": "china"}, {"entity": "column", "value": "population"}]
        result = self.engine.process_query_intent(entities)
        self.assertEqual(result, "1.4 billion")

    def test_returns_not_found_message_for_unknown_column(self):
        entities = [{"entity": "row", "value": "china"}, {"entity": "column", "value": "gdp"}]
        result = self.engine.process_query_intent(entities)
        self.assertIn("couldn't find data", result)

    def test_returns_not_found_message_for_unknown_row(self):
        entities = [{"entity": "row", "value": "atlantis"}, {"entity": "column", "value": "population"}]
        result = self.engine.process_query_intent(entities)
        self.assertIn("couldn't find data", result)

    def test_missing_row_or_column_prompts_for_both(self):
        result = self.engine.process_query_intent([{"entity": "row", "value": "china"}])
        self.assertIn("specify both", result)


class TestDfEngineQueryDispatch(unittest.TestCase):
    """Tests for intent dispatch in query()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        csv_path = Path(self.tmpdir) / "countries.csv"
        pd.DataFrame({"Country": ["China"], "Population": ["1.4 billion"]}).to_csv(
            csv_path, index=False, encoding="ISO-8859-1"
        )
        self.engine = DfEngine(str(csv_path), "Country")

    @patch("dfengine.DfEngine.query_rasa")
    def test_greet_intent_returns_std_response(self, mock_query_rasa):
        mock_query_rasa.return_value = {"intent": {"name": "greet"}, "entities": []}
        result = self.engine.query("Hi")
        self.assertIn(result, self.engine.std_responses["greet"])

    @patch("dfengine.DfEngine.query_rasa")
    def test_unknown_intent_returns_fallback(self, mock_query_rasa):
        mock_query_rasa.return_value = {"intent": {"name": "nlu_fallback"}, "entities": []}
        result = self.engine.query("gibberish")
        self.assertIn("didn't understand", result)


if __name__ == "__main__":
    unittest.main()
