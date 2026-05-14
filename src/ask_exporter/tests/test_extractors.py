"""Tests for BOM extractor — LLM is always mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extractors.bom import extract_bom, hardware_names_from_bom

SAMPLE_CONFIG = {
    "llm": {"model": "llama3-70b-8192", "temperature": 0.1, "max_tokens": 4096}
}

_VALID_BOM_JSON = json.dumps({
    "hardware": [
        {
            "name": "Dilution refrigerator",
            "subcategory": "cryogenics",
            "specifications": {"base_temperature": "10 mK"},
            "part_number": None,
            "manufacturer": "Oxford Instruments",
            "quantity": "1",
            "estimated_cost": None,
            "notes": "",
        },
        {
            "name": "Ti:Sapphire laser",
            "subcategory": "laser",
            "specifications": {"wavelength": "780 nm", "power": "1 W"},
            "part_number": None,
            "manufacturer": "Coherent",
            "quantity": "2",
            "estimated_cost": None,
            "notes": "",
        },
    ],
    "software": [
        {
            "name": "QuTiP",
            "version": "4.7",
            "purpose": "Quantum dynamics simulation",
            "license": "BSD",
            "url": "qutip.org",
        }
    ],
    "materials": [
        {
            "name": "Silicon substrate",
            "subcategory": "substrate",
            "specification": "100mm wafer",
            "quantity": "10",
            "supplier": "Sigma-Aldrich",
            "estimated_cost": None,
        }
    ],
})


class TestExtractBom:
    @patch("src.extractors.bom.ChatGroq")
    def test_returns_hardware_software_materials(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock()
        mock_chain_result = MagicMock()
        mock_chain_result.content = _VALID_BOM_JSON

        # Patch the chain (prompt | llm) to return our mock
        with patch("src.extractors.bom.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(
                invoke=MagicMock(return_value=mock_chain_result)
            ))
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_groq_cls.return_value = mock_llm

            result = extract_bom("Sample paper text about quantum experiments.", SAMPLE_CONFIG)

        assert "hardware" in result
        assert "software" in result
        assert "materials" in result
        assert len(result["hardware"]) == 2
        assert result["hardware"][0]["name"] == "Dilution refrigerator"

    @patch("src.extractors.bom.ChatGroq")
    def test_handles_malformed_json_gracefully(self, mock_groq_cls):
        mock_chain_result = MagicMock()
        mock_chain_result.content = "Sorry, I cannot extract a BOM from this text."

        with patch("src.extractors.bom.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(
                invoke=MagicMock(return_value=mock_chain_result)
            ))
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_groq_cls.return_value = MagicMock()

            result = extract_bom("Some text", SAMPLE_CONFIG)

        assert result == {"hardware": [], "software": [], "materials": []}

    @patch("src.extractors.bom.ChatGroq")
    def test_extracts_json_from_fenced_output(self, mock_groq_cls):
        fenced = f"```json\n{_VALID_BOM_JSON}\n```"
        mock_chain_result = MagicMock(content=fenced)

        with patch("src.extractors.bom.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(
                invoke=MagicMock(return_value=mock_chain_result)
            ))
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_groq_cls.return_value = MagicMock()

            result = extract_bom("Sample text", SAMPLE_CONFIG)

        assert len(result["hardware"]) == 2


class TestHardwareNamesFromBom:
    def test_extracts_names(self):
        bom = json.loads(_VALID_BOM_JSON)
        names = hardware_names_from_bom(bom)
        assert "Dilution refrigerator" in names
        assert "Ti:Sapphire laser" in names

    def test_empty_hardware(self):
        bom = {"hardware": [], "software": [], "materials": []}
        assert hardware_names_from_bom(bom) == []

    def test_items_missing_name_skipped(self):
        bom = {"hardware": [{"subcategory": "cryogenics"}, {"name": "Laser"}]}
        names = hardware_names_from_bom(bom)
        assert names == ["Laser"]
