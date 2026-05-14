"""LLM-based BOM extraction from paper text (single structured prompt)."""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from ..utils.helpers import load_config, setup_logger
from ..utils.parsers import truncate

logger = setup_logger(__name__)

_PROMPT = """\
You are an expert at reading quantum physics research papers and extracting a Bill of Materials (BOM).

Given the following paper text, extract ALL hardware/equipment, software, and materials mentioned.

Text:
{text}

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{{
  "hardware": [
    {{
      "name": "item name",
      "subcategory": "laser|detector|optics|cryogenics|electronics|quantum_device|vacuum|measurement|other",
      "specifications": {{"key": "value"}},
      "part_number": null,
      "manufacturer": null,
      "quantity": null,
      "estimated_cost": null,
      "notes": ""
    }}
  ],
  "software": [
    {{
      "name": "software/framework name",
      "version": null,
      "purpose": "description",
      "license": null,
      "url": null
    }}
  ],
  "materials": [
    {{
      "name": "material/chemical name",
      "subcategory": "chemical|substrate|gas|fiber|consumable|other",
      "specification": null,
      "quantity": null,
      "supplier": null,
      "estimated_cost": null
    }}
  ]
}}

Rules:
- Extract only items explicitly mentioned in the text.
- Capture specifications (wavelength, power, model numbers, temperature, etc.).
- Use null for fields not mentioned.
- For quantum experiments pay special attention to: cryogenic systems, qubit devices, laser systems, detectors, microwave electronics, vacuum chambers.
- Return valid JSON only.\
"""


def extract_bom(text: str, config: dict | None = None) -> dict:
    """
    Call the LLM with the paper text and return a raw BOM dict with keys
    'hardware', 'software', 'materials'.
    """
    cfg = config or load_config()
    llm_cfg = cfg.get("llm", {})

    llm = ChatGroq(
        model=llm_cfg.get("model", "llama3-70b-8192"),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
    )

    prompt = ChatPromptTemplate.from_template(_PROMPT)
    chain = prompt | llm

    # Keep within context window
    truncated_text = truncate(text, max_chars=6000)
    response = chain.invoke({"text": truncated_text})
    raw = response.content.strip()

    # Robust JSON extraction — handles LLM wrapping in markdown fences
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("BOM JSON parse failed; returning empty BOM")
    return {"hardware": [], "software": [], "materials": []}


def hardware_names_from_bom(bom: dict) -> list[str]:
    """Extract unique hardware item names for export-control checking."""
    return [item.get("name", "") for item in bom.get("hardware", []) if item.get("name")]
