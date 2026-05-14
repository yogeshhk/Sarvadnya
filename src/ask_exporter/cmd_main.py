"""
Headless CLI entry point — no argparse, all defaults from config.yaml.

Usage (from src/ask_exporter/):
    python cmd_main.py

To override input at runtime without touching config.yaml, set env vars:
    HEADLESS_INPUT_TYPE=arxiv
    HEADLESS_RAW_INPUT=2301.07715
    python cmd_main.py
"""

import json
import os
import sys
from pathlib import Path

# Make `src` importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import format_report, run_from_config, run_pipeline
from src.utils.helpers import load_config, setup_logger, ensure_dirs

logger = setup_logger(__name__)


def main() -> None:
    cfg = load_config()
    h = cfg.get("headless", {})

    # Allow lightweight env-var overrides without editing config.yaml
    input_type = os.environ.get("HEADLESS_INPUT_TYPE", h.get("input_type", "direct_items"))
    if input_type == "arxiv":
        raw_input = os.environ.get("HEADLESS_RAW_INPUT", h.get("arxiv_id", ""))
    else:
        raw_input = os.environ.get("HEADLESS_RAW_INPUT", h.get("raw_input", ""))

    fmt = h.get("output_format", "json")
    output_file = h.get("output_file")

    logger.info("Running headless pipeline — input_type=%s", input_type)
    report = run_pipeline(input_type=input_type, raw_input=raw_input, config=cfg)

    output = format_report(report, fmt=fmt)

    if output_file:
        ensure_dirs(str(Path(output_file).parent))
        Path(output_file).write_text(output, encoding="utf-8")
        logger.info("Report written to %s", output_file)
    else:
        print(output)


if __name__ == "__main__":
    main()
