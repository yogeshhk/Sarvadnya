"""Entry point — run from src/ask_exporter/ with: streamlit run streamlit_main.py"""

import sys
from pathlib import Path

# Add this directory to path so `src` package is importable
sys.path.insert(0, str(Path(__file__).parent))

# Streamlit re-executes the top-level script on every user interaction.
# exec() in globals() scope ensures st.* calls are visible to Streamlit's runner
# and session_state / widget keys behave identically to running the file directly.
_app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
exec(compile(_app_path.read_text(encoding="utf-8"), str(_app_path), "exec"), globals())  # noqa: S102
