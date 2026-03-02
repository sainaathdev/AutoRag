"""HuggingFace Spaces / Streamlit Cloud entrypoint.

This file is the universal entry point for deployment.
It patches paths and environment, then delegates to dashboard.py.
"""
import os
import sys
from pathlib import Path

# ── CRITICAL: add the project root to sys.path ───────────────────────
# HuggingFace Spaces runs the app from /app, but Python's module
# search path may not include it. This ensures rag_system, ingestion,
# retrieval, agents, feedback, utils are all importable.
APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ── Persistent storage on HuggingFace Spaces ────────────────────────
# HF Spaces provides /data as a persistent volume.
# Fall back to ./data for local runs and other platforms.
HF_DATA_DIR = Path("/data") if Path("/data").exists() else Path("./data")
HF_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Patch environment so config.yaml picks up the right paths
os.environ.setdefault("VECTOR_DB_PATH", str(HF_DATA_DIR / "vector_db"))
os.environ.setdefault("FEEDBACK_PATH",  str(HF_DATA_DIR / "feedback"))

# ── Load secrets from HF Spaces or local .env ───────────────────────
# On HF Spaces, secrets are injected as environment variables automatically.
# Locally, python-dotenv loads them from .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Run the dashboard ────────────────────────────────────────────────
# Use exec with __file__ set so relative imports inside dashboard.py
# resolve correctly against the project root.
_dashboard_path = APP_DIR / "dashboard.py"
exec(compile(_dashboard_path.read_text(encoding="utf-8"), str(_dashboard_path), "exec"))  # noqa: S102
