"""All settings in one place, read from environment variables (.env supported)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


# --- LLM (Grok via xAI's OpenAI-compatible API) ---
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "grok-4")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "0")  # empty string = omit
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# --- Agent ---
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
MAX_CONSECUTIVE_TOOL_ERRORS = int(os.getenv("MAX_CONSECUTIVE_TOOL_ERRORS", "3"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")

# --- Retrieval ---
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # vector | keyword | hybrid
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.85"))  # cosine distance cut-off
EMBEDDINGS = os.getenv("EMBEDDINGS", "default")  # default (MiniLM ONNX) | hash (offline/tests)
DOCS_DIR = _path("DOCS_DIR", ROOT / "data" / "docs")
TICKETS_FILE = _path("TICKETS_FILE", ROOT / "data" / "tickets.json")
CHROMA_DIR = _path("CHROMA_DIR", ROOT / ".chroma")

# --- Cost estimate (optional, USD per 1M tokens) ---
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M") or 0)
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M") or 0)
