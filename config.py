"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Multilingual model: handles Hindi + English queries against the same index.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
# FAISS returns L2 distance: LOWER = more similar. Chunks scoring above this are dropped.
# Calibrate with:  python analyze_scores.py
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "30"))

# LLM (any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, ...)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
