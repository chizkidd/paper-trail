"""
Central configuration for Paper-trail.

All model names, hyperparameters, and limits live here so they can be
tuned in one place rather than scattered across the codebase.
"""

# ── Retrieval ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Weight for dense (semantic) retrieval in the hybrid score.
# Remaining (1 - HYBRID_ALPHA) goes to sparse (TF-IDF).
HYBRID_ALPHA = 0.45

TFIDF_MAX_FEATURES = 30_000

# ── Chunking ───────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 800   # target chunk size in words
CHUNK_OVERLAP = 30    # words of overlap between adjacent chunks

# ── LLM — Hugging Face ────────────────────────────────────────────────────────

HF_MODEL          = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL        = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TIMEOUT        = 45   # seconds
HF_MAX_NEW_TOKENS = 300
HF_TEMPERATURE    = 0.2

# ── LLM — Ollama ──────────────────────────────────────────────────────────────

OLLAMA_MODEL   = "llama3.1:8b"
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 60   # seconds

# ── Ingest ────────────────────────────────────────────────────────────────────

# Maximum allowed PDF size. Files larger than this are rejected before parsing
# to avoid exhausting memory on the server.
PDF_MAX_BYTES = 50 * 1024 * 1024   # 50 MB

# Minimum characters for a page to be considered non-scanned text.
PDF_SCANNED_THRESHOLD = 40

# ── Cache ─────────────────────────────────────────────────────────────────────

import os

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".papertrail_cache")
