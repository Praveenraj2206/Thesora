"""Central configuration for Thesora.

Every tunable constant lives here so no magic numbers appear elsewhere.
Each value can be overridden with an environment variable of the same name
(``THESORA_DATA_DIR``, ``THESORA_PAPERS_DIR`` and ``THESORA_API_URL`` for the
path/URL settings, ``THESORA_OFFLINE=1`` to use only cached Hugging Face models).
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable, falling back to ``default``."""
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env_str("THESORA_DATA_DIR", str(PROJECT_ROOT / "data")))
PAPERS_DIR = Path(_env_str("THESORA_PAPERS_DIR", str(DATA_DIR / "papers")))  # corpus for the offline scripts
EVAL_DIR = DATA_DIR / "eval"
GOLDEN_QA_PATH = EVAL_DIR / "golden_qa.json"
EVAL_RESULTS_PATH = EVAL_DIR / "results.json"

# --------------------------------------------------------------------------- runtime environment
# Must be set before sentence-transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows
OFFLINE = _env_str("THESORA_OFFLINE", "0") == "1"
if OFFLINE:  # use only the Hugging Face cache (models must have been downloaded once)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --------------------------------------------------------------------------- chat sessions
# Every chat is an isolated session with its own in-memory index. Nothing is written to
# disk, so a new chat / page refresh never sees PDFs or references from an earlier one.
MAX_FILES_PER_REQUEST = _env_int("MAX_FILES_PER_REQUEST", 10)
SESSION_TTL_MIN = _env_int("SESSION_TTL_MIN", 120)  # idle sessions are discarded after this
MAX_SESSIONS = _env_int("MAX_SESSIONS", 20)  # oldest idle session is evicted beyond this

# --------------------------------------------------------------------------- ingestion
CHUNK_SIZE = _env_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)
MIN_CHUNK_CHARS = _env_int("MIN_CHUNK_CHARS", 30)
HEADING_SIZE_RATIO = _env_float("HEADING_SIZE_RATIO", 1.15)
HEADING_MIN_CHARS = 3
HEADING_MAX_CHARS = 120
HEADING_BOLD_MAX_CHARS = 80
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 50)  # per file
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
PDF_MAGIC = b"%PDF"
MAX_FILENAME_CHARS = 150

# --------------------------------------------------------------------------- embeddings
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 384)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

# --------------------------------------------------------------------------- retrieval
DENSE_TOP_K = _env_int("DENSE_TOP_K", 20)
BM25_TOP_K = _env_int("BM25_TOP_K", 20)
FUSED_TOP_K = _env_int("FUSED_TOP_K", 8)
FINAL_TOP_K = _env_int("FINAL_TOP_K", 5)
RRF_K = _env_int("RRF_K", 60)
RERANKER_MODEL = _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 16
# Reranked chunks below this relevance are dropped from the LLM context (always keep MIN_CONTEXT_CHUNKS).
RELEVANCE_FLOOR = _env_float("RELEVANCE_FLOOR", 0.1)
MIN_CONTEXT_CHUNKS = _env_int("MIN_CONTEXT_CHUNKS", 1)

# --------------------------------------------------------------------------- abstention / confidence
ABSTAIN_THRESHOLD = _env_float("ABSTAIN_THRESHOLD", 0.35)
CONFIDENCE_TOP_N = _env_int("CONFIDENCE_TOP_N", 3)
# Weights must sum to 1.0: calibrated relevance, verified-source ratio, claim coverage.
CONFIDENCE_RELEVANCE_WEIGHT = _env_float("CONFIDENCE_RELEVANCE_WEIGHT", 0.5)
CONFIDENCE_VERIFICATION_WEIGHT = _env_float("CONFIDENCE_VERIFICATION_WEIGHT", 0.3)
CONFIDENCE_COVERAGE_WEIGHT = _env_float("CONFIDENCE_COVERAGE_WEIGHT", 0.2)
# Confidence-only calibration of the cross-encoder logit: sigmoid((logit - midpoint) / scale).
# A scale > 1 stops scores saturating at 1.0; abstention still uses the raw sigmoid relevance.
CONFIDENCE_LOGIT_MIDPOINT = _env_float("CONFIDENCE_LOGIT_MIDPOINT", 0.0)
CONFIDENCE_LOGIT_SCALE = _env_float("CONFIDENCE_LOGIT_SCALE", 3.0)
ABSTAIN_MESSAGE = "The documents don't contain enough information to answer this question."
NO_DOCUMENTS_MESSAGE = f"No documents in this chat yet. Upload PDFs first (up to {MAX_FILES_PER_REQUEST} per upload)."
INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

# --------------------------------------------------------------------------- citation verification
VERIFY_PARTIAL_THRESHOLD = _env_float("VERIFY_PARTIAL_THRESHOLD", 75.0)
VERIFY_RECALL_THRESHOLD = _env_float("VERIFY_RECALL_THRESHOLD", 0.7)
VERIFY_MIN_CONTENT_WORDS = _env_int("VERIFY_MIN_CONTENT_WORDS", 2)
VERIFY_STEM_LENGTH = _env_int("VERIFY_STEM_LENGTH", 5)
VERIFY_MIN_WORD_LENGTH = _env_int("VERIFY_MIN_WORD_LENGTH", 3)
VERIFY_NUMBER_GUARD = _env_str("VERIFY_NUMBER_GUARD", "1") == "1"  # every number in a claim must be in the chunk
VERIFY_NUMBER_MISMATCH_SCORE = _env_float("VERIFY_NUMBER_MISMATCH_SCORE", 50.0)  # score cap on a number mismatch
VERIFY_COVERAGE_MIN_WORDS = _env_int("VERIFY_COVERAGE_MIN_WORDS", 4)  # uncited sentences shorter than this are ignored

# --------------------------------------------------------------------------- LLM (Ollama)
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "qwen3:4b-instruct")  # non-thinking: no reasoning leaks
OLLAMA_TIMEOUT_S = _env_int("OLLAMA_TIMEOUT_S", 600)
OLLAMA_HEALTH_TIMEOUT_S = 3
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 8192)
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_KEEP_ALIVE = _env_str("OLLAMA_KEEP_ALIVE", "10m")
ASK_MAX_TOKENS = _env_int("ASK_MAX_TOKENS", 900)
SYNTH_MAX_TOKENS = _env_int("SYNTH_MAX_TOKENS", 1600)
JUDGE_MAX_TOKENS = 20

# --------------------------------------------------------------------------- synthesis
SYNTH_CHUNKS_PER_DOC = _env_int("SYNTH_CHUNKS_PER_DOC", 3)
SYNTH_MAX_CHUNKS = _env_int("SYNTH_MAX_CHUNKS", 24)
SYNTH_CANDIDATE_POOL = _env_int("SYNTH_CANDIDATE_POOL", 60)
SYNTH_MIN_CHUNK_RELEVANCE = _env_float("SYNTH_MIN_CHUNK_RELEVANCE", 0.1)

# --------------------------------------------------------------------------- API / UI
API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
API_URL = _env_str("THESORA_API_URL", f"http://{API_HOST}:{API_PORT}")
UI_REQUEST_TIMEOUT_S = _env_int("UI_REQUEST_TIMEOUT_S", 900)
UI_HEALTH_TIMEOUT_S = _env_int("UI_HEALTH_TIMEOUT_S", 5)
UI_SNIPPET_CHARS = 700
UI_PROGRESS_POLL_S = 1.0

# --------------------------------------------------------------------------- evaluation
FAITHFULNESS_MAX_SCORE = 5.0
CONTEXT_PRECISION_SIM_THRESHOLD = _env_float("CONTEXT_PRECISION_SIM_THRESHOLD", 0.45)
CONTEXT_RECALL_SIM_THRESHOLD = _env_float("CONTEXT_RECALL_SIM_THRESHOLD", 0.5)
JUDGE_CONTEXT_CHARS = 6000
EVAL_MIN_SENTENCE_CHARS = 15
