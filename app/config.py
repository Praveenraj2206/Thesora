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
MAX_FILES_PER_REQUEST = _env_int("MAX_FILES_PER_REQUEST", 0)  # PDFs per upload; 0 = any number
SESSION_TTL_MIN = _env_int("SESSION_TTL_MIN", 120)  # idle sessions are discarded after this
MAX_SESSIONS = _env_int("MAX_SESSIONS", 20)  # oldest idle session is evicted beyond this

# --------------------------------------------------------------------------- ingestion
PDF_SPACE_GAP_EM = _env_float("PDF_SPACE_GAP_EM", 0.08)
# Layout cleaning (app/ingestion/clean_layout.py): keep a paper's content, drop headers, authors, figure labels.
PDF_CLEAN = _env_str("PDF_CLEAN", "1") == "1"
PDF_TINY_FONT_RATIO = _env_float("PDF_TINY_FONT_RATIO", 0.72)  # text below this x body font size is dropped
PDF_MARGIN_BAND = _env_float("PDF_MARGIN_BAND", 0.06)  # top/bottom page share treated as header/footer
PDF_REPEAT_BAND = 0.12  # top/bottom share searched for text repeated on many pages
PDF_REPEAT_MIN_SHARE = 0.3  # repeated on this share of pages (at least 3) = running header/footer
PDF_FIGURE_LABEL_MAX_WORDS = 8  # shorter text inside a figure is a label
PDF_FIGURE_MIN_AREA = 2000.0  # drawing clusters at least this big (pt^2) count as figures
PDF_PARAGRAPH_MIN_WORDS = 25  # a block this long and mostly lower-case is running prose
PDF_FOOTNOTE_MAX_WORDS = 15
PDF_PROSE_MIN_WORDS = 15  # sentence-like text this long is kept even in a small font or inside a box
PDF_COLUMN_TOLERANCE = 20.0  # points a column block may cross the page middle
PDF_SKIP_REFERENCES = _env_str("PDF_SKIP_REFERENCES", "1") == "1"  # the reference list is not searched
PDF_DROP_SUPERSCRIPT_MARKERS = _env_str("PDF_DROP_SUPERSCRIPT_MARKERS", "1") == "1"  # "latest²" -> "latest"  # gap (x font size) that counts as a word space
CHUNK_SIZE = _env_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)
MIN_CHUNK_CHARS = _env_int("MIN_CHUNK_CHARS", 30)
HEADING_SIZE_RATIO = _env_float("HEADING_SIZE_RATIO", 1.15)
HEADING_MIN_CHARS = 3
HEADING_MAX_CHARS = 120
HEADING_BOLD_MAX_CHARS = 80
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 15)  # per file
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
SCOPED_POOL = _env_int("SCOPED_POOL", 15)  # dense and BM25 candidates searched inside a paper the question names
SCOPED_MAX_TABLES = 20  # all tables of a named paper are reranked (up to this many)
PAPER_NAME_PAGES = 3  # pages scanned for a paper's system names (OpenScholar, LitQA)
PAPER_NAME_MIN_COUNT = 5  # a mixed-case name must appear this often to count as the paper's own
FINAL_TOP_K = _env_int("FINAL_TOP_K", 5)
TABLE_SLOT = _env_str("TABLE_SLOT", "1") == "1"  # number questions keep the best relevant table in the last slot
TABLE_SLOT_MIN_RELEVANCE = _env_float("TABLE_SLOT_MIN_RELEVANCE", 0.5)
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
NO_DOCUMENTS_MESSAGE = f"No documents in this chat yet. Upload PDFs first (up to {MAX_UPLOAD_MB} MB each)."
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

# --------------------------------------------------------------------------- question refiner (prompt-master)
REFINE_MAX_TOKENS = _env_int("REFINE_MAX_TOKENS", 200)
REFINE_MAX_WORDS = _env_int("REFINE_MAX_WORDS", 60)  # instruction given to the model
REFINE_WORD_SLACK = 20  # replies up to this many words over the instruction are kept whole

# --------------------------------------------------------------------------- synthesis
SYNTH_CHUNKS_PER_DOC = _env_int("SYNTH_CHUNKS_PER_DOC", 3)
SYNTH_MAX_CHUNKS = _env_int("SYNTH_MAX_CHUNKS", 24)
SYNTH_CANDIDATE_POOL = _env_int("SYNTH_CANDIDATE_POOL", 60)
SYNTH_DOC_POOL = _env_int("SYNTH_DOC_POOL", 8)  # candidates searched inside each paper for Compare
SYNTH_COVER_ALL_PAPERS = _env_str("SYNTH_COVER_ALL_PAPERS", "1") == "1"  # every paper keeps its best passage
SYNTH_MIN_CHUNK_RELEVANCE = _env_float("SYNTH_MIN_CHUNK_RELEVANCE", 0.1)

# --------------------------------------------------------------------------- API / UI
API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
API_URL = _env_str("THESORA_API_URL", f"http://{API_HOST}:{API_PORT}")
UI_REQUEST_TIMEOUT_S = _env_int("UI_REQUEST_TIMEOUT_S", 900)
UI_HEALTH_TIMEOUT_S = _env_int("UI_HEALTH_TIMEOUT_S", 20)  # laptops: model loading at startup can stall replies
UI_SNIPPET_CHARS = 700
UI_CITATION_PREVIEW_CHARS = 320
UI_CLOSEST_MIN_RELEVANCE = 0.1  # after a refusal, quote the closest passage only if it is at least loosely related
UI_CLOSEST_CHARS = 280
UI_TRY_INSTEAD = 2  # replacement questions under a follow-up the papers could not answer  # passage preview inside the Citations pop-up
UI_SIDEBAR_DOCS_HEIGHT = 230  # px: about three paper cards, scroll for the rest
UI_GLOSSARY_HEIGHT = 300  # px: about five terms, scroll for the rest
UI_PROGRESS_POLL_S = 1.0
UI_CONFIDENCE_HIGH = _env_float("UI_CONFIDENCE_HIGH", 0.7)  # confidence meter turns green at/above this
UI_CONFIDENCE_MEDIUM = _env_float("UI_CONFIDENCE_MEDIUM", 0.4)  # amber at/above this, red below
UI_DEFAULT_THEME = _env_str("UI_DEFAULT_THEME", "light")  # "light" or "dark"

# --------------------------------------------------------------------------- evaluation
FAITHFULNESS_MAX_SCORE = 5.0
CONTEXT_PRECISION_SIM_THRESHOLD = _env_float("CONTEXT_PRECISION_SIM_THRESHOLD", 0.45)
CONTEXT_RECALL_SIM_THRESHOLD = _env_float("CONTEXT_RECALL_SIM_THRESHOLD", 0.5)
JUDGE_CONTEXT_CHARS = 6000
EVAL_MIN_SENTENCE_CHARS = 15

# --------------------------------------------------------------------------- v2 batch A (trust)
NLI_ENABLED = _env_str("NLI_ENABLED", "0") == "1"  # N8; off by default: smallest model is 283 MB (> 200 MB budget)
NLI_MODEL = _env_str("NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
NLI_PASS = _env_float("NLI_PASS", 0.5)  # entailment probability for PASS
NLI_FAIL_CONTRADICTION = _env_float("NLI_FAIL_CONTRADICTION", 0.5)  # contradiction probability for FAIL
NLI_PREMISE_SENTENCES = _env_int("NLI_PREMISE_SENTENCES", 2)
SENTENCE_WEAK_SCORE = _env_float("SENTENCE_WEAK_SCORE", 60.0)  # word score for "weak" (below = unverified)
SELF_REPAIR = _env_str("SELF_REPAIR", "1") == "1"  # one stricter retry, then drop unsupported sentences (N10)
REPAIR_MAX_TOKENS = _env_int("REPAIR_MAX_TOKENS", 600)
METADATA_TITLE_MAX_CHARS = 250
METADATA_AUTHOR_MAX_CHARS = 300
METADATA_AUTHOR_LINES = 12  # lines after the title scanned for author names
METADATA_HEADER_FRACTION = 0.45  # page-1 share treated as the header when no Abstract heading is found
METADATA_TITLE_LINE_GAP = 2.2  # a bigger jump (x title size) between lines ends the title
METADATA_NAME_GAP_EM = 1.0  # a wider gap between spans separates two names on one line
LARGE_FILE_WARN_MB = _env_int("LARGE_FILE_WARN_MB", 5)  # N7: UI warns above this size
MAX_SESSION_PDF_MB = _env_int("MAX_SESSION_PDF_MB", 200)  # N7: total PDF size allowed per chat
PARTIAL_ANSWERS = _env_str("PARTIAL_ANSWERS", "1") == "1"  # N6: answer multi-part questions part by part
PART_MIN_WORDS = _env_int("PART_MIN_WORDS", 4)

# --------------------------------------------------------------------------- v2 N9 / N11
VIEWER_WIDTH_PX = _env_int("VIEWER_WIDTH_PX", 300)  # explicit width: the viewer needs one inside tabs (fits the 30% panel)
HIGHLIGHT_COLOR = _env_str("HIGHLIGHT_COLOR", "#E8A317")
HIGHLIGHT_MIN_SENTENCE_CHARS = 25
EVIDENCE_MIN_WORDS = 5  # shorter 'sentences' (run-in labels like 'Query Validation.') are never evidence
EVIDENCE_MIN_SCORE = 40.0  # a sentence must match a claim at least this well (0-100) to be highlighted
EVIDENCE_SCORE_MARGIN = 10.0  # other sentences within this of the best are highlighted too
EVIDENCE_MAX_SENTENCES = 2
HIGHLIGHT_PIECE_CHARS = 30  # a sentence not found whole is matched in pieces of this many letters
HIGHLIGHT_MIN_COVERAGE = 0.3  # share of the passage that must be found on the page to show a highlight
PAGE_PNG_ZOOM = _env_float("PAGE_PNG_ZOOM", 1.5)  # fallback page image when the viewer cannot load
EXPLAIN_MAX_TOKENS = _env_int("EXPLAIN_MAX_TOKENS", 350)
SUGGEST_MAX_TOKENS = _env_int("SUGGEST_MAX_TOKENS", 260)  # up to six candidate questions
SUGGEST_CANDIDATES = 6  # the model writes this many; only answerable ones are shown
SUGGEST_COUNT = 3  # questions shown (main chat and Explain "Ask next")
EXPLAIN_FOLLOW_UP_CANDIDATES = 5
SUGGEST_MIN_RELEVANCE = _env_float("SUGGEST_MIN_RELEVANCE", 0.5)  # a suggestion's best passage must score this
SUGGEST_HISTORY = 5  # recent questions the suggestions should not repeat
EXPLAIN_MAX_SELECTION_CHARS = _env_int("EXPLAIN_MAX_SELECTION_CHARS", 1500)
GLOSSARY_ENABLED = _env_str("GLOSSARY_ENABLED", "1") == "1"
GLOSSARY_MAX_TERMS = _env_int("GLOSSARY_MAX_TERMS", 200)
PDF_PANEL_MODE = _env_str("PDF_PANEL_MODE", "zoom")  # "zoom" (sharp zoomable page + highlight), "viewer" (pdf.js) or "image"
PANEL_PAGE_ZOOM = _env_float("PANEL_PAGE_ZOOM", 3.0)  # page rendered at 3 x 72 = 216 dpi: stays sharp when zoomed
PANEL_PAGE_ZOOM_MAX = 4.0
PANEL_HEIGHT_PX = _env_int("PANEL_HEIGHT_PX", 520)  # height of the zoomable page area

# --------------------------------------------------------------------------- v2 N1 / N2 / N12
PDF_LAYOUT_MODE = _env_str("PDF_LAYOUT_MODE", "auto")  # "auto": add table chunks; "plain": text only (old behaviour)
TABLE_MIN_ROWS = _env_int("TABLE_MIN_ROWS", 2)
TABLE_MIN_RULES = 3  # pages with fewer horizontal rules are not searched for tables (speed)
TABLE_MIN_CELLS_PER_ROW = 1.3  # boxes filling no more cells per row than this are text in a box, not a table
TABLE_CAPTION_MAX_GAP = 60.0  # points between a table and its "Table N:" caption
TABLE_CAPTION_MAX_CHARS = 400
COLUMN_GAP_RATIO = _env_float("COLUMN_GAP_RATIO", 0.15)  # check_extraction.py: share of blocks on each side = two columns
CONTEXT_HEADERS = _env_str("CONTEXT_HEADERS", "0") == "1"  # N2a (default set from compare_retrievers results)
MULTI_QUERY = _env_str("MULTI_QUERY", "0") == "1"  # N2b: off by default (one extra LLM call)
MULTI_QUERY_VARIANTS = _env_int("MULTI_QUERY_VARIANTS", 2)
MULTI_QUERY_MAX_TOKENS = _env_int("MULTI_QUERY_MAX_TOKENS", 60)
CONTEXT_EXPANSION = _env_str("CONTEXT_EXPANSION", "1") == "1"  # N2c: give the LLM the top passages plus their neighbours
CONTEXT_EXPANSION_TOP = _env_int("CONTEXT_EXPANSION_TOP", 2)  # only the best N passages are expanded (keeps prompts short)
CHUNK_PARENT_CHARS = _env_int("CHUNK_PARENT_CHARS", 1200)
AGREE_NUMBER_WINDOW = _env_int("AGREE_NUMBER_WINDOW", 6)  # words around a number that name its metric
AGREE_MIN_RELEVANCE = _env_float("AGREE_MIN_RELEVANCE", 0.3)  # a paper is quoted only if its passage is clearly on topic
