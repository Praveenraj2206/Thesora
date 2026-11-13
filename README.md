# Thesora: Evidence-Grounded RAG Research Assistant

Backend-first RAG over research PDFs that *knows when it doesn't know*. Python 3.11+, CPU-only,
Windows-friendly. The LLM is local Ollama (`qwen3:4b-instruct`) behind a `BaseLLM` interface. No API keys.

Pipeline: PDF → PyMuPDF text + heading/section detection → RecursiveCharacterTextSplitter (512/100) →
all-MiniLM-L6-v2 (384-d, normalised) → FAISS IndexFlatIP + BM25 → cross-encoder rerank → Ollama →
citation verification → FastAPI → Streamlit.

## The 5 improvements over naive RAG (+ stretch)
| # | Feature | Where |
|---|---|---|
| F1 | **Hybrid retrieval**: BM25 top 20 + dense top 20, Reciprocal Rank Fusion (k=60) → top 8 | `app/retrieval/retriever.py` |
| F2 | **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` → top 5, relevance = sigmoid(logit); chunks below `RELEVANCE_FLOOR` (0.1) dropped | `app/retrieval/retriever.py` |
| F3 | **Citation verification**: each `[n]` claim fuzzy-matched (rapidfuzz partial ratio + stemmed content-word recall) against chunk n, number guard, joint check for `[1][2]` claims; per-source `cited` / `verified` / `match_score` | `app/generation/verifier.py` |
| F4 | **Abstention + confidence**: best relevance < 0.35 → abstain *without calling the LLM*; model reply `INSUFFICIENT_CONTEXT` → abstain. Confidence = 0.5·calibrated relevance + 0.3·verified ratio + 0.2·claim coverage | `app/pipelines/rag.py` |
| F5 | **Evaluation harness**: faithfulness (local LLM judge 0–5), context precision/recall, answer relevancy, abstention accuracy, retrieval hit rate → `results.json` | `scripts/evaluate.py` |
| ★ | **Cross-paper synthesis**: comparison table + "Research Gaps", chunks picked round-robin so every paper is covered | `POST /sessions/{id}/synthesize` |

## Chat rules
- **Max 10 PDFs per upload request** (`MAX_FILES_PER_REQUEST`). An 11-file request is rejected with HTTP 400
  before anything is ingested; the UI blocks it too. Each file is also limited to `MAX_UPLOAD_MB` (50).
- **Every chat starts empty.** Each chat is a backend session with its own in-memory FAISS + BM25 index.
  Opening the UI, refreshing the page or pressing **New chat** creates a new session, so earlier PDFs and
  references are never used. Nothing is written to disk: uploads are parsed from a temporary folder that is
  deleted straight away, ended chats are freed immediately, idle chats expire after `SESSION_TTL_MIN` (120),
  and a server restart forgets everything.

## Setup (Windows, Git Bash)
```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
ollama pull qwen3:4b-instruct
```
The embedding model and cross-encoder download from Hugging Face on first use (internet needed once;
afterwards `export THESORA_OFFLINE=1` uses only the cache).

## Run
```bash
# 1. Ollama must be running (desktop app, or: ollama serve)
# 2. Backend (no --reload on Windows: StatReload crashes)
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
# 3. UI, in a second terminal
streamlit run ui/streamlit_app.py
```

## Check everything works
```bash
python -m unittest discover -s tests -v        # 36 tests; no Ollama or model downloads needed
python scripts/verify_backend.py               # F1–F5 + API checks (backend + Ollama running)
python scripts/verify_backend.py --skip-api    # in-process checks only
python scripts/compare_retrievers.py           # naive dense vs BM25 vs hybrid vs rerank vs floor
python scripts/evaluate.py --no-judge          # add --limit 3 for a quick run; drop --no-judge for faithfulness
```
The scripts build a fresh in-memory corpus from `data/papers/` (DejaVu1–3) on every run; pass `--docs FOLDER`
for other papers. Question sets: `data/eval/golden_qa.json` (5 grounded + 2 must-abstain) and
`data/eval/retrieval_qa.json` (10 exact-term + 10 paraphrase).

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, Ollama reachability/model, limits, live chats |
| `POST /sessions` | Start a new empty chat → `{"session_id": ...}` |
| `GET /sessions/{id}` | That chat's documents and chunk count |
| `DELETE /sessions/{id}` | End the chat and forget its PDFs |
| `POST /sessions/{id}/ingest` | Multipart field `files`, 1–10 PDFs; per-file `ingested` / `skipped` / `failed` |
| `GET /sessions/{id}/documents` | Documents in that chat |
| `POST /sessions/{id}/ask` `{"question"}` | Answer, sources (`cited`, `verified`, `match_score`), `confidence`, `abstained` |
| `POST /sessions/{id}/synthesize` `{"topic"}` | Comparison table + research gaps across that chat's papers |

Errors are JSON `{"detail": "..."}` with a friendly, actionable message (INT-2): 400 bad input / too many
files, 404 chat ended, 413 file too large, 422 unreadable PDF, 502 LLM error reply, 503 Ollama down or
model missing. Interactive docs: http://127.0.0.1:8000/docs

## Configuration
All constants live in `app/config.py`; each can be overridden by an environment variable of the same name
(e.g. `OLLAMA_MODEL`, `ABSTAIN_THRESHOLD`, `RELEVANCE_FLOOR`, `MAX_FILES_PER_REQUEST`, `SESSION_TTL_MIN`).

## Project layout
```
app/
  api/main.py              FastAPI endpoints, upload validation, error mapping
  config.py                every tunable constant (env-overridable)
  core/                    dataclasses + error types
  embeddings/embedder.py   MiniLM wrapper (normalised vectors)
  generation/llm.py        BaseLLM, Ollama client, prompts, <think> stripping
  generation/verifier.py   F3 citation verification
  ingestion/               PDF loading (sections) + chunking (dehyphenation)
  pipelines/rag.py         ask / synthesize orchestration, F4 abstention + confidence
  pipelines/sessions.py    per-chat isolated corpora, expiry
  retrieval/retriever.py   F1 hybrid + RRF, F2 cross-encoder rerank + floor
  store/vector_store.py    in-memory FAISS store
data/papers/               sample corpus for the scripts
data/eval/                 question sets (+ results written by the scripts)
scripts/                   evaluate.py, compare_retrievers.py, verify_backend.py
tests/                     test_core.py (pipeline logic), test_api.py (sessions + upload limit)
ui/streamlit_app.py        basic chat UI
```

## Known limits
- Verification is lexical (fuzzy match / content-word recall + number guard); it catches unsupported claims
  and invented numbers but cannot judge logical entailment.
- `/synthesize` does not verify table cells (they mix several claims), so its sources show `verified: null`.
- If the cross-encoder cannot load, relevance falls back to dense cosine (`relevance_source: "dense-fallback"`);
  the 0.35 threshold was tuned for cross-encoder scores.
- Scanned PDFs (no text layer) are rejected; OCR is not included.
- Chats live in the backend's memory: restarting uvicorn ends every chat (by design).
