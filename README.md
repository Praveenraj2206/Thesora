# Thesora: Evidence-Grounded RAG Research Assistant

**Thesora lets you upload research papers (PDFs) and ask questions about them. Every answer cites the
exact passage it came from, each citation is checked against that passage, and when the papers don't
contain the answer, Thesora says so instead of guessing.**

It runs completely on your own computer: a local Ollama model (`qwen3:4b-instruct`), CPU only, no API
keys, no cloud. Built with Python 3.11, FastAPI and Streamlit, and tested on Windows.

---

## Why Thesora? (what's wrong with "naive" RAG)

A basic Retrieval-Augmented Generation (RAG) system finds a few similar-looking chunks and lets the model
write an answer. That causes three problems: it misses passages that use exact terms, it makes up claims
the papers don't support, and it answers confidently even when the papers say nothing about the question.
Thesora adds five improvements and two extras to fix this:

| # | Feature | What it does for you | Where |
|---|---|---|---|
| F1 | **Hybrid retrieval** | Searches by keywords (BM25) *and* by meaning (embeddings), then merges both lists with Reciprocal Rank Fusion (k=60), so exact terms like "5-4-3-2-1" and reworded questions both work | `app/retrieval/retriever.py` |
| F2 | **Cross-encoder reranking** | Re-scores the top candidates with `ms-marco-MiniLM-L-6-v2`, keeps the best 5 and drops passages below a relevance floor (0.1) | `app/retrieval/retriever.py` |
| F3 | **Citation verification** | Checks every `[n]` claim against passage *n* (fuzzy match + word recall + a number guard), and marks each source ✅ verified or ⚠️ unverified | `app/generation/verifier.py` |
| F4 | **Abstention + confidence** | If no passage is relevant enough (< 0.35), it refuses *without calling the LLM*, and every answer gets a confidence score | `app/pipelines/rag.py` |
| F5 | **Evaluation harness** | Measures faithfulness, context precision/recall, answer relevancy, abstention accuracy and retrieval hit rate, and writes `results.json` | `scripts/evaluate.py` |
| ★ | **Cross-paper synthesis** | Builds a comparison table with one row per paper, plus a "Research Gaps" section | `POST /sessions/{id}/synthesize` |
| ★ | **Isolated chats + upload limit** | Every chat starts empty and only sees its own PDFs; at most 10 PDFs per upload | `app/pipelines/sessions.py`, `app/api/main.py` |

**Pipeline:** PDF → PyMuPDF text + section detection → chunks (512 characters, 100 overlap) →
all-MiniLM-L6-v2 embeddings → FAISS + BM25 → RRF fusion → cross-encoder rerank → Ollama →
citation verification → FastAPI → Streamlit.

---

## Contributors

| Contributor | Contribution |
|---|---|
| **Praveen Raj S** | Citation verification (F3); the 10-PDF upload limit; isolated new-chat sessions |
| **ReddyMasi Manasa** | Major features: hybrid retrieval (F1), cross-encoder reranking (F2), abstention + confidence (F4), evaluation harness (F5), cross-paper synthesis |
| **Pramod V** | Literature survey; Streamlit user interface |

---

## Getting started

Click each step to open it.

<details>
<summary><b>Step 1: Install what you need (one time)</b></summary>

<br>

- **Python 3.11:** https://www.python.org/downloads/release/python-3119/ (tick *"Add Python to PATH"*)
- **Git for Windows** (includes Git Bash): https://git-scm.com/download/win
- **Ollama:** https://ollama.com/download

Use **Git Bash** for all the commands below. Keep the project **outside OneDrive** (for example `C:\dev`),
because OneDrive sync can delete files inside the virtual environment.

</details>

<details>
<summary><b>Step 2: Download the project</b></summary>

<br>

```bash
mkdir -p /c/dev && cd /c/dev
```
```bash
git clone https://github.com/Praveenraj2206/Thesora.git
```
```bash
cd Thesora
```

</details>

<details>
<summary><b>Step 3: Create the virtual environment and install packages</b></summary>

<br>

```bash
py -3.11 -m venv .venv
```
```bash
source .venv/Scripts/activate
```
```bash
pip install --timeout 120 --retries 10 -r requirements.txt
```

If the install stops with a `Read timed out` error, your connection was slow. Run the same `pip install`
command again; it continues where it stopped.

</details>

<details>
<summary><b>Step 4: Download the language model (skipped if you already have it)</b></summary>

<br>

```bash
ollama list | grep -q "qwen3:4b-instruct" && echo "model already installed" || ollama pull qwen3:4b-instruct
```

The model is about 2.5 GB. The embedding model and reranker (small) download from Hugging Face the first
time the backend starts. After that, `export THESORA_OFFLINE=1` makes it use only the local cache.

</details>

<details>
<summary><b>Step 5: Check the installation</b></summary>

<br>

```bash
python -m unittest discover -s tests -v
```

Expected: `Ran 36 tests ... OK`. These tests need neither Ollama nor any model downloads.

</details>

<details>
<summary><b>Step 6: Start the app (2 terminals)</b></summary>

<br>

Make sure the **Ollama app is running** first.

**Terminal 1: backend**
```bash
cd /c/dev/Thesora && source .venv/Scripts/activate
```
```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
Wait for `Application startup complete`. Don't add `--reload`: it crashes on Windows.

**Terminal 2: user interface**
```bash
cd /c/dev/Thesora && source .venv/Scripts/activate
```
```bash
streamlit run ui/streamlit_app.py
```
Your browser opens at **http://localhost:8501**.

Optional full backend check (in terminal 2, while the backend runs). Expected: `PASS=16 FAIL=0`:
```bash
python scripts/verify_backend.py
```

</details>

---

## How to use it

<details>
<summary><b>Upload papers and ask questions</b></summary>

<br>

1. In the left sidebar, check for the green **"Ollama ready (qwen3:4b-instruct)"** box.
2. Under **Upload PDFs**, click **Browse files** and choose up to 10 PDFs, then click **Ingest**.
   Each paper shows how many chunks were indexed, and it appears under **Documents in this chat**.
3. Keep **Mode** on **Ask a question**, type your question in the box at the bottom and press Enter.
4. Read the answer:
   - The `[1]`, `[2]` numbers are citations.
   - The **Confidence** bar shows how sure Thesora is.
   - Each source row shows the paper, page and section, with **✅ verified** or **⚠️ unverified**.
     Click a row to see the exact passage and its relevance score.
   - A yellow **Abstained** banner means the papers don't contain the answer.
5. To compare papers, switch **Mode** to **Compare papers (synthesis)** and type a topic, for example
   `memory errors and false familiarity`. You get one table row per paper and a **Research Gaps** section.
6. Click **➕ New chat** to start over with no documents. Refreshing the page does the same.

</details>

<details>
<summary><b>Try it with the sample papers (questions and expected answers)</b></summary>

<br>

Upload the three papers in `data/papers/` (DejaVu1, DejaVu2, DejaVu3), then ask:

| Feature | Question | Expected result |
|---|---|---|
| F1 exact keyword | `What is the 5-4-3-2-1 technique?` | A grounding exercise using the five senses (DejaVu1) |
| F1 exact keyword | `What is Dimension-W?` | The "ambient possibility field" (DejaVu3) |
| F1 reworded | `Which parts of the brain are linked to the feeling of having lived a moment before?` | The hippocampus and parietal lobes (DejaVu1) |
| F1 reworded | `How many AI systems and chemistry test sets were examined in the audit?` | 22 language models, 12 benchmarks (DejaVu2) |
| F2 rerank | `What causes the Mandela Effect in the strange attractor model?` | Top source [1] is DejaVu3: orphaned topological cache pointers |
| F3 citations | `How much more often were higher reasoning levels flagged for retrieval than the lowest level?` | "89% more often", source ✅ verified |
| F3 citations | `What are the three types of intuition identified by Feinberg?` | Guesswork, discovering intuition, fundamental intuition, ✅ verified |
| F4 abstain | `What is the capital of Australia and when was it founded?` | **Abstained**, confidence 0% |
| F4 abstain | `What GPU model was used to train the models in these papers?` | **Abstained** |
| Synthesis | (Compare mode) `memory errors and false familiarity` | A table row for each paper + Research Gaps |
| New chat | Click **New chat**, then ask `What is deja vu?` | Abstains: no documents loaded in this chat |
| Upload limit | Select 11 PDFs | Blocked: "at most 10 per upload" |

If a longer answer shows **⚠️ unverified** on a citation, that's F3 working: the model reworded or combined
claims more than the cited passage directly supports, and Thesora flags it rather than hiding it.

</details>

<details>
<summary><b>Run the evaluation and retrieval comparison (F5)</b></summary>

<br>

```bash
python scripts/evaluate.py
```
Answers the 7 questions in `data/eval/golden_qa.json` (5 answerable and 2 that must be refused), grades
them with the local model and writes `data/eval/results.json`. Add `--no-judge` to skip the model grading
(faster), or `--limit 3` for a quick run.

```bash
python scripts/compare_retrievers.py
```
Compares dense-only, BM25-only, hybrid, hybrid + rerank, and hybrid + rerank + floor on 20 questions
(10 exact-term and 10 reworded), and writes `data/eval/retriever_comparison.json`.

Both scripts build a fresh corpus from `data/papers/`. Pass `--docs FOLDER` to use other papers.

**Sample results** (Windows laptop, CPU, the 3 sample papers):

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Context precision | 1.00 |
| Context recall | 1.00 |
| Answer relevancy | 0.80 |
| Abstention accuracy | 1.00 |
| Retrieval hit rate | 1.00 |

| Retriever | Hit rate | MRR | Exact-term hit | Reworded hit |
|---|---|---|---|---|
| Dense only (naive RAG) | 0.65 | 0.43 | 0.80 | 0.50 |
| BM25 only | 0.75 | 0.66 | 1.00 | 0.50 |
| Hybrid (RRF) | 0.75 | 0.64 | 1.00 | 0.50 |
| Hybrid + rerank | 0.75 | 0.68 | 1.00 | 0.50 |
| Hybrid + rerank + floor | 0.70 | 0.65 | 1.00 | 0.40 |

The relevance floor cuts the context from 5 passages to about 2, removing the weak passages that most
often lead to unsupported claims, at a small cost in recall on reworded questions.

</details>

---

## Chat rules

- **At most 10 PDFs per upload** (`MAX_FILES_PER_REQUEST`). An 11-file request is rejected with HTTP 400
  before anything is read, and the UI blocks it too. Each file can be at most 50 MB (`MAX_UPLOAD_MB`).
- **Every chat starts empty.** Each chat has its own in-memory index. Opening the UI, refreshing the page
  or pressing **New chat** starts a new chat, so earlier PDFs and references are never used.
- **Nothing is saved to disk.** Uploads are read from a temporary folder that is deleted straight away.
  Ended chats are freed immediately, idle chats expire after 120 minutes (`SESSION_TTL_MIN`), and
  restarting the backend forgets every chat.

---

<details>
<summary><b>API reference</b></summary>

<br>

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, Ollama reachability and model, limits, number of live chats |
| `POST /sessions` | Start a new empty chat → `{"session_id": ...}` |
| `GET /sessions/{id}` | That chat's documents and chunk count |
| `DELETE /sessions/{id}` | End the chat and forget its PDFs |
| `POST /sessions/{id}/ingest` | Multipart field `files`, 1–10 PDFs; per-file `ingested` / `skipped` / `failed` |
| `GET /sessions/{id}/documents` | Documents in that chat |
| `POST /sessions/{id}/ask` `{"question"}` | Answer, sources (`cited`, `verified`, `match_score`), `confidence`, `abstained` |
| `POST /sessions/{id}/synthesize` `{"topic"}` | Comparison table + research gaps across that chat's papers |

Errors are JSON `{"detail": "..."}` with a friendly, actionable message: 400 bad input or too many files,
404 chat ended, 413 file too large, 422 unreadable PDF, 502 bad LLM reply, 503 Ollama down or model
missing. Interactive docs: http://127.0.0.1:8000/docs

</details>

<details>
<summary><b>Configuration</b></summary>

<br>

All settings live in `app/config.py`, and each can be overridden by an environment variable of the same
name, for example:

```bash
export OLLAMA_MODEL=qwen3:4b-instruct
```
```bash
export ABSTAIN_THRESHOLD=0.35
```

Other common ones: `RELEVANCE_FLOOR`, `MAX_FILES_PER_REQUEST`, `MAX_UPLOAD_MB`, `SESSION_TTL_MIN`,
`API_URL` (where the UI finds the backend), `THESORA_OFFLINE`.

</details>

<details>
<summary><b>Project layout</b></summary>

<br>

```
app/
  api/main.py              FastAPI endpoints, upload validation, friendly errors
  config.py                every setting (env-overridable)
  core/                    data classes + error types
  embeddings/embedder.py   MiniLM embeddings (normalised vectors)
  generation/llm.py        BaseLLM, Ollama client, prompts
  generation/verifier.py   F3 citation verification
  ingestion/               PDF loading (sections) + chunking
  pipelines/rag.py         ask / synthesize, F4 abstention + confidence
  pipelines/sessions.py    isolated per-chat indexes, expiry
  retrieval/retriever.py   F1 hybrid + RRF, F2 rerank + floor
  store/vector_store.py    in-memory FAISS store
data/papers/               sample papers (DejaVu1–3)
data/eval/                 question sets + script results
scripts/                   evaluate.py, compare_retrievers.py, verify_backend.py
tests/                     test_core.py (pipeline), test_api.py (sessions, upload limit, errors)
ui/streamlit_app.py        Streamlit chat interface
```

</details>

<details>
<summary><b>Troubleshooting</b></summary>

<br>

| Problem | Fix |
|---|---|
| `Read timed out` during `pip install` | Slow connection. Run the same `pip install --timeout 120 --retries 10 -r requirements.txt` again |
| `No module named ...` | You skipped the install, or the venv isn't active. Run `source .venv/Scripts/activate`, then install again |
| `No pyvenv.cfg file` | The venv is broken (often by OneDrive). Delete `.venv` and create it again (Step 3) |
| "Ollama is not reachable" in the sidebar | Start the Ollama app, or run `ollama serve` |
| "Model missing" | `ollama pull qwen3:4b-instruct` |
| "That chat had ended..." | The backend restarted. Upload your PDFs again |
| A PDF is rejected | Scanned PDFs (images without a text layer) aren't supported |

</details>

---

## Known limits

- Citation checking is word-based (fuzzy match, word recall and a number guard). It catches unsupported
  claims and invented numbers, but it can't judge logical reasoning.
- Synthesis table cells aren't verified, because each cell combines several claims.
- If the reranker can't load, relevance falls back to embedding similarity, and the 0.35 threshold was
  tuned for reranker scores.
- Scanned PDFs (no text layer) are rejected; OCR isn't included.
- Chats live in the backend's memory, so restarting the backend ends every chat (by design).
- Very broad questions ("summarize every paper") may abstain in question mode; use **Compare papers** instead.
