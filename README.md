# Thesora: Evidence-Grounded RAG Research Assistant

**Thesora lets you upload research papers (PDFs) and ask questions about them. Every answer cites the
exact passage it came from, each citation is checked against that passage, and when the papers don't
contain the answer, Thesora says so instead of guessing.**

It runs completely on your own computer: a local Ollama model (`qwen3:4b-instruct`), CPU only, no API
keys, no cloud. Built with Python 3.11, FastAPI and Streamlit, and tested on Windows.

---

## What Thesora does (Phase 1)

A basic Retrieval-Augmented Generation (RAG) system finds a few similar-looking chunks and lets the model write an answer: it misses exact terms, makes up unsupported claims, and answers confidently even when the papers say nothing. Thesora fixes this with five improvements and four extras. Click each to open it.

<details>
<summary><b>F1 · Hybrid retrieval</b></summary>

<br>

Searches by keywords (BM25, top 20) **and** by meaning (all-MiniLM-L6-v2 embeddings in FAISS, top 20), then merges both lists with Reciprocal Rank Fusion (k = 60) into 8 candidates. Exact terms such as "omni-moderation-latest" and reworded questions both work. Code: `app/retrieval/retriever.py`.

</details>

<details>
<summary><b>F2 · Cross-encoder reranking</b></summary>

<br>

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the 8 candidates; the best 5 are kept and any passage below a relevance floor of 0.1 is dropped, so junk never reaches the model. Code: `app/retrieval/retriever.py`.

</details>

<details>
<summary><b>F3 · Citation verification</b></summary>

<br>

Every `[n]` claim is checked against passage *n* with fuzzy matching and content-word recall; every number in a claim must appear in the passage (number guard). The cited paper's own name and reporting words ("reports", "according to") are not required to be in the passage. Sources show ✓ Verified or ! Unverified. Code: `app/generation/verifier.py`.

</details>

<details>
<summary><b>F4 · Abstention + confidence</b></summary>

<br>

If no passage is relevant enough (best relevance below 0.35) Thesora refuses **without calling the model**; it also refuses when the model says the context is insufficient. Every answer gets a confidence score (relevance, verified share and claim coverage). Code: `app/pipelines/rag.py`.

</details>

<details>
<summary><b>F5 · Evaluation harness</b></summary>

<br>

`scripts/evaluate.py` measures faithfulness, context precision and recall, answer relevancy, abstention accuracy and retrieval hit rate, and writes `results.json`. `scripts/compare_retrievers.py` compares naive dense search with every improvement.

</details>

<details>
<summary><b>Extra · Cross-paper synthesis</b></summary>

<br>

**Compare papers** mode builds a table with one row per paper plus a **Research Gaps** section. Each paper is searched separately and its passages are picked round-robin, so every uploaded paper gets a row (long papers can no longer crowd short ones out). Wide tables wrap inside the chat column.

</details>

<details>
<summary><b>Extra · Isolated chats and upload limit</b></summary>

<br>

Every chat starts empty and only sees its own PDFs (in memory, never saved to disk); any number of PDFs, up to 15 MB each and 200 MB per chat. Code: `app/pipelines/sessions.py`, `app/api/main.py`.

</details>

<details>
<summary><b>Extra · Prompt refiner (question suggestions)</b></summary>

<br>

Optional **✨ Prompt refiner**: rewrites a rough question into a precise one that retrieves better; you choose whether to send it. Adapted from prompt-master. Code: `app/generation/prompt_refiner.py`.

</details>

<details>
<summary><b>Extra · Research-grade interface</b></summary>

<br>

A calm "paper" light theme with a one-click dark theme, citation badges, a compact citations pop-up and a confidence meter. Code: `ui/streamlit_app.py`, `ui/styles.py`.

</details>

**Questions that name a paper** ("Which embedding model does PaperQA use?") are searched inside that paper only, with a wider pool, and every passage keeps its best reranker score with and without the paper's name: inside one paper the name no longer tells passages apart, it only favours passages that repeat it. Paper names come from the title (the part before ":"), the first author and system names the paper repeats ("OpenScholar", "LitQA"). Paper-specific abbreviations are spelled out for the reranker too ("LFS (Logic Fidelity Score)"). For questions asking for numbers, the best clearly relevant table keeps the last of the 5 slots. Measured on the six survey papers with `scripts/survey_check.py`: 14 of 14 test questions get the answer passage (a harder paraphrase whose answer the PDF splits across two columns still abstains), both out-of-scope questions abstain, Compare covers 6 of 6 papers.

**Clean extraction** (`app/ingestion/clean_layout.py`): only a paper's content is indexed. Running headers and footers, page numbers, the sideways arXiv stamp, the page-1 author / affiliation / date block, e-mail and URL footnotes, tiny figure labels, superscript citation markers and the reference list are dropped; table cells are left to their own table passage; boxed prompts and examples are kept. Two-column pages are read column by column (whole left column, then right). On the six survey papers this left 1,037 passages instead of 1,243, and every passage except one code line and one formula can be highlighted on its page. `PDF_CLEAN=0` turns it off.

**Citation panel**: 📄 Open page shows the page as a sharp image (216 dpi) with the sentence that supports the answer highlighted (not the whole retrieved passage); pinch or Ctrl + scroll zooms, drag moves, and the toolbar has −, +, Fit and Go to highlight. It runs in the browser with no external scripts. `PDF_PANEL_MODE=viewer` switches back to the pdf.js viewer.

**Pipeline:** PDF → PyMuPDF text + section detection → chunks (512 characters, 100 overlap) → all-MiniLM-L6-v2 embeddings → FAISS + BM25 → RRF fusion → cross-encoder rerank → Ollama → citation verification → FastAPI → Streamlit.

---

## Contributors

| Contributor | Contribution |
|---|---|
| **Praveen Raj S** | Hybrid retrieval (F1); abstention + confidence (F4); prompt refiner; v2: citation labels with BibTeX and notes export (N4), per-sentence check and silent papers (N5), partial answers (N6), large-file handling (N7) |
| **ReddyMasi Manasa** | Cross-encoder reranking (F2); citation verification (F3); evaluation harness (F5); v2: PDF tables (N1), search upgrades (N2), table-cell verification (N3), meaning check (N8), self-repair (N10), Agree? contradiction finder (N12), stress-test evaluation (N13) |
| **Pramod V** | Documentation: literature survey of 23 papers, research-gap analysis (the Research Gaps section), feature innovation for v2, README and reports |
| **E Akhila Reddy** | Tech-stack identification, environment setup, architecture design and system design (diagrams) |
| **Kushal U G** | User interface (light/dark themes, compact v2 layout, PDF side panel N9, Explain panel and glossary N11) and testing (unit tests, backend checks, browser tests) |

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

Expected: `Ran 68 tests ... OK`. These tests need neither Ollama nor any model downloads.

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

Optional full backend check (in terminal 2, while the backend runs). Expected: `PASS=17 FAIL=0`:
```bash
python scripts/verify_backend.py
```

</details>

---

## How to use it

<details>
<summary><b>Upload papers and ask questions</b></summary>

<br>

1. In the left sidebar, check for the green **"Ready · qwen3:4b-instruct"** badge.
2. Under **Library**, click **Upload** (or drag PDFs onto the box) and choose your PDFs (any number, up to 15 MB each and 200 MB per chat), then click
   **Ingest papers**. Each paper then appears under **In this chat** with its pages and passages.
3. Keep **Ask a question** selected, type your question in the box at the bottom and press Enter.
4. Read the answer card:
   - The small numbered badges are citations. An amber badge means that claim could not be verified.
   - The **Confidence** meter is green (high), amber (medium) or red (low).
   - Each source card shows the paper, page and section, with **✓ Verified** or **! Unverified**, and a
     relevance bar. Click a card to read the exact passage.
   - A **"Not answered from your papers"** notice means the papers don't contain the answer.
5. To compare papers, select **Compare papers** and type a topic, for example
   `memory errors and false familiarity`. You get one table row per paper and a **Research Gaps** section.
6. Click **＋ New chat** to start over with no documents. Refreshing the page does the same.

</details>

<details>
<summary><b>Prompt refiner and dark theme</b></summary>

<br>

**Prompt refiner (optional).** Turn on **✨ Prompt refiner** above the conversation. Type a rough
question, for example `deepera reranking how good`, and press **✨ Refine**. Thesora suggests a precise version,
such as *"How much does DeepEra improve over other rerankers?"*,
with a one-line reason. Then choose:

- **Use suggestion** sends the refined question.
- **Edit it** puts the suggestion in the box so you can change it first.
- **Send my original** sends your text unchanged.

In **Compare papers** mode it refines a comparison topic instead of a question. It never answers the
question itself and never sends anything without your click. It takes about 5–10 seconds on a CPU.

**Dark theme.** Use the **🌙 Dark** switch at the top right. The choice is saved in the page address
(`?theme=dark`), so it survives a refresh. To make dark the default, set `UI_DEFAULT_THEME=dark`.

</details>

<details>
<summary><b>Try it with the sample papers (questions and expected answers)</b></summary>

<br>

Upload the six papers in `data/papers/` (the RAG-for-science papers from our literature survey: Ai2 Scholar QA,
DeepEra, HySemRAG, the Agentic Hybrid RAG framework, OpenScholar and PaperQA), then ask:

| Feature | Question | Expected result |
|---|---|---|
| F1 exact keyword | `What is omni-moderation-latest used for?` | Query validation: screening harmful queries (Ai2 Scholar QA) |
| F1 meaning search | `Which moderation model screens user queries in Ai2 Scholar QA?` | omni-moderation-latest |
| F1 | `What value of K is used in Reciprocal Rank Fusion?` | K = 60 (HySemRAG), the same constant Thesora uses |
| F2 rerank | `Which embedding model does PaperQA use?` | text-embedding-ada-002; source [1] is PaperQA, not a paper with a similar sentence |
| F2 rerank | `What relevance threshold does DeepEra use to filter passages?` | τ = 0.8 (DeepEra) |
| F3 citations | `How often does GPT-4o hallucinate citations according to OpenScholar?` | 78–90% of the time, ✓ verified |
| F3 citations | `What citation accuracy and single-pass success rate does HySemRAG report?` | 99.0% and 68.3%, ✓ verified |
| N1 tables | `What accuracy and precision did PaperQA get on LitQA?` | 69.5% and 87.9%, from Table 2 |
| N6 parts | `What chunk size does PaperQA use? What is the price of an NVIDIA H100 GPU?` | Part 1: 4,000-character chunks; part 2: "Not found in your papers" |
| N11 glossary | `How is LFS computed?` | Logic Fidelity Score: an LLM scores answers from 0 to 5 (DeepEra) |
| F4 abstain | `What is the price of an NVIDIA H100 GPU?` | **Abstained** (DeepEra mentions H100 GPUs, never a price) |
| F4 abstain | `How do these systems handle speech or audio input?` | **Abstained** |
| Synthesis | (Compare mode) `Compare how these papers retrieve and rerank evidence` | One row per paper (6 rows) + Research Gaps |
| Agree? | (Agree? mode) `citation accuracy`, then `cost per question` | 4 papers quoted; then "Only one paper discusses this topic" (PaperQA, $0.18) |
| New chat | Click **New chat**, then ask `What is LitQA?` | No documents loaded in this chat |
| Upload limit | Select a PDF over 15 MB | Rejected: "file exceeds the 15 MB limit"; more than 200 MB in one chat is refused too |
| Prompt refiner | Turn on **✨ Prompt refiner**, type `deepera reranking how good`, press **✨ Refine** | A precise question about DeepEra's reranking results; **Use suggestion** answers it |

The full walk-through with reasons for every question is in `TEST_PLAN_SURVEY_PAPERS.md`.

If a longer answer shows **⚠️ unverified** on a citation, that's F3 working: the model reworded or combined
claims more than the cited passage directly supports, and Thesora flags it rather than hiding it.

</details>

<details>
<summary><b>Run the evaluation and retrieval comparison (F5)</b></summary>

<br>

```bash
python scripts/evaluate.py
```
Answers the 12 questions in `data/eval/golden_qa.json` (10 answerable and 2 that must be refused), grades
them with the local model and writes `data/eval/results.json`. Add `--no-judge` to skip the model grading
(faster), or `--limit 3` for a quick run.

```bash
python scripts/compare_retrievers.py
```
Compares dense-only, BM25-only, hybrid, hybrid + rerank, and hybrid + rerank + floor on 20 questions
(10 exact-term and 10 reworded), and writes `data/eval/retriever_comparison.json`.

Both scripts build a fresh corpus from `data/papers/`. Pass `--docs FOLDER` to use other papers.

**Answer quality** (v1, Windows laptop, CPU, measured on the earlier three DejaVu sample papers; re-run
`python scripts/evaluate.py` to measure the survey papers):

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Context precision | 1.00 |
| Context recall | 1.00 |
| Answer relevancy | 0.80 |
| Abstention accuracy | 1.00 |
| Retrieval hit rate | 1.00 |

**Retrieval** (the six survey papers, 20 questions: 10 exact-term, 10 reworded; hit = the passage holding the
answer is among the 5 passages the model sees):

| Retriever | Hit rate | MRR | Exact-term hit | Reworded hit |
|---|---|---|---|---|
| Dense only (naive RAG) | 0.60 | 0.24 | 0.70 | 0.50 |
| BM25 only | 0.65 | 0.59 | 0.80 | 0.50 |
| Hybrid (RRF) | 0.70 | 0.53 | 0.80 | 0.60 |
| Hybrid + rerank | 0.70 | 0.68 | 0.80 | 0.60 |
| Hybrid + rerank + floor (v1) | 0.70 | 0.68 | 0.80 | 0.60 |
| **Thesora v2 (full)** | **0.85** | **0.78** | **0.90** | **0.80** |

v2 adds paper-scoped search for questions that name a paper, abbreviations spelled out for the reranker,
the table slot for number questions and context expansion of the 2 best passages.

On these papers the relevance floor removes the 25% of passages that score below 0.1 (the context averages 3.8
passages instead of 5) without losing any answer passage.

</details>

---

## Chat rules

- **Any number of PDFs per upload**, each at most **15 MB** (`MAX_UPLOAD_MB`), and at most **200 MB** of PDFs per
  chat (`MAX_SESSION_PDF_MB`); a too-large file is reported as failed while the others are still read.
  `MAX_FILES_PER_REQUEST` (0 = no limit) can cap the number of files per upload.
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
| `POST /sessions/{id}/ingest` | Multipart field `files`, any number of PDFs (15 MB each, 200 MB per chat); per-file `ingested` / `skipped` / `failed` |
| `GET /sessions/{id}/documents` | Documents in that chat |
| `POST /sessions/{id}/ask` `{"question"}` | Answer, sources (`cited`, `verified`, `match_score`), `confidence`, `abstained` |
| `POST /sessions/{id}/synthesize` `{"topic"}` | Comparison table + research gaps across that chat's papers |
| `POST /sessions/{id}/refine` `{"text", "mode"}` | Optional: a sharper question (`mode: "ask"`) or comparison topic (`"compare"`), with a one-line reason |

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
`THESORA_API_URL` (where the UI finds the backend), `THESORA_OFFLINE`, `UI_DEFAULT_THEME` (`light` or
`dark`), `REFINE_MAX_WORDS`.

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
  generation/prompt_refiner.py  optional question refiner (adapted from prompt-master)
  ingestion/               PDF loading (sections) + chunking
  pipelines/rag.py         ask / synthesize, F4 abstention + confidence
  pipelines/sessions.py    isolated per-chat indexes, expiry
  retrieval/retriever.py   F1 hybrid + RRF, F2 rerank + floor
  store/vector_store.py    in-memory FAISS store
data/papers/               sample papers (the six survey papers on RAG for science)
data/eval/                 question sets + script results
scripts/                   evaluate.py, compare_retrievers.py, verify_backend.py
tests/                     test_core.py (pipeline), test_api.py (sessions, upload limit, errors),
                           test_refiner.py (prompt refiner)
ui/streamlit_app.py        Streamlit chat interface
ui/styles.py               light and dark themes (all colours and CSS)
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

- Citation checking is word-based by default. The meaning check (N8) judges logical support but is off by
  default because its smallest model is 283 MB; turn it on with `NLI_ENABLED=1`.
- Questions that share no key terms with the papers can be refused even when the answer is there; the prompt
  refiner (or `MULTI_QUERY=1`) usually fixes it.
- Look-alike passages from another paper can still be cited (stress test: citation precision 0.50), and a
  passage that is itself wrong is followed (Thesora cannot know the paper is wrong).
- Paper details (authors, year) are auto-detected and can be wrong; correct them through the API.
- If the reranker can't load, relevance falls back to embedding similarity, and the 0.35 threshold was
  tuned for reranker scores.
- Scanned PDFs (no text layer) are rejected; OCR isn't included.
- Chats live in the backend's memory, so restarting the backend ends every chat (by design).
- Very broad questions ("summarize every paper") may abstain in question mode; use **Compare papers** instead.

---

## Thesora v2: 13 new features

Click a feature to open it. ✅ done · ⚠️ done with a limit or switched off by default.

<details>
<summary><b>Using the v2 interface (where everything is)</b></summary>

<br>

- **Mode bar** (Ask a question · Compare papers · Agree? · ✨ Prompt refiner) stays pinned at the top while you scroll.
- **Under every answer** one row of four buttons keeps the card small:
  - **📚 Citations (n)** opens a pop-up with the confidence bar and every retrieved passage: label, ✓ Verified,
    rel score, a short preview with the supporting sentence marked, and its own **📄 Open** and **💡 Explain**.
    The **Sentence check** and **Table check** (comparisons) are at the bottom of the pop-up.
  - **📄 Open top source** and **💡 Explain** act on the passage the answer leans on most.
  - **✨ Suggested questions** shows three questions to ask next, written from your own papers, your question and
    what you asked before; one click asks it. The model writes six candidates and only those whose best passage
    clears a relevance bar (`SUGGEST_MIN_RELEVANCE=0.5`, checked by retrieval, no extra model call) are shown,
    best-supported first; the Explain panel's **Ask next** questions are filtered the same way.
- **Citation numbers**: each number after a sentence is the passage it came from (the n-th entry in 📚 Citations);
  several numbers mean the sentence combines several passages, often from different papers. Hover a number to
  see its paper and page.
- **When Thesora refuses**, it shows "Closest thing your papers say (not an answer)": the sentence of the nearest
  passage that shares most with the question, quoted with its source, only when that passage is at least loosely
  related. A refused follow-up in the Explain panel also offers **Try instead** questions.
- **Side panel**: **Citation** shows the page with the answer's supporting sentence highlighted (zoomable);
  **Explain** gives a short plain-English explanation and three **Ask next** questions, answered with citations
  in the panel, plus your own follow-up box with **✨ Refine**. Both tabs keep their content; **✕ Close panel**
  hides it.
- Small notes under an answer: part chips for multi-part questions, "N claims removed" after self-repair, and
  "No evidence from … — silence is not disagreement".
- **✨ Prompt refiner** (mode bar): type a rough question, press **✨ Refine**, then **Use suggestion**,
  **Edit it** or **Send my original**.
- Sidebar: uploaded files and this chat's papers show three at a time (scroll for more); **Glossary** lists the
  most used abbreviations first (five visible, scroll for more); **Download .bib**, **Download notes (.md)**.

</details>

<details>
<summary><b>N1 · Tables in PDFs become citable passages</b> — ✅</summary>

<br>

Each table becomes its own passage (`chunk_type="table"`, markdown rows, headed by its "Table N: …" caption) with page and section, so it can be searched and cited. `scripts/check_extraction.py` reports two-column pages and tables (Ai2 Scholar QA: two columns on 9 of 11 pages, 3 tables). Two-column pages are read column by column (see Clean extraction). Tightly justified two-column lines no longer come out glued ("Priortoprocessingaquery"): word spaces are rebuilt from the gaps between letters (`PDF_SPACE_GAP_EM=0.08`), and the PDF highlight matches letter by letter when the PDF itself has no space characters. Setting: `PDF_LAYOUT_MODE=auto` (or `plain` for text only). Scanned PDFs are still rejected.

</details>

<details>
<summary><b>N2 · Search upgrades (three toggles)</b> — ⚠️ all off by default</summary>

<br>

(a) Contextual headers: "Title › Section" added to the search text only — **off**: measured worse for hybrid search (0.80 → 0.70). `CONTEXT_HEADERS=1`. (b) Multi-query: one short model call writes 2 reworded questions, results merged — off (extra call). `MULTI_QUERY=1`. (c) Context expansion — **on** for the 2 best passages: the model sees each one plus the next (then previous) passage of the same section, and verification uses the same text, so an answer cut at a chunk boundary is not lost. Limited to the top 2 to keep prompts short on a CPU (+58% context instead of +116%). `CONTEXT_EXPANSION=0` turns it off, `CONTEXT_EXPANSION_TOP` changes how many.

</details>

<details>
<summary><b>N3 · Comparison-table cells are verified</b> — ✅</summary>

<br>

Each Compare cell is split into claims and checked against that paper's own passages: ✓ Supported, ~ Partly supported, ✗ Not found in text, – Not stated, with the evidence page. See **Table check** under a comparison.

</details>

<details>
<summary><b>N4 · Real citations, BibTeX and notes export</b> — ✅ (edit form: API only)</summary>

<br>

Title, authors, year and DOI are detected at upload (marked auto-detected). Rotated arXiv stamps and running headers such as "A PREPRINT" are skipped; a title that starts after a logo gets the file name in front; the year comes from the arXiv number, then the page-1 header (journal line, conference date, ©, preprint date). Sources show "Smith et al., 2023 · p.4 · Results". Sidebar: **Download .bib** and **Download notes (.md)** (verified citations only, no model call). Correct details with `PATCH /sessions/{id}/documents/{doc_id}/metadata`.

</details>

<details>
<summary><b>N5 · Per-sentence check and silent papers</b> — ✅</summary>

<br>

Every answer sentence is ✓ Verified, ~ Weak, ! Unverified or ○ Uncited (icon + text). Footer: "No evidence from: Paper X — silence is not disagreement". See **Sentence check** under an answer.

</details>

<details>
<summary><b>N6 · Partial answers</b> — ✅</summary>

<br>

Questions with several parts (`?`, `;`, "as well as", "and also"; each part 4+ words; no model call to split) are answered part by part; only unsupported parts say "Not found in your papers". `PARTIAL_ANSWERS=1`.

</details>

<details>
<summary><b>N7 · Large files</b> — ✅</summary>

<br>

Up to 15 MB per PDF (`MAX_UPLOAD_MB`), a warning above 5 MB (`LARGE_FILE_WARN_MB`), and at most 200 MB of PDFs per chat (`MAX_SESSION_PDF_MB`), any number of files with a friendly error. Still 10 PDFs per upload.

</details>

<details>
<summary><b>N8 · Meaning check (NLI)</b> — ⚠️ built, off</summary>

<br>

A local NLI model checks whether the cited passage really supports each claim (entailed / weak / contradicted; contradiction always fails). Off because the smallest suitable model is 283 MB, over the 200 MB budget. Turn on with `NLI_ENABLED=1` (downloads `cross-encoder/nli-deberta-v3-xsmall` once). Without it, checking is by wording only and the UI says so.

</details>

<details>
<summary><b>N9 · PDF side panel with highlight</b> — ✅</summary>

<br>

Click **📄 [n] p.X** under an answer: the page splits 70/30 and the cited page opens on the right with the passage outlined. PDFs are served from memory only. If the viewer clips the page in a narrow panel, use `PDF_PANEL_MODE=image` (a highlighted page image that always fits; also the automatic fallback).

</details>

<details>
<summary><b>N10 · Self-repair</b> — ✅</summary>

<br>

If a sentence fails the check, the model gets one stricter retry; anything still unsupported is removed and the answer says "N claims removed". Never more than one retry. `SELF_REPAIR=1`.

</details>

<details>
<summary><b>N11 · Explain and glossary</b> — ✅ (text box, not mouse selection)</summary>

<br>

**💡 Explain** gives a short plain-English explanation of an answer or a passage, from that answer's passages only and labelled "Plain-English explanation — not evidence", then three **Ask next** follow-up questions; clicking one answers it through the normal cited pipeline (it abstains when the papers do not say). Citation and Explain keep their content when you switch tabs. The sidebar **Glossary** lists abbreviations found in the papers (scispaCy's Schwartz–Hearst detector, no model download); their long forms are added to keyword search. Each term appears once per chat ("LLM — in 4 papers"), and "LLMs" counts as "LLM".

</details>

<details>
<summary><b>N12 · "Agree?" mode</b> — ✅ (numbers only while N8 is off)</summary>

<br>

A third mode: for a topic, each paper's position is a sentence quoted word for word (no model call), with page and section. Conflicts: ⚠️ Numbers differ (severity 2), ⚠️ Contradiction (severity 3, needs N8), ~ Wording differs (severity 1: not a disagreement, listed only when **Also list pairs that are only worded differently** is switched on). Each paper's position is the sentence the cross-encoder scores highest for the topic (captions and code lines skipped), and a paper is quoted only if that sentence scores at least 0.3 (`AGREE_MIN_RELEVANCE`). Enter one topic at a time. If only one paper discusses the topic, Thesora says so instead of reporting "no conflicts".

</details>

<details>
<summary><b>N13 · Stress test</b> — ✅</summary>

<br>

`python scripts/stress_eval.py [--limit N] [--no-llm]` mixes in unrelated passages, look-alike passages and passages with altered numbers, and reports citation recall, citation precision, distractor-cited rate and altered-numbers-repeated rate (results below).

</details>

<details>
<summary><b>New API endpoints and response fields</b></summary>

<br>

New endpoints (all additive): `GET /sessions/{id}/metadata`, `PATCH /sessions/{id}/documents/{doc_id}/metadata`,
`GET /sessions/{id}/bibtex`, `GET /sessions/{id}/export`, `GET /sessions/{id}/documents/{doc_id}/pdf`,
`GET /sessions/{id}/documents/{doc_id}/page/{n}.png`, `POST /sessions/{id}/highlight`, `POST /sessions/{id}/explain`,
`GET /sessions/{id}/glossary`, `POST /sessions/{id}/agree`. `/ask` adds `sentences`, `silent_papers`, `repair`, `parts`,
`answer_id`, `nli_available`; `/synthesize` adds `table_cells`; sources add `label` and `meaning`.

</details>

<details>
<summary><b>Measured results (retrieval comparison and stress test)</b></summary>

<br>

**N2 retrieval comparison** (earlier DejaVu sample papers, 20 questions, `compare_retrievers.py`, hit rate / MRR;
the current survey-paper numbers are in *Run the evaluation and retrieval comparison* above):

| Mode | v1 | v2 (tables) | v2 + contextual headers |
|---|---|---|---|
| Dense only | 0.65 / 0.43 | 0.65 / 0.43 | 0.70 / 0.54 |
| Hybrid (RRF) | 0.75 / 0.64 | 0.80 / 0.65 | 0.70 / 0.60 |
| Hybrid + rerank + floor (Thesora) | 0.70 / 0.65 | 0.70 / 0.65 | 0.70 / 0.65 |

**N13 stress test** (`stress_eval.py --limit 3`, qwen3:4b-instruct, NLI off, earlier DejaVu sample papers; the
stress cases now use the survey-paper golden questions):

| Group | Citation recall | Citation precision | Distractor cited | Altered numbers repeated | Distractor ranked first |
|---|---|---|---|---|---|
| Noisy | 0.75 | 1.00 | 0.00 | – | 0.00 |
| Look-alike (SSLI) | 0.67 | 0.50 | 0.33 | – | 0.00 |
| Counterfactual | 0.86 | 0.83 | – | 0.67 | – |

Repeating altered numbers means the model follows the passage it is given; Thesora cannot detect a passage that is
itself wrong.

</details>

---

## Acknowledgements

Repositories and libraries Thesora uses or learned from. Licences are those stated by each project.

| Project | How Thesora uses it | Licence |
|---|---|---|
| [prompt-master](https://github.com/nidhinjs/prompt-master) | Rules adapted for the optional prompt refiner | MIT |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) / [PyMuPDF4LLM](https://github.com/pymupdf/pymupdf4llm) | PDF text, sections, tables (N1), highlights and page images (N9) | AGPL-3.0 |
| [streamlit-pdf-viewer](https://github.com/lfoppiano/streamlit-pdf-viewer) | PDF side panel with highlighted passages (N9) | see repository |
| [scispaCy](https://github.com/allenai/scispacy) via [unv-scispacy-abbr](https://pypi.org/project/unv-scispacy-abbr/) | Abbreviation glossary (N11) | Apache-2.0 |
| [spaCy](https://github.com/explosion/spaCy) | Blank English pipeline for the glossary | MIT |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Embeddings, cross-encoder reranking, NLI meaning check (N8) | Apache-2.0 |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector search | MIT |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25) | Keyword search | Apache-2.0 |
| [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) | Citation verification | MIT |
| [LangChain text splitters](https://github.com/langchain-ai/langchain) | Chunking | MIT |
| [Ollama](https://github.com/ollama/ollama) + [Qwen3](https://github.com/QwenLM/Qwen3) | Local language model (qwen3:4b-instruct) | MIT / Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) and [Streamlit](https://github.com/streamlit/streamlit) | Backend API and web interface | MIT / Apache-2.0 |
| [PaperQA2](https://github.com/Future-House/paper-qa) | Idea only: grading contradictions by severity (N12) | Apache-2.0 |
| [ALCE](https://github.com/princeton-nlp/ALCE) | Idea only: citation recall and precision metrics (N13) | MIT |
| [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | Idea only: contextual chunk headers (N2a); no code copied | see repository |
