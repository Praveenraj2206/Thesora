# Thesora v2 — Batch A (backend trust) — how to apply (Windows Git Bash)

Files: app/config.py, app/core/models.py, app/pipelines/rag.py, app/api/main.py,
app/generation/trust.py (new), ui/streamlit_app.py, tests/test_core.py, tests/test_api.py,
tests/test_trust.py (new), .gitignore. No files deleted. No new pip packages.

1. Stop the backend and UI (Ctrl+C).
2. Extract this zip over the repo root:
   cd /c/dev/Thesora && powershell -Command "Expand-Archive -Force -Path \"$HOME/Downloads/thesora_v2_batchA.zip\" -DestinationPath 'C:/dev/Thesora'"
3. source .venv/Scripts/activate && python -m unittest discover -s tests -v     # expect 53 tests OK
4. Start backend + UI as usual.

What's new (all additive; old fields unchanged):
- N5 /ask returns "sentences" [{text, citations, status verified|weak|unverified|uncited, score, meaning}]
  and "silent_papers"; UI: "Sentence check" expander + "No evidence from: … — silence is not disagreement".
- N10 self-repair (SELF_REPAIR=1): one stricter retry, then unverified sentences are removed;
  /ask returns "repair" {attempted, removed, before_unverified, after_unverified}; UI note.
- N3 /synthesize returns "table_cells" [{paper, column, status supported|partly|not_found|not_stated, evidence}];
  UI: "Table check" expander.
- N4 metadata per paper (title/authors/year/DOI, auto-detected): GET /sessions/{id}/metadata,
  PATCH /sessions/{id}/documents/{doc_id}/metadata, GET /sessions/{id}/bibtex, GET /sessions/{id}/export;
  source cards show "Smith et al., 2023 · p.4 · Section"; sidebar download buttons.
- N8 NLI meaning check: code ready, OFF by default (NLI_ENABLED=0). Smallest candidate models are
  283 MB (nli-deberta-v3-xsmall) and 329 MB (nli-MiniLM2-L6-H768) — over the 200 MB budget.
  To enable: export NLI_ENABLED=1 (downloads cross-encoder/nli-deberta-v3-xsmall once).
Not done in batch A: PATCH-metadata edit form in the UI (API works), NLI threshold calibration.
