# Thesora v2 — N9 PDF side panel + N11 Explain & glossary (apply on top of Batch A + N7 + N6)

## Apply (Windows Git Bash)
1. Stop backend and UI (Ctrl+C).
2. cd /c/dev/Thesora && powershell -Command "Expand-Archive -Force -Path 'C:\Users\NAVEEN RAJ\OneDrive\Desktop\PRAVEEN RAJ S\py_bro\thesora_v2_N9_N11.zip' -DestinationPath 'C:\dev\Thesora'"
3. source .venv/Scripts/activate && pip install -r requirements.txt      # adds streamlit-pdf-viewer, unv-scispacy-abbr, spacy
4. python -m unittest discover -s tests -v                                 # expect 60 tests OK
5. Restart backend + UI, then: git add -A && git commit -m "v2: N9 + N11"   (local only)

## N9 — PDF side panel with highlight
- Under every answer: "📄 [n] p.X" buttons. Click → the page splits 70/30; right panel tabs "Citation" | "Explain".
- Citation tab: the cited page with the passage outlined (streamlit-pdf-viewer). Close with "✕ Close panel".
- Backend (memory only, never disk): GET /sessions/{id}/documents/{doc_id}/pdf,
  POST /sessions/{id}/highlight {doc_id, page, passage} → rectangles in PDF points (top-left origin, matches the viewer),
  GET /sessions/{id}/documents/{doc_id}/page/{n}.png (fallback image with highlight).
- Settings: VIEWER_WIDTH_PX=300, PDF_PANEL_MODE=viewer|image, HIGHLIGHT_COLOR, PAGE_PNG_ZOOM.
- Known limit: in the narrow panel the pdf.js viewer clips the right edge of the page slightly.
  export PDF_PANEL_MODE=image  → full page image that always fits (also used automatically if the viewer fails).
- Not done: lightly compressed viewer copy of each PDF (N7 extra) — original bytes are served.

## N11 — Explain + glossary
- "💡 Explain" under each answer opens the Explain tab: edit the text to just the part you want, optional follow-up,
  press Explain. Output: What / Why / How / Example, always labelled "Plain-English explanation — not evidence".
  Uses ONLY that answer's passages (stored per answer_id in memory). POST /sessions/{id}/explain
  {answer_id, selected_text, follow_up?, previous?}. /ask now also returns "answer_id".
- Glossary: abbreviations detected at upload with unv-scispacy-abbr on spacy.blank("en") (no model download).
  Sidebar "Glossary (N)" expander; GET /sessions/{id}/glossary. Known abbreviations in the selected text show
  "From the paper: X = …, p.N" first; abbreviations in a question add their long form to the BM25 query only.
  Sample papers: DejaVu1 has 1 (LSD = lysergic acid diethylamide); DejaVu2/3 define none.
- Not done: true mouse text-selection component (the editable text box + button is the reliable fallback).

## Tested here
60 unit/API tests pass. Browser test with Ollama: ask → "📄 [1] p.11" opens the page with the 89% passage
outlined; "💡 Explain" returned What/Why/How/Example with the not-evidence label; glossary shows 1 term.
