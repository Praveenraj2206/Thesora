# Thesora v2 — N6 partial answers (apply on top of Batch A + N7)
Extract over C:\dev\Thesora, then: python -m unittest discover -s tests -v   (expect 54 OK)
N6: multi-part questions (split on "?", ";", " as well as ", " and also "; each part >= 4 words, no LLM call)
are answered part by part; unsupported parts say "Not found in your papers". /ask adds "parts"
[{question, answered, abstained}]; citations are renumbered across parts. PARTIAL_ANSWERS=1 (set 0 to disable).
UI shows "✓ Answered" / "✗ Not found in your papers" per part. Single-part questions behave as before.
