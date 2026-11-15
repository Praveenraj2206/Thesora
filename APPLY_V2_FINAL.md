# Thesora v2 — N1, N2, N12, N13 (final patch; apply on top of A + N7 + N6 + N9/N11)
1. Stop backend and UI (Ctrl+C).
2. cd /c/dev/Thesora && powershell -Command "Expand-Archive -Force -Path 'C:\Users\NAVEEN RAJ\OneDrive\Desktop\PRAVEEN RAJ S\py_bro\thesora_v2_final.zip' -DestinationPath 'C:\dev\Thesora'"
3. source .venv/Scripts/activate && pip install --timeout 120 --retries 10 -r requirements.txt
4. python -m unittest discover -s tests -v          # expect 68 tests OK (about 1 minute)
5. python scripts/check_extraction.py               # N1 report: DejaVu1 two-column + 2 tables
6. python scripts/stress_eval.py --no-llm           # N13 retrieval-level (fast); drop --no-llm with Ollama running
7. Restart backend + UI; try mode "Agree?" with: reasoning levels and memorisation
8. git add -A && git commit -m "v2: N1 N2 N12 N13"   (local only)
See README "Thesora v2 additions" for every feature, setting, endpoint and the measured tables.
