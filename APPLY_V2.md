# Thesora v2 — Batch A + N7 (apply over the repo root, Windows Git Bash)
1. Stop backend and UI (Ctrl+C).
2. cd /c/dev/Thesora && powershell -Command "Expand-Archive -Force -Path \"$HOME/Downloads/thesora_v2_A_N7.zip\" -DestinationPath 'C:/dev/Thesora'"
3. source .venv/Scripts/activate && python -m unittest discover -s tests -v   # expect 53 tests OK
4. Restart backend and UI.
Batch A: see APPLY_BATCH_A.md. N7: upload limit 100 MB per file (config, API, .streamlit/config.toml, README);
UI warns for files over 5 MB (LARGE_FILE_WARN_MB); a chat holds at most 400 MB of PDFs (MAX_SESSION_PDF_MB) —
extra files get a friendly error. Not done: compressed viewer copy (belongs with N9 viewer).
Still to build: N1, N2, N6 (Batch B) and N9, N11, N12, N13 (Batch C).
