"""N9 PDF panel + N11 explain/glossary tests. No Ollama or model downloads (stub LLM)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NLI_ENABLED", "0")
os.environ.setdefault("SELF_REPAIR", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.generation import viewer  # noqa: E402
from app.ingestion.pdf_loader import load_pdf  # noqa: E402

PAPER = ROOT / "data" / "papers" / "DejaVu2.pdf"


class ViewerTests(unittest.TestCase):
    def test_highlight_finds_passage_in_pdf_points(self):
        pages = load_pdf(PAPER, "DejaVu2.pdf")
        text = next(p.text for p in pages if p.page_number == 11)
        i = text.find("89%")
        result = viewer.highlight(PAPER.read_bytes(), 11, text[max(0, i - 120):i + 80])
        self.assertTrue(result["found"])
        box = result["annotations"][0]
        self.assertTrue(0 <= box["x"] < result["page_width"] and 0 <= box["y"] < result["page_height"])

    def test_missing_passage_and_bad_page(self):
        self.assertFalse(viewer.highlight(PAPER.read_bytes(), 1, "zzzz not in this paper at all zzzz")["found"])
        with self.assertRaises(ValueError):
            viewer.highlight(PAPER.read_bytes(), 999, "anything at all here")

    def test_page_png_fallback(self):
        self.assertTrue(viewer.page_png(PAPER.read_bytes(), 1).startswith(b"\x89PNG"))

    def test_glossary_and_query_expansion(self):
        pages = load_pdf(ROOT / "data" / "papers" / "DejaVu1.pdf", "DejaVu1.pdf")
        glossary = viewer.build_glossary(pages, "DejaVu1.pdf")
        self.assertIn("LSD", [g["abbr"] for g in glossary])
        self.assertIn("lysergic acid", viewer.expand_query("What does LSD do?", glossary))
        self.assertIsNone(viewer.expand_query("What is deja vu?", glossary))

    def test_explain_prompt_rules(self):
        self.assertIn("Illustrative example — not from the papers", viewer.EXPLAIN_SYSTEM_PROMPT)
        self.assertIn("ONLY the given passages", viewer.EXPLAIN_SYSTEM_PROMPT)
        prompt = viewer.build_explain_prompt("RRF merges lists", ["p1", "p2"], "why?", "before")
        self.assertIn("[2] p2", prompt)
        self.assertIn("Question: why?", prompt)


class PanelApiTests(unittest.TestCase):
    def test_pdf_highlight_explain_glossary_endpoints(self):
        from fastapi.testclient import TestClient
        from test_api import EchoLLM, HashEmbedder, NoReranker, PAPERS
        from app.api import main
        from app.pipelines.sessions import SessionManager

        main.set_manager(SessionManager(embedder=HashEmbedder(), reranker=NoReranker(), llm=EchoLLM()))
        try:
            client = TestClient(main.app)
            sid = client.post("/sessions").json()["session_id"]
            client.post(f"/sessions/{sid}/ingest", files=[("files", (p.name, p.read_bytes(), "application/pdf")) for p in PAPERS])
            pdf = client.get(f"/sessions/{sid}/documents/DejaVu2.pdf/pdf")
            self.assertEqual(pdf.content[:4], b"%PDF")
            self.assertEqual(client.post(f"/sessions/{sid}/highlight", json={"doc_id": "DejaVu2.pdf", "page": 1,
                                                                             "passage": "zzzz nothing"}).status_code, 200)
            self.assertIn("LSD", [g["abbr"] for g in client.get(f"/sessions/{sid}/glossary").json()["glossary"]])
            answer = client.post(f"/sessions/{sid}/ask", json={"question": "What is deja vu?"}).json()
            if answer["answer_id"]:
                body = client.post(f"/sessions/{sid}/explain", json={"answer_id": answer["answer_id"],
                                                                    "selected_text": "LSD and deja vu"}).json()
                self.assertEqual(body["label"], "Plain-English explanation — not evidence")
                self.assertEqual(body["glossary"][0]["abbr"], "LSD")
            self.assertEqual(client.post(f"/sessions/{sid}/explain", json={"answer_id": "nope",
                                                                         "selected_text": "abc def"}).status_code, 400)
        finally:
            main.set_manager(None)


if __name__ == "__main__":
    unittest.main()
