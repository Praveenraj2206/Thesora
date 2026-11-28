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

PAPER = ROOT / "data" / "papers" / "PaperQA.pdf"


class ViewerTests(unittest.TestCase):
    def test_highlight_finds_passage_in_pdf_points(self):
        pages = load_pdf(PAPER, "PaperQA.pdf")
        page = next(p for p in pages if "4,000" in p.text)
        i = page.text.find("4,000")
        result = viewer.highlight(PAPER.read_bytes(), page.page_number, page.text[max(0, i - 120):i + 80])
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
        pages = load_pdf(ROOT / "data" / "papers" / "DeepEra.pdf", "DeepEra.pdf")
        glossary = viewer.build_glossary(pages, "DeepEra.pdf")
        self.assertIn("LFS", [g["abbr"] for g in glossary])
        self.assertIn("Logic Fidelity Score", viewer.expand_query("How is LFS computed?", glossary))
        self.assertIsNone(viewer.expand_query("What is a reranker?", glossary))

    def test_explain_prompt_rules(self):
        self.assertIn("ONLY the given passages", viewer.EXPLAIN_SYSTEM_PROMPT)
        self.assertIn("five short follow-up questions", viewer.EXPLAIN_SYSTEM_PROMPT)
        self.assertNotIn("What, Why, How", viewer.EXPLAIN_SYSTEM_PROMPT)
        prompt = viewer.build_explain_prompt("RRF merges lists", ["p1", "p2"], "why?", "before", "What is RRF?")
        self.assertIn("[2] p2", prompt)
        self.assertIn("Original question: What is RRF?", prompt)
        self.assertIn("Student's question: why?", prompt)

    def test_parse_explanation_json_and_fallback(self):
        text, qs = viewer.parse_explanation('{"explanation": "RRF merges  lists.", "follow_ups": ["Why k = 60?", '
                                            '"What is BM25", "Why k = 60?"]}')
        self.assertEqual(text, "RRF merges lists.")
        self.assertEqual(qs, ["Why k = 60?", "What is BM25?"])
        text, qs = viewer.parse_explanation("RRF merges ranked lists.\nFollow-up questions:\n1. Why 60?\n2. What is BM25?")
        self.assertEqual(text, "RRF merges ranked lists.")
        self.assertEqual(qs, ["Why 60?", "What is BM25?"])
        broken = ('{"explanation": "Omni checks each query.", "follow_ups": ["What does it detect?", '
                  '"Is it used only at the start?"}')  # the model forgot the closing bracket
        self.assertEqual(viewer.parse_explanation(broken, 5),
                         ("Omni checks each query.", ["What does it detect?", "Is it used only at the start?"]))
        self.assertEqual(viewer.parse_questions('{"questions": ["One?", "Two?"'), ["One?", "Two?"])
        text, qs = viewer.parse_explanation('{"questions": ["A?", "B?"]}')  # wrong key: no raw JSON shown
        self.assertEqual((text, qs), ("", ["A?", "B?"]))
        self.assertEqual(viewer.parse_questions('{"questions": ["One", "Two?"]}'), ["One?", "Two?"])
        self.assertEqual(viewer.parse_questions("1. First?\n2. Second?\nnot a question"), ["First?", "Second?"])


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
            pdf = client.get(f"/sessions/{sid}/documents/PaperQA.pdf/pdf")
            self.assertEqual(pdf.content[:4], b"%PDF")
            self.assertEqual(client.post(f"/sessions/{sid}/highlight", json={"doc_id": "PaperQA.pdf", "page": 1,
                                                                             "passage": "zzzz nothing"}).status_code, 200)
            self.assertIn("LFS", [g["abbr"] for g in client.get(f"/sessions/{sid}/glossary").json()["glossary"]])
            answer = client.post(f"/sessions/{sid}/ask", json={"question": "What is LitQA?"}).json()
            if answer["answer_id"]:
                body = client.post(f"/sessions/{sid}/explain", json={"answer_id": answer["answer_id"],
                                                                    "selected_text": "LFS and reranking"}).json()
                self.assertEqual(body["label"], "Plain-English explanation — not evidence")
                self.assertEqual(body["glossary"][0]["abbr"], "LFS")
            self.assertEqual(client.post(f"/sessions/{sid}/explain", json={"answer_id": "nope",
                                                                         "selected_text": "abc def"}).status_code, 400)
        finally:
            main.set_manager(None)


if __name__ == "__main__":
    unittest.main()
