"""N1 tables / columns, N2 toggles, N12 agree, N13 stress helpers. No Ollama or model downloads."""
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

from app.generation import agree  # noqa: E402
from app.ingestion import layout  # noqa: E402

PAPERS = ROOT / "data" / "papers"


class LayoutTests(unittest.TestCase):
    def test_tables_become_citable_chunks(self):
        tables = layout.table_chunks(PAPERS / "DejaVu1.pdf", "DejaVu1.pdf", 100, {1: "Intro"})
        self.assertTrue(tables)
        self.assertEqual(tables[0].chunk_type, "table")
        self.assertIn("|", tables[0].text)
        self.assertGreaterEqual(tables[0].chunk_index, 100)

    def test_column_report_flags_two_column_paper(self):
        self.assertTrue(any(r["two_column"] for r in layout.column_report(PAPERS / "DejaVu1.pdf")))
        self.assertFalse(any(r["two_column"] for r in layout.column_report(PAPERS / "DejaVu3.pdf")))


class AgreeTests(unittest.TestCase):
    def test_position_is_extractive(self):
        text = "Intro words here now. Reasoning levels were flagged 89% more often for retrieval. End of it here."
        self.assertEqual(agree.position(text, "reasoning flagged retrieval"),
                         "Reasoning levels were flagged 89% more often for retrieval.")

    def test_number_conflict_is_severity_two(self):
        rows = [{"doc_id": "a", "position": "Accuracy reached 91 percent on the benchmark test."},
                {"doc_id": "b", "position": "Accuracy reached 75 percent on the benchmark test."},
                {"doc_id": "c", "position": "Participants described a strong sense of familiarity."}]
        conflicts = agree.compare(rows)
        self.assertEqual(conflicts[0]["severity"], 2)
        self.assertEqual(set(conflicts[0]["papers"]), {"a", "b"})
        self.assertTrue(all(c["severity"] == 1 for c in conflicts[1:]))

    def test_nli_contradiction_is_severity_three(self):
        class Stub:
            def judge(self, p, h):
                return "FAIL"
        conflicts = agree.compare([{"doc_id": "a", "position": "X works."}, {"doc_id": "b", "position": "X fails."}], Stub())
        self.assertEqual(conflicts[0]["severity"], 3)

    def test_agree_endpoint(self):
        from fastapi.testclient import TestClient
        from test_api import EchoLLM, HashEmbedder, NoReranker, PAPERS as P
        from app.api import main
        from app.pipelines.sessions import SessionManager

        main.set_manager(SessionManager(embedder=HashEmbedder(), reranker=NoReranker(), llm=EchoLLM()))
        try:
            client = TestClient(main.app)
            sid = client.post("/sessions").json()["session_id"]
            empty = client.post(f"/sessions/{sid}/agree", json={"topic": "memory and familiarity"}).json()
            self.assertEqual(empty["rows"], [])
            client.post(f"/sessions/{sid}/ingest", files=[("files", (p.name, p.read_bytes(), "application/pdf")) for p in P])
            body = client.post(f"/sessions/{sid}/agree", json={"topic": "memory and familiarity"}).json()
            self.assertTrue(body["rows"])
            self.assertIn(body["rows"][0]["position"], " ".join(body["rows"][0]["passage"].split()))
            self.assertFalse(body["nli_available"])
        finally:
            main.set_manager(None)


class StressTests(unittest.TestCase):
    def test_alter_numbers(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import stress_eval
        self.assertEqual(stress_eval.alter_numbers("rose 89% in 12 runs"), "rose 96% in 19 runs")


class MultiQueryTests(unittest.TestCase):
    def test_multi_query_falls_back_on_llm_error(self):
        from app.pipelines.rag import ResearchAssistant

        class Boom:
            def generate(self, *a, **k):
                raise RuntimeError("down")
        assistant = ResearchAssistant.__new__(ResearchAssistant)
        assistant.llm = Boom()
        self.assertEqual(ResearchAssistant._multi_query(assistant, "q", ["x"]), ["x"])


if __name__ == "__main__":
    unittest.main()
