"""Batch A tests (N3, N4, N5, N8, N10). No Ollama or model downloads: NLI and LLM are stubbed."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("NLI_ENABLED", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generation import trust  # noqa: E402

PASSAGE = "Experiments at higher reasoning levels were flagged for retrieval 89% more often than the lowest level."


class StubNLI:
    def __init__(self, verdict):
        self.verdict = verdict
        self.available = True

    def judge(self, premise, hypothesis):
        return self.verdict


class TrustTests(unittest.TestCase):
    def test_classify_contradiction_always_fails(self):
        self.assertEqual(trust.classify({"contradiction": 0.6, "entailment": 0.39}), trust.FAIL)
        self.assertEqual(trust.classify({"entailment": 0.8, "contradiction": 0.1}), trust.PASS)
        self.assertEqual(trust.classify({"neutral": 0.9}), trust.WEAK)

    def test_sentence_statuses(self):
        answer = ("Higher reasoning levels were flagged for retrieval 89% more often [1]. "
                  "They were flagged 95% more often [1]. This is an uncited sentence about something.")
        report = trust.sentence_report(answer, {1: PASSAGE})
        self.assertEqual([r["status"] for r in report], ["verified", "unverified", "uncited"])

    def test_nli_contradiction_overrides_word_check(self):
        report = trust.sentence_report("Higher reasoning levels were flagged 89% more often [1].", {1: PASSAGE}, StubNLI(trust.FAIL))
        self.assertEqual(report[0]["status"], "unverified")
        self.assertEqual(report[0]["meaning"], "contradicted")

    def test_drop_unsupported(self):
        answer = "Flagged 89% more often for retrieval at higher reasoning levels [1]. Flagged 95% more often [1]."
        new, removed = trust.drop_unsupported(answer, trust.sentence_report(answer, {1: PASSAGE}))
        self.assertEqual(removed, 1)
        self.assertNotIn("95%", new)

    def test_silent_papers(self):
        self.assertEqual(trust.silent_papers(["a.pdf", "b.pdf"], {"a.pdf": 0.9, "b.pdf": 0.01}), ["b.pdf"])

    def test_verify_table_cells(self):
        table = ("| Paper | Approach / Method | Key Results |\n| --- | --- | --- |\n"
                 "| DejaVu2.pdf | Not stated | Higher reasoning levels were flagged for retrieval 89% more often [1]; "
                 "accuracy rose to 99% [1] |")
        cells = trust.verify_table(table, [{"index": 1, "doc_id": "DejaVu2.pdf", "page": 11, "section": "6", "text": PASSAGE}])
        self.assertEqual([c["status"] for c in cells], ["not_stated", "partly"])
        self.assertEqual(cells[1]["evidence"][0]["page"], 11)

    def test_metadata_labels_bibtex_and_notes(self):
        meta = trust.extract_metadata(ROOT / "data" / "papers" / "PaperQA.pdf", "PaperQA.pdf")
        self.assertTrue(meta["auto"])
        self.assertTrue(meta["title"].startswith("PaperQA"))
        self.assertEqual((meta["authors"][0], meta["year"]), ("Jakub Lála", 2023))
        meta = {"doc_id": "x.pdf", "title": "A Study", "authors": ["Ann Smith", "Bo Lee"], "year": 2023, "doi": None}
        self.assertEqual(trust.short_label(meta, 4, "Results"), "Smith et al., 2023 · p.4 · Results")
        self.assertIn("author = {Ann Smith and Bo Lee}", trust.bibtex([meta]))
        notes = trust.notes_markdown([{"question": "Q?", "answer": "A [1].", "abstained": False,
                                       "sources": [{"index": 1, "doc_id": "x.pdf", "verified": True, "label": "Smith"},
                                                   {"index": 2, "doc_id": "y.pdf", "verified": False}]}])
        self.assertIn("Smith", notes)
        self.assertNotIn("y.pdf", notes)


if __name__ == "__main__":
    unittest.main()


class PartialAnswerTests(unittest.TestCase):
    def test_split_question(self):
        from app.pipelines.rag import split_question
        self.assertEqual(len(split_question("What is the 5-4-3-2-1 technique? What GPU trained the models in these papers?")), 2)
        self.assertEqual(len(split_question("What is deja vu?")), 1)
        self.assertEqual(len(split_question("What is BM25? Why?")), 1)  # a part shorter than 4 words: no split
