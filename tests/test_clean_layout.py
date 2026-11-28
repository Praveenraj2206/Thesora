"""Clean extraction: only a paper's content, two columns read column by column, readable sources,
highlights that survive PDF line breaks, and the zoomable page viewer."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NLI_ENABLED", "0")

import fitz  # noqa: E402

from app.generation.viewer import _fold, highlight  # noqa: E402
from app.ingestion.pdf_loader import load_pdf  # noqa: E402
from ui.page_viewer import viewer_html  # noqa: E402
from ui.text_format import is_table, paragraphs, readable_html, table_markdown  # noqa: E402

LEFT = ("The left column starts the paper body with a long sentence about retrieval augmented generation "
        "that keeps going so the block counts as running prose for the cleaner.")
RIGHT = ("The right column continues the argument with a second long sentence about reranking passages "
         "that also keeps going so it counts as running prose for the cleaner.")


def _paper(path: Path) -> None:
    """A two-page, two-column 'paper' with every kind of clutter the cleaner must drop."""
    doc = fitz.open()
    for n in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 25), "Proceedings of the Workshop on Testing, pages 1-9", fontsize=8)  # running header
        page.insert_text((290, 830), str(n + 1), fontsize=9)  # page number
        page.insert_text((30, 600), "arXiv:2601.00001v1 [cs.CL] 1 Jan 2026", fontsize=18, rotate=90)  # side stamp
        top = 80
        if n == 0:
            page.insert_text((120, 80), "Clean Extraction for Testing", fontsize=18)  # title
            page.insert_text((150, 110), "Ada Lovelace   Alan Turing", fontsize=12)  # authors
            page.insert_text((170, 126), "University of Testing, ada@test.org", fontsize=10)  # affiliation
            page.insert_text((270, 160), "Abstract", fontsize=12)
            top = 180
        page.insert_textbox(fitz.Rect(60, top, 285, top + 160), LEFT if n == 0 else LEFT.replace("left", "LEFT"),
                            fontsize=10)
        page.insert_textbox(fitz.Rect(310, top, 535, top + 160), RIGHT, fontsize=10)
        page.draw_rect(fitz.Rect(60, 420, 285, 520))  # a figure: box with a tiny label inside
        page.insert_text((80, 470), "Encoder", fontsize=5)
        page.insert_text((60, 540), "Figure 1: The pipeline.", fontsize=9)
    doc.save(str(path))


class CleanExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "paper.pdf"
        _paper(cls.path)
        cls.pages = load_pdf(cls.path)
        cls.text = "\n".join(p.text for p in cls.pages)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_content_and_title_kept(self):
        self.assertIn("Clean Extraction for Testing", self.text)
        self.assertIn("long sentence about retrieval", " ".join(self.text.split()))
        self.assertIn("Figure 1: The pipeline.", self.text)

    def test_clutter_dropped(self):
        for junk in ("Proceedings of the Workshop", "arXiv:2601", "Ada Lovelace", "ada@test.org", "Encoder"):
            self.assertNotIn(junk, self.text, junk)
        self.assertNotRegex(self.pages[0].text, r"(?m)^\s*1\s*$")  # page number

    def test_left_column_before_right_column(self):
        flat = " ".join(self.text.split())
        self.assertLess(flat.index("The left column"), flat.index("The right column"))
        second = " ".join(self.pages[1].text.split())
        self.assertLess(second.index("The LEFT column"), second.index("The right column"))


class ReadableSourceTests(unittest.TestCase):
    def test_lines_join_into_paragraphs_and_headings_stand_out(self):
        text = "2 Pipeline\n\nThe system has three rel-\nevant parts and a cross-\nencoder.\n\nQuery Validation. Prior"
        parts = paragraphs(text)
        self.assertEqual(parts[0], ("heading", "2 Pipeline"))
        self.assertIn("three relevant parts and a crossencoder.", parts[1][1].replace("cross-encoder", "crossencoder"))
        html = readable_html(text, 1000)
        self.assertIn("<strong>2 Pipeline</strong>", html)

    def test_long_passage_is_cut_at_a_word(self):
        self.assertTrue(readable_html("word " * 400, 50).count("word") <= 12)

    def test_table_passages_are_shown_as_tables(self):
        text = "Table 2: Results.\n\n|Model|Acc|\n|---|---|\n|PaperQA|69.5%|"
        self.assertTrue(is_table(text))
        caption, table = table_markdown(text)
        self.assertEqual(caption, "Table 2: Results.")
        self.assertTrue(table.startswith("|Model|"))


class HighlightTests(unittest.TestCase):
    def test_fold_ignores_spaces_hyphens_ligatures_and_accents(self):
        self.assertEqual(_fold("ﬁne-tuned  Lála"), _fold("finetuned Lala"))

    def test_passage_split_by_a_hyphen_at_line_end_is_highlighted(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 80), "Passages are embedded with a binary quan-", fontsize=11)
        page.insert_text((50, 95), "tization step and placed into a dense index.", fontsize=11)
        result = highlight(doc.tobytes(), 1, "Passages are embedded with a binary quantization step and placed into "
                                              "a dense index.")
        self.assertTrue(result["found"])
        self.assertEqual(len(result["annotations"]), 2)  # one box per printed line
        self.assertGreaterEqual(result["coverage"], 0.99)

    def test_unrelated_passage_is_not_highlighted(self):
        doc = fitz.open()
        doc.new_page().insert_text((50, 80), "Only this sentence is printed on the page.", fontsize=11)
        self.assertFalse(highlight(doc.tobytes(), 1, "A completely different sentence about cooking pasta.")["found"])


class EvidenceSentenceTests(unittest.TestCase):
    def test_highlight_is_the_supporting_sentence_not_the_heading_or_passage(self):
        from app.generation.trust import evidence_sentences

        passage = ("2 Pipeline The Scholar QA architecture (Figure 1) has three primary components. Query Validation. "
                   "Prior to processing a query, we employ OpenAI's omni-moderation-latest model for safeguarding "
                   "against potentially harmful content and return appropriate error messages. 2.1 Retrieval We use "
                   "the Semantic Scholar API for retrieval.")
        answer = ("Omni-moderation-latest is used for safeguarding against potentially harmful content and returning "
                  "appropriate error messages prior to processing a query [1].")
        found = evidence_sentences(answer, {1: passage}, {1: "Ai2 Scholar QA"}, {1: "2 Pipeline"})
        self.assertEqual(len(found[1]), 1)
        self.assertTrue(found[1][0].startswith("Prior to processing a query"))
        self.assertNotIn("Pipeline", found[1][0])

    def test_dropdown_marks_the_evidence(self):
        html = readable_html("Query Validation. Prior to processing a query, we employ the model.", 500,
                             ["Prior to processing a query, we employ the model."])
        self.assertIn("<mark>Prior to processing a query, we employ the model.</mark>", html)


class SuggestionAndRefusalTests(unittest.TestCase):
    def test_only_questions_the_papers_can_answer_are_suggested(self):
        from types import SimpleNamespace

        from app.pipelines.rag import ResearchAssistant

        scores = {"Answerable?": 0.98, "Loosely related?": 0.2, "Unrelated?": 0.02, "Also answerable?": 0.9}
        a = ResearchAssistant.__new__(ResearchAssistant)
        a.retrieve_for = lambda q: [SimpleNamespace(relevance=scores[q])]
        self.assertEqual(a.answerable(list(scores), 3), ["Answerable?", "Also answerable?"])

    def test_refusal_quotes_the_sentence_closest_to_the_question(self):
        from app.generation.trust import closest_sentence

        passage = ("2 Pipeline The architecture has three primary components for retrieval. Query Validation. "
                   "Prior to processing a query, we employ the omni-moderation-latest model for safeguarding.")
        self.assertEqual(closest_sentence("What does omni-moderation-latest do to a query?", passage, "2 Pipeline"),
                         ["Prior to processing a query, we employ the omni-moderation-latest model for safeguarding."])
        self.assertEqual(closest_sentence("What is the price of a GPU?", "Cats sit on mats in the sunny garden."), [])


class PageViewerTests(unittest.TestCase):
    def test_viewer_is_self_contained_with_boxes_in_page_fractions(self):
        page = viewer_html(b"\x89PNG fake", 600.0, 800.0, [{"x": 60, "y": 80, "width": 300, "height": 12}], 500, "p.2")
        self.assertIn('"x": 0.1', page)
        self.assertIn("data:image/png;base64,", page)
        self.assertNotIn("http", page.split("<script>")[1])  # no external scripts: works offline
        self.assertIn("ctrlKey", page)  # trackpad pinch zoom


if __name__ == "__main__":
    unittest.main()
