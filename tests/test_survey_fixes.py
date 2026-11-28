"""Regression tests for the problems found with the six survey papers (glued words, paper names,
Compare coverage, verifier wording, context expansion, glossary duplicates)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NLI_ENABLED", "0")

import fitz  # noqa: E402

from app import config  # noqa: E402
from app.generation.trust import extract_metadata, short_label  # noqa: E402
from app.generation.verifier import score_claim, verify_citations  # noqa: E402
from app.generation.viewer import expand_query, highlight, merge_glossary  # noqa: E402
from app.ingestion.pdf_loader import load_pdf  # noqa: E402
from app.pipelines.rag import _merge_overlap  # noqa: E402

SENTENCE = "Query Validation. Prior to processing a query we employ the moderation model"


def _tight_pdf(gap_em: float) -> bytes:
    """One line whose words are placed with a small gap and no space characters (like justified LaTeX)."""
    doc = fitz.open()
    page = doc.new_page()
    font, size, x = fitz.Font("helv"), 10, 50.0
    writer = fitz.TextWriter(page.rect)
    for word in SENTENCE.split():
        writer.append((x, 80), word, font=font, fontsize=size)
        x += font.text_length(word, size) + gap_em * size
    writer.write_text(page)
    return doc.tobytes()


def _put(page, x, y, text, size=10, rotate=0):
    page.insert_text((x, y), text, fontsize=size, fontname="helv", rotate=rotate)


def _paperqa_like() -> fitz.Document:
    doc = fitz.open()
    p = doc.new_page()
    _put(p, 30, 700, "arXiv:2312.07559v2 [cs.CL] 14 Dec 2023", 20, rotate=90)
    _put(p, 120, 80, "PaperQA: Retrieval-Augmented Generative Agent for", 17)
    _put(p, 120, 100, "Scientific Research", 17)
    _put(p, 120, 130, "Jakub Lala")
    _put(p, 120, 142, "Future House")
    _put(p, 300, 130, "Odhran O'Donoghue")
    _put(p, 250, 200, "ABSTRACT", 12)
    _put(p, 120, 220, "Earlier work 2019, 2020, 2020, 2021 studied this.")
    return doc


def _preprint_like() -> fitz.Document:
    doc = fitz.open()
    p = doc.new_page()
    _put(p, 100, 60, "HYSEMRAG: A HYBRID SEMANTIC RETRIEVAL-AUGMENTED", 14)
    _put(p, 100, 78, "GENERATION FRAMEWORK FOR AUTOMATED LITERATURE", 14)
    _put(p, 250, 100, "A PREPRINT", 11)
    _put(p, 230, 125, "Alejandro Godinez")
    _put(p, 180, 137, "Department of Environmental Health Sciences")
    _put(p, 250, 160, "2025-07-31")
    _put(p, 260, 190, "ABSTRACT", 11)
    _put(p, 80, 210, "Prior work [Gu 2024, Li 2024, Tsai 2024, Barnett 2024] found problems.")
    return doc


def _logo_title_like() -> fitz.Document:
    doc = fitz.open()
    p = doc.new_page()
    _put(p, 60, 40, "Proceedings of the 63rd Annual Meeting of the ACL, pages 513-523", 8)
    _put(p, 60, 50, "July 27 - August 1, 2025 (c)2025 Association for Computational Linguistics", 8)
    _put(p, 150, 90, ": Organized Literature Synthesis with Attribution", 15)
    _put(p, 90, 120, "Amanpreet Singh*")
    _put(p, 220, 120, "Joseph Chee Chang*")
    _put(p, 200, 140, "Allen Institute for AI")
    _put(p, 150, 180, "Abstract", 12)
    _put(p, 60, 200, "Retrieval-augmented generation (Lewis et al., 2020; Asai et al., 2024).")
    return doc


class GluedWordsTests(unittest.TestCase):
    def test_tight_justified_line_gets_its_spaces_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tight.pdf"
            path.write_bytes(_tight_pdf(0.1))
            text = load_pdf(path)[0].text
        self.assertIn("Prior to processing a query", text)

    def test_highlight_finds_passage_in_glued_pdf(self):
        result = highlight(_tight_pdf(0.05), 1, "Prior to processing a query we employ the moderation model")
        self.assertTrue(result["found"])

    def test_letters_of_one_word_are_not_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tight.pdf"
            path.write_bytes(_tight_pdf(0.3))
            text = load_pdf(path)[0].text
        self.assertIn("moderation", text.split())


class MetadataTests(unittest.TestCase):
    def _meta(self, doc: fitz.Document, name: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            doc.save(str(path))
            return extract_metadata(path, name)

    def test_arxiv_stamp_is_not_the_title_and_gives_the_year(self):
        meta = self._meta(_paperqa_like(), "PaperQA.pdf")
        self.assertTrue(meta["title"].startswith("PaperQA: Retrieval-Augmented"))
        self.assertEqual(meta["year"], 2023)
        self.assertEqual(meta["authors"][0], "Jakub Lala")
        self.assertNotIn("Future House", meta["authors"])

    def test_preprint_header_is_skipped_and_date_wins_over_cited_years(self):
        meta = self._meta(_preprint_like(), "HySEMRAG.pdf")
        self.assertTrue(meta["title"].startswith("HYSEMRAG"))
        self.assertEqual(meta["authors"], ["Alejandro Godinez"])
        self.assertEqual(meta["year"], 2025)
        self.assertTrue(short_label(meta, 3, "Methods").startswith("Godinez, 2025"))

    def test_title_after_a_logo_gets_the_file_name(self):
        meta = self._meta(_logo_title_like(), "Ai2_Scholar_QA.pdf")
        self.assertEqual(meta["title"], "Ai2 Scholar QA: Organized Literature Synthesis with Attribution")
        self.assertEqual(meta["year"], 2025)
        self.assertEqual(meta["authors"][:2], ["Amanpreet Singh", "Joseph Chee Chang"])


class VerifierWordingTests(unittest.TestCase):
    PASSAGE = ("The agentic self-correction system demonstrated 68.3% single-pass success rate. Quality-controlled "
               "analysis showed citation accuracy of 99.0% ± 4.5% and semantic similarity scores of 1.000.")
    SUBJECT = "HySEMRAG HYSEMRAG: A HYBRID SEMANTIC RETRIEVAL-AUGMENTED GENERATION FRAMEWORK Godinez"

    def test_paper_name_and_reporting_words_do_not_count_against_a_claim(self):
        ok, _ = score_claim("HySemRAG reports a citation accuracy of 99.0%", self.PASSAGE, self.SUBJECT)
        self.assertTrue(ok)

    def test_wrong_number_still_fails(self):
        ok, _ = score_claim("HySemRAG reports a citation accuracy of 97.0%", self.PASSAGE, self.SUBJECT)
        self.assertFalse(ok)

    def test_another_papers_name_still_counts(self):
        ok, _ = score_claim("PaperQA reports a citation accuracy of 99.0%", self.PASSAGE, self.SUBJECT)
        self.assertFalse(ok)

    def test_verify_citations_uses_subjects(self):
        answer = "HySemRAG reports a citation accuracy of 99.0% [1]."
        self.assertEqual(verify_citations(answer, {1: self.PASSAGE}, {1: self.SUBJECT}).claims_verified, 1)


class ContextAndGlossaryTests(unittest.TestCase):
    def test_merge_overlap_drops_shared_text(self):
        first = "The Logic Fidelity Score evaluates answers. Given a question q"
        second = "Given a question q, golden answer a and candidate answer, the LLM assigns a score from 0 to 5."
        merged = _merge_overlap(first, second)
        self.assertEqual(merged.count("Given a question q"), 1)
        self.assertIn("0 to 5", merged)

    def test_expansion_uses_neighbours_in_paper_order_not_page_index(self):
        from types import SimpleNamespace

        from app.core.models import DocumentChunk
        from app.pipelines.rag import ResearchAssistant

        def c(page, pos, text):
            return DocumentChunk(chunk_id=f"P.pdf:p{page}:c{pos}", doc_id="P.pdf", page_number=page, section="S",
                                 text=text, chunk_index=pos)

        chunks = [c(1, 0, "Unrelated first page text."), c(1, 1, "We introduce the Logic Fidelity Score. Given a ques-"),
                  c(2, 0, "tion q, the LLM assigns a score from 0 to 5."), c(2, 1, "Other text on page two.")]
        a = ResearchAssistant.__new__(ResearchAssistant)
        a.store = SimpleNamespace(_chunks=chunks)
        expanded = a._context_text(chunks[1], 1)
        self.assertIn("from 0 to 5", expanded)            # the next passage, on the next page
        self.assertNotIn("Other text on page two", expanded)  # not another passage that shares chunk_index 1
        self.assertIn("Given a ques", a._context_text(chunks[2], 1))  # mid-sentence start: previous first

    def test_expansion_is_on_by_default_for_top_passages(self):
        self.assertTrue(config.CONTEXT_EXPANSION)
        self.assertGreaterEqual(config.CONTEXT_EXPANSION_TOP, 1)

    def test_glossary_has_one_entry_per_term(self):
        glossary: list = []
        for doc in ("A.pdf", "B.pdf", "C.pdf"):
            glossary = merge_glossary(glossary, [
                {"abbr": "LLM", "long_form": "large language model", "page": 1, "doc_id": doc},
                {"abbr": "LLMs", "long_form": "large language models", "page": 2, "doc_id": doc},
            ])
        self.assertEqual([g["abbr"] for g in glossary], ["LLM"])
        self.assertEqual(glossary[0]["papers"], ["A.pdf", "B.pdf", "C.pdf"])
        self.assertIn("large language model", expand_query("How do LLMs cite?", glossary))


class ScopeAndRankingTests(unittest.TestCase):
    def _assistant(self):
        from types import SimpleNamespace

        from app.generation.trust import paper_names
        from app.pipelines.rag import ResearchAssistant

        a = ResearchAssistant.__new__(ResearchAssistant)
        page = SimpleNamespace(text="OpenScholar " * 6 + "ScholarQABench " * 5 + "LLMs " * 9)
        a.paper_names = {
            "PaperQA.pdf": paper_names({"title": "PaperQA: Retrieval-Augmented Agent", "authors": ["Jakub Lála"]}, []),
            "Asai.pdf": paper_names({"title": "Synthesizing scientific literature", "authors": ["Akari Asai"]}, [page]),
            "DeepEra.pdf": paper_names({"title": "DeepEra: A Deep Evidence Reranking Agent"}, []),
        }
        return a

    def test_paper_names_from_title_author_and_repeated_system_names(self):
        names = self._assistant().paper_names
        self.assertIn("PaperQA", names["PaperQA.pdf"])
        self.assertIn("Lála", names["PaperQA.pdf"])
        self.assertIn("OpenScholar", names["Asai.pdf"])
        self.assertNotIn("LLMs", names["Asai.pdf"])

    def test_question_naming_a_paper_is_scoped_and_unnamed(self):
        scope, unnamed = self._assistant()._scope("Which embedding model does PaperQA use?")
        self.assertEqual(scope, {"PaperQA.pdf"})
        self.assertEqual(unnamed, "Which embedding model does the paper use?")

    def test_similar_name_is_not_a_match_and_no_name_means_no_scope(self):
        a = self._assistant()
        self.assertEqual(a._scope("How does PaperQA2 compare?")[0], None)
        self.assertEqual(a._scope("What is a reranker?"), (None, "What is a reranker?"))

    def test_spell_out_for_reranker(self):
        from app.generation.viewer import spell_out

        glossary = [{"abbr": "LFS", "long_form": "Logic Fidelity Score", "page": 5, "doc_id": "D.pdf"}]
        self.assertEqual(spell_out("How is LFS computed?", glossary), "How is LFS (Logic Fidelity Score) computed?")
        self.assertIsNone(spell_out("How is it computed?", glossary))

    def test_number_questions_keep_a_relevant_table(self):
        from app.core.models import DocumentChunk, RetrievedChunk
        from app.retrieval.retriever import _with_table_slot

        def item(kind, rel, i):
            chunk = DocumentChunk(chunk_id=str(i), doc_id="P.pdf", page_number=1, section="", text=f"t{i}",
                                  chunk_index=i, chunk_type=kind)
            return RetrievedChunk(chunk=chunk, store_index=i, dense_score=0.0, bm25_score=None, rrf_score=0.0,
                                  relevance=rel)

        ranked = [item("text", 0.99, i) for i in range(5)] + [item("table", 0.9, 5)]
        top = _with_table_slot("What accuracy did it get?", ranked, 5)
        self.assertEqual(top[-1].chunk.chunk_type, "table")
        self.assertEqual(len(top), 5)
        self.assertTrue(all(c.chunk.chunk_type == "text" for c in _with_table_slot("Who wrote it?", ranked, 5)))

    def test_accent_drawn_as_its_own_glyph_joins_the_letter(self):
        from app.ingestion.pdf_loader import line_text

        def ch(c, x0):
            return {"c": c, "bbox": (x0, 0, x0 + 5, 10)}

        line = {"spans": [{"size": 10, "chars": [ch("L", 0), ch("´", 7), ch("a", 6), ch("l", 11), ch("a", 16)]}]}
        self.assertEqual(line_text(line), "Lála")

    def test_agree_candidate_sentences_skip_captions_and_code(self):
        from app.generation.agree import candidate_sentences

        text = ("Table 4: Comparison of Hallucination in Citations. GPT-4o hallucinates citations in most of the "
                "cases we checked. # Process chunks in embedding batches for i in range")
        self.assertEqual(candidate_sentences(text), ["GPT-4o hallucinates citations in most of the cases we checked."])


class TableCaptionTests(unittest.TestCase):
    def test_table_passage_starts_with_its_caption(self):
        from app.ingestion.layout import table_chunks

        doc = fitz.open()
        page = doc.new_page()
        _put(page, 50, 60, "Table 2: Evaluation on LitQA. We compare several systems.")
        rows = [("Model", "Accuracy"), ("PaperQA", "69.5%"), ("Human", "66.8%")]
        for r, (a, b) in enumerate(rows):
            y = 90 + r * 20
            page.draw_rect(fitz.Rect(50, y - 14, 150, y + 6))
            page.draw_rect(fitz.Rect(150, y - 14, 250, y + 6))
            _put(page, 55, y, a)
            _put(page, 155, y, b)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pdf"
            doc.save(str(path))
            chunks = table_chunks(path, "t.pdf", 0, {})
        self.assertTrue(chunks, "table not detected")
        self.assertTrue(chunks[0].text.startswith("Table 2: Evaluation on LitQA"))
        self.assertIn("69.5%", chunks[0].text)


class CompareCoverageTests(unittest.TestCase):
    """Uses the real embedder and reranker on the bundled sample papers (skipped if they cannot load)."""

    @classmethod
    def setUpClass(cls):
        from app.pipelines.rag import ResearchAssistant

        papers = Path(__file__).resolve().parents[1] / "data" / "papers"
        try:
            cls.assistant = ResearchAssistant()
            cls.assistant.ingest_folder(papers)
        except Exception as exc:  # models unavailable offline
            raise unittest.SkipTest(f"models unavailable: {exc}")

    def test_every_paper_gets_passages_even_when_one_dominates(self):
        chunks = self.assistant.retriever.retrieve_across_documents("mxbai-rerank-large-v1 latency on the reranking benchmark")
        papers = {p.name for p in (Path(__file__).resolve().parents[1] / "data" / "papers").glob("*.pdf")}
        self.assertEqual({c.chunk.doc_id for c in chunks}, papers)


if __name__ == "__main__":
    unittest.main()
