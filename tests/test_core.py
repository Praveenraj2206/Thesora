"""Dependency-light unit tests: fusion, verification, abstention, LLM errors.

Run with:  python -m unittest discover -s tests -v
No Ollama, FAISS or model downloads are needed (components are stubbed).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.core.errors import LLMUnavailableError  # noqa: E402
from app.core.models import DocumentChunk, RetrievedChunk  # noqa: E402
from app.embeddings.embedder import embedding_dimension  # noqa: E402
from app.generation.llm import BaseLLM, OllamaLLM, strip_think  # noqa: E402
from app.generation.verifier import extract_claims, verify_citations  # noqa: E402
from app.ingestion.pdf_loader import _Line, _build_pages, is_heading_line  # noqa: E402
from app.pipelines.rag import ResearchAssistant, compute_confidence  # noqa: E402
from app.ingestion.chunker import dehyphenate  # noqa: E402
from app.retrieval.retriever import apply_relevance_floor, reciprocal_rank_fusion, sigmoid  # noqa: E402
from scripts.compare_retrievers import first_hit_rank, squash, summarize  # noqa: E402
from scripts.evaluate import aggregate, context_precision, parse_judge_score  # noqa: E402

CHUNK_TEXT = (
    "Reciprocal Rank Fusion merges ranked lists from BM25 and dense retrievers by summing "
    "one over k plus rank, which removes the need to calibrate scores."
)


def make_chunk(text: str = CHUNK_TEXT, doc: str = "paper.pdf") -> DocumentChunk:
    return DocumentChunk("paper.pdf:p1:c0", doc, 1, "Methods", text, 0)


def retrieved(relevance: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(), store_index=0, relevance=relevance)


class FakeStore:
    def __len__(self) -> int:
        return 1


class FakeRetriever:
    def __init__(self, relevance: float) -> None:
        self.relevance = relevance

    def retrieve(self, query: str):
        return [retrieved(self.relevance)]


class ExplodingLLM(BaseLLM):
    def generate(self, system_prompt, user_prompt, max_tokens=0):
        raise AssertionError("LLM must not be called when abstaining")


class ScriptedLLM(BaseLLM):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, system_prompt, user_prompt, max_tokens=0):
        return self.reply


def assistant(relevance: float, llm: BaseLLM) -> ResearchAssistant:
    return ResearchAssistant(store=FakeStore(), embedder=object(), retriever=FakeRetriever(relevance), llm=llm)


class FusionTests(unittest.TestCase):
    def test_rrf_scores_and_order(self):
        fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))
        self.assertAlmostEqual(fused[1], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused[3], 1 / 63 + 1 / 61)
        ranked = [i for i, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)]
        self.assertEqual(ranked[0], 1)
        self.assertEqual(set(ranked), {1, 2, 3})

    def test_rrf_top_k_and_empty(self):
        self.assertEqual(len(reciprocal_rank_fusion([[1, 2, 3, 4], [4, 3]], top_k=2)), 2)
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_sigmoid_bounds(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(10), 0.99)
        self.assertLess(sigmoid(-1000), 1e-6)


class VerificationTests(unittest.TestCase):
    def test_supported_claim_verifies(self):
        answer = "RRF merges ranked lists from BM25 and dense retrievers by summing one over k plus rank [1]."
        result = verify_citations(answer, {1: CHUNK_TEXT})
        self.assertTrue(result.per_source[1].verified)
        self.assertEqual(result.unverified, [])

    def test_fabricated_claim_is_flagged(self):
        answer = "The method was trained on ten million images using eight GPUs for several weeks [1]."
        result = verify_citations(answer, {1: CHUNK_TEXT})
        self.assertFalse(result.per_source[1].verified)
        self.assertEqual(result.unverified, [1])

    def test_invalid_citation_number(self):
        result = verify_citations("Rank fusion merges BM25 and dense lists [7].", {1: CHUNK_TEXT})
        self.assertEqual(result.invalid, [7])

    def test_citation_after_period_attaches_to_previous_sentence(self):
        claims = extract_claims("Fusion merges ranked lists. [1] Another unrelated sentence.")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0][1], [1])


class AbstentionTests(unittest.TestCase):
    def test_abstains_below_threshold_without_calling_llm(self):
        result = assistant(0.2, ExplodingLLM()).ask("What is quantum gravity?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.message, config.ABSTAIN_MESSAGE)
        self.assertAlmostEqual(result.confidence, 0.2)

    def test_answers_above_threshold_with_verification(self):
        reply = "RRF merges ranked lists from BM25 and dense retrievers by summing one over k plus rank [1]."
        result = assistant(0.9, ScriptedLLM(reply)).ask("What is RRF?")
        self.assertFalse(result.abstained)
        self.assertTrue(result.sources[0].verified)
        self.assertGreater(result.confidence, 0.5)

    def test_unverifiable_citation_flagged_in_response(self):
        reply = "The system achieved ninety nine percent accuracy on every benchmark ever created [1]."
        result = assistant(0.9, ScriptedLLM(reply)).ask("How accurate is it?")
        self.assertEqual(result.unverified_citations, [1])
        self.assertFalse(result.sources[0].verified)

    def test_model_declared_insufficiency_abstains(self):
        result = assistant(0.9, ScriptedLLM(config.INSUFFICIENT_MARKER)).ask("Unrelated?")
        self.assertTrue(result.abstained)

    def test_confidence_range(self):
        self.assertEqual(compute_confidence([], 1.0), 0.0)
        self.assertLessEqual(compute_confidence([1.0, 1.0, 1.0], 1.0), 1.0)


class FakeResponse:
    def __init__(self, status: int, payload=None, text: str = "") -> None:
        self.status_code, self._payload, self.text = status, payload or {}, text
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses) -> None:
        self.responses, self.payloads = list(responses), []

    def post(self, url, json=None, timeout=None):
        self.payloads.append(json)
        return self.responses.pop(0)


class HardeningTests(unittest.TestCase):
    def test_think_option_rejected_then_retried_without_it(self):
        llm = OllamaLLM(base_url="http://x", model="m")
        good = FakeResponse(200, {"message": {"content": "Answer [1]"}, "done_reason": "stop"})
        llm._session = FakeSession([FakeResponse(400, text='{"error":"model does not support think"}'), good])
        self.assertEqual(llm.generate("s", "u"), "Answer [1]")
        self.assertIn("think", llm._session.payloads[0])
        self.assertNotIn("think", llm._session.payloads[1])

    def test_other_400_is_not_retried(self):
        llm = OllamaLLM(base_url="http://x", model="m")
        llm._session = FakeSession([FakeResponse(400, text="bad json body")])
        from app.core.errors import LLMGenerationError
        with self.assertRaises(LLMGenerationError):
            llm.generate("s", "u")
        self.assertEqual(len(llm._session.payloads), 1)

    def test_embedding_dimension_old_and_new_names(self):
        class Old:
            def get_sentence_embedding_dimension(self): return 384
        class New:
            def get_embedding_dimension(self): return 384
        self.assertEqual(embedding_dimension(Old()), 384)
        self.assertEqual(embedding_dimension(New()), 384)
        from app.core.errors import EmbeddingError
        with self.assertRaises(EmbeddingError):
            embedding_dimension(object())

    def test_defaults(self):
        self.assertEqual(config.OLLAMA_TIMEOUT_S, 600)
        self.assertEqual(__import__("os").environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING"), "1")


class RetrievalQualityTests(unittest.TestCase):
    def test_floor_drops_weak_chunks_but_keeps_best(self):
        ranked = [retrieved(r) for r in (0.9, 0.5, 0.05, 0.01)]
        self.assertEqual([c.relevance for c in apply_relevance_floor(ranked, 0.1)], [0.9, 0.5])
        weak_only = [retrieved(r) for r in (0.05, 0.01)]
        self.assertEqual([c.relevance for c in apply_relevance_floor(weak_only, 0.1)], [0.05])
        self.assertEqual(len(apply_relevance_floor(ranked, 0.0)), 4)
        self.assertEqual(apply_relevance_floor([], 0.1), [])

    def test_dehyphenate_rejoins_split_words_only(self):
        self.assertEqual(dehyphenate("the hippo-\ncampus and dem-\n onstrated it"), "the hippocampus and demonstrated it")
        self.assertEqual(dehyphenate("state-of-the-art models"), "state-of-the-art models")
        self.assertEqual(dehyphenate("Gtr1p-\nGtr2p complex"), "Gtr1p-\nGtr2p complex")

    def test_evidence_matching_ignores_hyphenation_and_case(self):
        texts = ["unrelated text", "the HIPPO-\ncampus and parietal lobes matter", "89% more often"]
        self.assertEqual(first_hit_rank(texts, ["hippocampus and parietal lobes"]), 2)
        self.assertEqual(first_hit_rank(texts, ["no such phrase", "89% more often"]), 3)
        self.assertIsNone(first_hit_rank(texts, ["absent"]))
        self.assertEqual(squash("5-4-3-2-1 Rag"), "54321rag")

    def test_summarize_hit_rate_and_mrr(self):
        summary = summarize([("exact", 1), ("exact", None), ("paraphrase", 2), ("paraphrase", 4)])
        self.assertEqual(summary["all"], {"n": 4, "hit": 0.75, "mrr": round((1 + 0.5 + 0.25) / 4, 3)})
        self.assertEqual(summary["exact"]["hit"], 0.5)
        self.assertEqual(summary["paraphrase"]["mrr"], round((0.5 + 0.25) / 2, 3))


class MiscTests(unittest.TestCase):
    def test_ollama_down_raises_clean_error(self):
        llm = OllamaLLM(base_url="http://127.0.0.1:9", timeout_s=2)
        with self.assertRaises(LLMUnavailableError) as ctx:
            llm.generate("sys", "user")
        self.assertIn("ollama", str(ctx.exception).lower())
        self.assertFalse(llm.health()["reachable"])

    def test_strip_think(self):
        self.assertEqual(strip_think("<think>hmm</think>Answer [1]"), "Answer [1]")
        self.assertEqual(strip_think("Answer<think>unfinished"), "Answer")

    def test_strip_think_lone_closing_tag(self):
        raw = "Okay, the user asks about X. Passage [1] says Y.\n</think>\n\nThe answer is Y [1]."
        self.assertEqual(strip_think(raw), "The answer is Y [1].")

    def test_marker_mentioned_in_answer_does_not_abstain(self):
        reply = "Rule: say INSUFFICIENT_CONTEXT if unknown. But the paper audited 22 models across 12 benchmarks [1]."
        self.assertFalse(assistant(0.9, ScriptedLLM(reply)).ask("How many models?").abstained)
        self.assertTrue(assistant(0.9, ScriptedLLM("**INSUFFICIENT_CONTEXT**")).ask("Unrelated?").abstained)

    def test_wrapped_heading_is_merged(self):
        lines = [_Line("MAIN ASPECTS AND COMPONENTS", 12.0, True), _Line("OF CONSCIOUSNESS IN HUMAN", 12.0, True),
                 _Line("EXPERIENCE", 12.0, True), _Line("", 0.0, False),
                 _Line("Body text of the section goes here and continues for a while.", 10.0, False)]
        body = [_Line("Plain body sentence number one is quite long indeed.", 10.0, False)] * 5
        pages = _build_pages([lines + body], "doc.pdf")
        self.assertEqual(len(pages[0].headings), 1)
        self.assertEqual(pages[0].headings[0][1], "MAIN ASPECTS AND COMPONENTS OF CONSCIOUSNESS IN HUMAN EXPERIENCE")

    def test_heading_detection(self):
        self.assertTrue(is_heading_line("2.1 Hybrid Retrieval", 10.0, True, 10.0))
        self.assertTrue(is_heading_line("Introduction", 14.0, False, 10.0))
        self.assertFalse(is_heading_line("This is a normal sentence of body text.", 10.0, False, 10.0))
        self.assertFalse(is_heading_line("12", 14.0, False, 10.0))

    def test_eval_helpers(self):
        self.assertEqual(parse_judge_score("SCORE: 4"), 4.0)
        self.assertEqual(parse_judge_score("I'd say 3/5."), 3.0)
        self.assertIsNone(parse_judge_score("no idea"))
        self.assertAlmostEqual(context_precision([0.9, 0.1, 0.8], threshold=0.5), (1 + 2 / 3) / 2)
        rows = [{"abstained": False, "metrics": {"faithfulness": 1.0, "context_precision": 0.5,
                                                 "context_recall": 0.5, "answer_relevancy": 0.8}},
                {"abstained": True, "metrics": {"faithfulness": None, "context_precision": 0.0,
                                                "context_recall": 0.0, "answer_relevancy": 0.0}}]
        summary = aggregate(rows)
        self.assertEqual(summary["faithfulness"], 1.0)
        self.assertEqual(summary["abstention_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
