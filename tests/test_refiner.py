"""Tests for the optional question refiner (prompt-master feature).

Run with:  python -m unittest discover -s tests -v
Uses stub LLMs only, so no Ollama or model downloads are needed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.core.errors import LLMGenerationError  # noqa: E402
from app.generation.llm import BaseLLM  # noqa: E402
from app.generation.prompt_refiner import (  # noqa: E402
    MODE_ASK,
    MODE_COMPARE,
    build_refine_prompts,
    parse_refiner_reply,
    refine_prompt,
)


class StubLLM(BaseLLM):
    """Returns a fixed reply and records the prompts it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, int]] = []

    def generate(self, system_prompt, user_prompt, max_tokens=0):
        self.calls.append((system_prompt, user_prompt, max_tokens))
        return self.reply


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        refined, why = parse_refiner_reply('{"refined": "Which brain regions are involved?", "why": "Named the target."}')
        self.assertEqual(refined, "Which brain regions are involved?")
        self.assertEqual(why, "Named the target.")

    def test_fenced_json_with_surrounding_text(self):
        reply = 'Here you go:\n```json\n{"refined": "What is BM25?", "why": "Clearer."}\n```'
        self.assertEqual(parse_refiner_reply(reply)[0], "What is BM25?")

    def test_non_json_falls_back_to_first_line_without_label(self):
        refined, why = parse_refiner_reply('Refined question: "What datasets were used?"\nextra line')
        self.assertEqual(refined, "What datasets were used?")
        self.assertEqual(why, "")


class RefineTests(unittest.TestCase):
    def test_refines_and_passes_papers_and_mode(self):
        llm = StubLLM('{"refined": "Which brain regions do Cleary and Brown link to deja vu?", "why": "Added the authors."}')
        result = refine_prompt(llm, "  brain   stuff deja vu ", MODE_ASK, ["DejaVu1.pdf"])
        self.assertTrue(result.changed)
        self.assertEqual(result.original, "brain stuff deja vu")
        self.assertIn("Cleary", result.refined)
        system, user, max_tokens = llm.calls[0]
        self.assertIn("question", system)
        self.assertIn("DejaVu1.pdf", user)
        self.assertEqual(max_tokens, config.REFINE_MAX_TOKENS)

    def test_compare_mode_asks_for_a_topic(self):
        system, _ = build_refine_prompts("compare them", MODE_COMPARE, [])
        self.assertIn("comparison topic", system)
        self.assertIn("not a question", system)

    def test_unchanged_question_is_reported(self):
        result = refine_prompt(StubLLM('{"refined": "What is BM25?", "why": ""}'), "what is bm25?")
        self.assertFalse(result.changed)
        self.assertTrue(result.rationale)

    def test_overlong_reply_is_truncated(self):
        long_reply = '{"refined": "' + " ".join(["word"] * 200) + '", "why": "x"}'
        result = refine_prompt(StubLLM(long_reply), "tell me everything")
        self.assertLessEqual(len(result.refined.split()), config.REFINE_MAX_WORDS + config.REFINE_WORD_SLACK)

    def test_unusable_reply_raises_friendly_error(self):
        with self.assertRaises(LLMGenerationError):
            refine_prompt(StubLLM(config.INSUFFICIENT_MARKER), "what is this")

    def test_bad_mode_and_empty_text_are_rejected(self):
        with self.assertRaises(ValueError):
            refine_prompt(StubLLM("{}"), "question", "poem")
        with self.assertRaises(ValueError):
            refine_prompt(StubLLM("{}"), "   ")


if __name__ == "__main__":
    unittest.main()
