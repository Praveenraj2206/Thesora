"""Optional question refiner ("Refine my question"), adapted from prompt-master.

prompt-master (https://github.com/nidhinjs/prompt-master, MIT License, (c) 2026 Nidhin Joseph
Nelson) is a prompt-engineering skill that turns a rough request into one sharp, ready-to-use
prompt for a specific AI tool. Its full instructions are ~500 lines, far too heavy for a 4B model
on a CPU, so this module keeps only the rules that matter when the "tool" is Thesora itself:

* extract the real intent and turn vague verbs into a precise operation (task, output format,
  constraints, success criteria - prompt-master's intent-extraction dimensions);
* produce ONE copy-ready result with no padding or theory;
* never invent facts (no hallucination invite) and never ask for hidden reasoning;
* short, flat instructions with an explicit output format (its guidance for Qwen / Ollama models).

The refiner never answers the question and never changes what is asked; it only rewrites the
user's text into a clearer question (or synthesis topic) for retrieval over the chat's papers.
The user decides whether to use the suggestion.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from app import config
from app.core.errors import LLMGenerationError
from app.generation.llm import BaseLLM

MODE_ASK = "ask"
MODE_COMPARE = "compare"
MODES = (MODE_ASK, MODE_COMPARE)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_LABEL_RE = re.compile(r"^(refined( question| topic)?|question|topic)\s*:\s*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

REFINE_SYSTEM_PROMPT = (
    "You are a prompt engineer for Thesora, a research assistant that answers ONLY from the user's uploaded "
    "research papers by retrieving matching passages. Rewrite the user's rough text into one precise {target} "
    "that will retrieve the right passages.\n"
    "Rules:\n"
    "1. Keep the user's intent exactly. Do not answer it, do not add facts, names, numbers or claims that are "
    "not in the user's text.\n"
    "2. Turn vague words into a specific request: say what to find (method, result, number, definition, "
    "limitation, comparison) and the expected answer shape (e.g. a number, a list, a short explanation).\n"
    "3. Keep the user's key terms, and add standard research synonyms only when they help retrieval.\n"
    "4. Name a paper only if the user named it or it is in the paper list.\n"
    "5. {shape}\n"
    "6. At most {max_words} words. Plain text, no markdown.\n"
    'Reply with ONLY this JSON: {{"refined": "<the rewritten {target}>", "why": "<one short sentence on what '
    'you made clearer>"}}'
)

_TARGETS = {
    MODE_ASK: ("question", "Write a single question ending with a question mark."),
    MODE_COMPARE: (
        "comparison topic",
        "Write a short topic phrase naming the aspects to compare across papers (e.g. methods, datasets, results, "
        "limitations), not a question.",
    ),
}


@dataclass
class RefinedPrompt:
    """Result of refining one user prompt."""

    original: str
    refined: str
    rationale: str
    mode: str
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_refine_prompts(text: str, mode: str, documents: Sequence[str]) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the refiner."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}.")
    target, shape = _TARGETS[mode]
    system = REFINE_SYSTEM_PROMPT.format(target=target, shape=shape, max_words=config.REFINE_MAX_WORDS)
    papers = ", ".join(documents) if documents else "none uploaded yet"
    user = f"Papers in this chat: {papers}\n\nUser's rough text: {text.strip()}"
    return system, user


def _clean(value: Any) -> str:
    """Normalise one reply field: strip quotes, labels and extra whitespace."""
    cleaned = _LABEL_RE.sub("", _WHITESPACE_RE.sub(" ", str(value or "")).strip())
    return cleaned.strip().strip('"').strip()


def parse_refiner_reply(reply: str) -> tuple[str, str]:
    """Extract ``(refined, rationale)`` from the model reply.

    Accepts the requested JSON (optionally inside a ```json fence or surrounded by text); falls back to
    treating the first non-empty line as the refined text when the model ignored the format.
    """
    body = _FENCE_RE.sub("", reply.strip())
    match = _JSON_OBJECT_RE.search(body)
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = None
        if isinstance(data, dict) and _clean(data.get("refined")):
            return _clean(data.get("refined")), _clean(data.get("why"))
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    return _clean(first_line), ""


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "…"


def refine_prompt(llm: BaseLLM, text: str, mode: str = MODE_ASK, documents: Sequence[str] = ()) -> RefinedPrompt:
    """Rewrite ``text`` into a sharper question (``ask``) or synthesis topic (``compare``).

    Raises:
        ValueError: unknown mode or empty text.
        LLMUnavailableError / LLMGenerationError: from the LLM backend, or when the reply is unusable.
    """
    original = _WHITESPACE_RE.sub(" ", text).strip()
    if not original:
        raise ValueError("Type a question first, then press Refine.")
    system, user = build_refine_prompts(original, mode, documents)
    reply = llm.generate(system, user, max_tokens=config.REFINE_MAX_TOKENS)
    refined, rationale = parse_refiner_reply(reply)
    if not refined or refined.upper() == config.INSUFFICIENT_MARKER:
        raise LLMGenerationError("The model could not refine this text. Try rephrasing it, or send it as it is.")
    # A little headroom over the instruction so a slightly long reply is kept intact.
    refined = _truncate_words(refined, config.REFINE_MAX_WORDS + config.REFINE_WORD_SLACK)
    changed = refined.casefold() != original.casefold()
    if not rationale:
        rationale = "Made the request more specific." if changed else "Your question was already clear."
    return RefinedPrompt(original=original, refined=refined, rationale=rationale, mode=mode, changed=changed)
