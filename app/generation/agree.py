"""N12 "Do the papers agree?" — extractive positions per paper + conflicts graded by severity (idea: PaperQA2).

Severity 1 = wording differs (hidden by default), 2 = numbers differ for the same metric words, 3 = explicit
contradiction (NLI, only when the meaning model is available). Positions are copied sentences, never generated.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Mapping, Optional, Sequence

from app import config

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z][a-z-]{2,}")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "than", "more", "less", "which"}


def position(passage: str, topic: str) -> str:
    """The passage sentence sharing most words with the topic (extractive, no LLM)."""
    words = set(_WORD.findall(topic.lower())) - _STOP
    sentences = [s.strip() for s in _SENT.split(" ".join(passage.split())) if len(s.split()) >= 4] or [passage]
    return max(sentences, key=lambda s: len(words & set(_WORD.findall(s.lower()))))


def numbers_with_context(sentence: str) -> dict[str, set[str]]:
    """Map each metric word near a number to the numbers it appears with."""
    tokens = sentence.lower().split()
    found: dict[str, set[str]] = {}
    for i, tok in enumerate(tokens):
        for num in _NUM.findall(tok):
            window = tokens[max(0, i - config.AGREE_NUMBER_WINDOW): i + config.AGREE_NUMBER_WINDOW + 1]
            for w in window:
                for word in _WORD.findall(w):
                    if word not in _STOP:
                        found.setdefault(word, set()).add(num)
    return found


def compare(rows: Sequence[Mapping[str, Any]], nli: Optional[Any] = None) -> list[dict[str, Any]]:
    """Pairwise conflicts between papers' positions."""
    conflicts = []
    for a, b in combinations(rows, 2):
        na, nb = numbers_with_context(a["position"]), numbers_with_context(b["position"])
        shared = [w for w in set(na) & set(nb) if na[w] != nb[w]]
        severity, reason = 1, "Different wording"
        if shared:
            severity, reason = 2, f"Numbers differ for: {', '.join(sorted(shared)[:3])}"
        if nli is not None:
            if nli.judge(a["position"], b["position"]) == "FAIL" or nli.judge(b["position"], a["position"]) == "FAIL":
                severity, reason = 3, "The statements contradict each other"
        conflicts.append({"papers": [a["doc_id"], b["doc_id"]], "severity": severity, "reason": reason,
                          "quotes": [a["position"], b["position"]]})
    return sorted(conflicts, key=lambda c: -c["severity"])
