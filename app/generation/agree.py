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
_STOP = {"the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "than", "more", "less", "which",
         "per", "each", "using", "used", "based", "about", "into", "over", "our", "their", "its", "has", "have", "had",
         "also", "all", "can", "may", "such", "these", "those", "between", "both", "most", "only", "other"}
_CAPTION = re.compile(r"^(table|figure|fig\.)\s*\d+", re.IGNORECASE)


def _norm(word: str) -> str:
    """'costs' and 'cost', 'questions' and 'question' are the same topic word."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def candidate_sentences(passage: str) -> list[str]:
    """Sentences that can state a position: 6+ words, not a table or figure caption, not a code line."""
    sentences = [x.strip() for x in _SENT.split(" ".join(passage.split()))]
    return [x for x in sentences if len(x.split()) >= 6 and not _CAPTION.match(x) and not x.lstrip().startswith(("#", "|"))]


def position(passage: str, topic: str) -> str:
    """The passage sentence sharing most words with the topic (extractive, no LLM).

    Table and figure captions are skipped (they name a topic but state no position); among equally
    good sentences, one that reports a number is preferred."""
    words = {_norm(w) for w in _WORD.findall(topic.lower())} - _STOP
    sentences = [s.strip() for s in _SENT.split(" ".join(passage.split())) if len(s.split()) >= 4]
    sentences = [s for s in sentences if not _CAPTION.match(s)] or sentences or [passage]

    def score(sentence: str) -> tuple[int, bool]:
        overlap = len(words & {_norm(w) for w in _WORD.findall(sentence.lower())})
        return overlap, bool(_NUM.search(sentence))

    return max(sentences, key=score)


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
                        found.setdefault(_norm(word), set()).add(num)
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
