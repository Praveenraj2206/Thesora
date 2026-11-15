"""N13 stress test (metric definitions from ALCE, MIT): noisy, SSLI look-alike and counterfactual passages.

Metrics: citation recall (answer claims that carry a citation), citation precision (citations whose passage
supports the claim, via the word verifier + NLI when enabled), counterfactual-followed rate, distractor-cited rate,
distractor-ranked-first rate (retrieval level).
Usage: python scripts/stress_eval.py [--limit N] [--no-llm]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.core.models import RetrievedChunk  # noqa: E402
from app.generation.llm import SYSTEM_PROMPT, build_ask_prompt  # noqa: E402
from app.generation.trust import sentence_report  # noqa: E402
from app.pipelines.rag import assistant_from_folder, is_insufficient  # noqa: E402

CASES = config.EVAL_DIR / "stress_cases.json"
OUT = config.EVAL_DIR / "stress_results.json"
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z]{4,}")


def alter_numbers(text: str) -> str:
    return _NUM.sub(lambda m: str(int(float(m.group(0))) + 7), text)


def build(assistant, question: str, group: str, rng: random.Random) -> tuple[list, set[int]]:
    """Returns (passages for the prompt, indexes of distractor passages)."""
    true = assistant.retriever.retrieve(question)[:1]
    if not true:
        return [], set()
    chunks = list(getattr(assistant.store, "_chunks", []))
    base = true[0]
    others = [c for c in chunks if c.doc_id != base.chunk.doc_id]
    if group == "noisy":
        extra = rng.sample(others, min(3, len(others)))
    elif group == "ssli":
        words = set(_WORD.findall(base.chunk.text.lower()))
        extra = sorted(others, key=lambda c: -len(words & set(_WORD.findall(c.text.lower()))))[:3]
    else:
        changed = RetrievedChunk(chunk=type(base.chunk)(**{**base.chunk.__dict__, "text": alter_numbers(base.chunk.text)}),
                                 store_index=base.store_index, relevance=base.relevance)
        return [changed], set()
    items = [base] + [RetrievedChunk(chunk=c, store_index=-1, relevance=0.0) for c in extra]
    rng.shuffle(items)
    return items, {i for i, it in enumerate(items, start=1) if it.store_index == -1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="retrieval-level checks only")
    args = parser.parse_args()
    golden = json.loads(config.GOLDEN_QA_PATH.read_text(encoding="utf-8"))
    picks = json.loads(CASES.read_text(encoding="utf-8"))["questions"][: args.limit]
    assistant, _ = assistant_from_folder()
    rng = random.Random(7)
    stats = {g: {"cases": 0, "abstained": 0, "claims": 0, "cited_claims": 0, "citations": 0, "supported": 0,
                 "distractor_cited": 0, "followed_counterfactual": 0, "distractor_ranked_first": 0}
             for g in ("noisy", "ssli", "counterfactual")}
    for qi in picks:
        question = golden[qi]["question"]
        for group, st in stats.items():
            passages, distractors = build(assistant, question, group, rng)
            if not passages:
                continue
            st["cases"] += 1
            if distractors:  # retrieval level: does the reranker put a distractor above the true passage?
                logits = assistant.retriever._reranker.logits(question, [p.chunk.text for p in passages]) \
                    if getattr(assistant.retriever, "_reranker", None) else None
                if logits is not None and (max(range(len(logits)), key=logits.__getitem__) + 1) in distractors:
                    st["distractor_ranked_first"] += 1
            if args.no_llm:
                continue
            raw = assistant.llm.generate(SYSTEM_PROMPT, build_ask_prompt(question, passages), config.ASK_MAX_TOKENS)
            if is_insufficient(raw):
                st["abstained"] += 1
                continue
            report = sentence_report(raw, {n: p.chunk.text for n, p in enumerate(passages, start=1)}, assistant._nli())
            st["claims"] += len(report)
            st["cited_claims"] += sum(1 for r in report if r["citations"])
            st["citations"] += sum(len(r["citations"]) for r in report)
            st["supported"] += sum(len(r["citations"]) for r in report if r["status"] == "verified")
            st["distractor_cited"] += int(any(set(r["citations"]) & distractors for r in report))
            if group == "counterfactual":
                altered = set(_NUM.findall(passages[0].chunk.text)) - set(_NUM.findall(golden[qi].get("expected_answer", "")))
                st["followed_counterfactual"] += int(bool(altered & set(_NUM.findall(raw))))
    summary = {}
    for group, st in stats.items():
        n = max(1, st["cases"])
        summary[group] = {
            "cases": st["cases"], "abstention_rate": round(st["abstained"] / n, 3),
            "citation_recall": round(st["cited_claims"] / st["claims"], 3) if st["claims"] else None,
            "citation_precision": round(st["supported"] / st["citations"], 3) if st["citations"] else None,
            "distractor_cited_rate": round(st["distractor_cited"] / n, 3) if not args.no_llm else None,
            "counterfactual_followed_rate": round(st["followed_counterfactual"] / n, 3)
            if group == "counterfactual" and not args.no_llm else None,
            "distractor_ranked_first_rate": round(st["distractor_ranked_first"] / n, 3) if group != "counterfactual" else None,
        }
    OUT.write_text(json.dumps({"no_llm": args.no_llm, "nli": assistant._nli() is not None, "groups": summary}, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
