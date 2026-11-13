"""Compare retrieval strategies on a verified question set (no LLM needed).

Modes, each returning the top-k chunks for every question:
  dense-only               FAISS cosine search only (the "naive RAG" retriever)
  bm25-only                sparse keyword search only
  hybrid (RRF)             dense + BM25 fused with Reciprocal Rank Fusion
  hybrid + rerank          RRF candidates reordered by the cross-encoder
  hybrid + rerank + floor  same, with weak chunks (relevance < RELEVANCE_FLOOR) dropped

Each question in data/eval/retrieval_qa.json has ``evidence`` phrases; a retrieved
chunk is a hit if it contains ANY of them (punctuation, case and line-break
hyphenation are ignored). Metrics: hit@k (evidence in any top-k chunk) and MRR@k
(1 / rank of the first hit). Questions whose evidence is in NO indexed chunk are
reported as unreachable and excluded, so they cannot blame the retriever.

The corpus is rebuilt in memory from ``--docs`` (default ``data/papers``) on every run.

Usage:  python scripts/compare_retrievers.py [--docs FOLDER] [--qa PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

EXIT_OK, EXIT_FAILURE = 0, 1
QA_PATH = config.EVAL_DIR / "retrieval_qa.json"
OUTPUT_PATH = config.EVAL_DIR / "retriever_comparison.json"
MODES = ("dense-only", "bm25-only", "hybrid (RRF)", "hybrid + rerank", "hybrid + rerank + floor")
KINDS = ("exact", "paraphrase")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def squash(text: str) -> str:
    """Lower-case and drop everything but letters and digits (robust to hyphenation)."""
    return _NON_ALNUM_RE.sub("", text.lower())


def first_hit_rank(texts: Sequence[str], evidence: Sequence[str]) -> Optional[int]:
    """1-based rank of the first text containing any evidence phrase, else None."""
    needles = [squash(e) for e in evidence if squash(e)]
    for rank, text in enumerate(texts, start=1):
        flat = squash(text)
        if any(needle in flat for needle in needles):
            return rank
    return None


def summarize(outcomes: Sequence[tuple[str, Optional[int]]]) -> dict[str, dict[str, Optional[float]]]:
    """Aggregate ``(kind, first_hit_rank)`` pairs into hit-rate and MRR per kind and overall."""
    summary: dict[str, dict[str, Optional[float]]] = {}
    for label in ("all",) + KINDS:
        ranks = [rank for kind, rank in outcomes if label == "all" or kind == label]
        if not ranks:
            summary[label] = {"n": 0, "hit": None, "mrr": None}
            continue
        summary[label] = {
            "n": len(ranks),
            "hit": round(sum(1 for r in ranks if r is not None) / len(ranks), 3),
            "mrr": round(sum(1 / r for r in ranks if r is not None) / len(ranks), 3),
        }
    return summary


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the retrieval question set."""
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    for position, item in enumerate(items):
        if not item.get("question") or not item.get("evidence") or item.get("kind") not in KINDS:
            raise ValueError(f"Item {position} needs 'question', 'evidence' and kind in {KINDS}.")
    return items


def ranked_texts(assistant: Any, question: str, k: int) -> dict[str, list[tuple[str, str]]]:
    """Return ``{mode: [(doc_id, chunk_text), ...]}`` for one question."""
    retriever, store = assistant.retriever, assistant.store
    qvec = assistant.embedder.embed_query(question)

    def pack(ids: Sequence[int]) -> list[tuple[str, str]]:
        return [(store.get_chunk(i).doc_id, store.get_chunk(i).text) for i in ids]

    fused = retriever._candidates(  # diagnostic use of the internal stage
        question, qvec, config.DENSE_TOP_K, config.BM25_TOP_K, config.FUSED_TOP_K
    )
    reranked = retriever.retrieve(question, min_relevance=0.0)
    floored = retriever.retrieve(question)
    return {
        "dense-only": pack([i for i, _ in retriever.dense_search(qvec, k)]),
        "bm25-only": pack([i for i, _ in retriever.sparse_search(question, k)]),
        "hybrid (RRF)": pack([c.store_index for c in fused][:k]),
        "hybrid + rerank": pack([c.store_index for c in reranked]),
        "hybrid + rerank + floor": pack([c.store_index for c in floored]),
        "_junk": [("", f"{c.relevance:.4f}") for c in reranked],
        "_kept": [("", str(len(floored)))],
    }


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare retrieval strategies without an LLM.")
    parser.add_argument("--docs", type=Path, default=config.PAPERS_DIR, help="Folder of PDFs to index")
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    args = parser.parse_args(argv)
    try:
        items = load_questions(args.qa)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    from app.pipelines.rag import assistant_from_folder

    try:
        assistant, _ = assistant_from_folder(args.docs)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    if len(assistant.store) == 0:
        print(f"Error: no usable PDFs in {args.docs}.", file=sys.stderr)
        return EXIT_FAILURE
    print(f"Corpus: {len(assistant.documents())} documents, {len(assistant.store)} chunks")
    all_texts = [c.text for c in assistant.store.all_chunks()]

    outcomes: dict[str, list[tuple[str, Optional[int]]]] = {mode: [] for mode in MODES}
    unreachable: list[str] = []
    misses: dict[str, list[str]] = {mode: [] for mode in MODES}
    junk_values: list[float] = []
    kept_counts: list[int] = []
    for number, item in enumerate(items, start=1):
        if first_hit_rank(all_texts, item["evidence"]) is None:
            unreachable.append(item["question"])
            continue
        print(f"[{number}/{len(items)}] {item['kind']:<10} {item['question'][:70]}")
        per_mode = ranked_texts(assistant, item["question"], args.k)
        junk_values += [float(text) for _, text in per_mode["_junk"]]
        kept_counts.append(int(per_mode["_kept"][0][1]))
        for mode in MODES:
            rank = first_hit_rank([text for _, text in per_mode[mode]], item["evidence"])
            outcomes[mode].append((item["kind"], rank))
            if rank is None:
                found = sorted({doc for doc, _ in per_mode[mode]})
                misses[mode].append(f"{item['question'][:60]} (expected {item.get('doc')}, got {found})")

    table = {mode: summarize(outcomes[mode]) for mode in MODES}
    print("\n" + "=" * 96)
    print(f"{'mode':<26}{'ALL hit':>9}{'MRR':>7}   {'EXACT hit':>10}{'MRR':>7}   {'PARA hit':>9}{'MRR':>7}")
    for mode in MODES:
        s = table[mode]
        fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"  # noqa: E731
        print(f"{mode:<26}{fmt(s['all']['hit']):>9}{fmt(s['all']['mrr']):>7}   {fmt(s['exact']['hit']):>10}"
              f"{fmt(s['exact']['mrr']):>7}   {fmt(s['paraphrase']['hit']):>9}{fmt(s['paraphrase']['mrr']):>7}")
    print("=" * 96)
    weak = sum(1 for v in junk_values if v < config.RELEVANCE_FLOOR)
    print(f"questions scored: {table[MODES[0]]['all']['n']}; unreachable (evidence not in any chunk): {len(unreachable)}")
    if junk_values:
        print(f"without the floor, {weak}/{len(junk_values)} final chunks ({weak / len(junk_values):.0%}) had "
              f"relevance < {config.RELEVANCE_FLOOR}; with it, the context averages "
              f"{sum(kept_counts) / len(kept_counts):.1f} chunks")
    for question in unreachable:
        print(f"  UNREACHABLE: {question}")
    for mode in ("hybrid + rerank + floor",):
        for line in misses[mode]:
            print(f"  MISS [{mode}]: {line}")

    report = {"k": args.k, "modes": table, "unreachable": unreachable, "misses": misses,
              "weak_chunk_rate": round(weak / len(junk_values), 3) if junk_values else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
