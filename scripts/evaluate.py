"""RAGAS-style local evaluation harness (no external APIs).

Reads ``data/eval/golden_qa.json`` (a list of ``{"question", "expected_answer"}``),
runs each question through the full pipeline and writes ``data/eval/results.json``.

Metrics (all in [0, 1]):
* faithfulness       - local LLM judge scores 0-5 how well the answer's claims are
                       supported by the retrieved context; normalised by 5.
* context_precision  - average precision over the retrieved chunks, where a chunk
                       is relevant if its embedding cosine to the expected answer
                       reaches CONTEXT_PRECISION_SIM_THRESHOLD.
* context_recall     - share of expected-answer sentences whose best cosine to any
                       retrieved chunk reaches CONTEXT_RECALL_SIM_THRESHOLD.
* answer_relevancy   - embedding cosine between generated and expected answers.

Abstained answers get answer_relevancy 0.0 and no faithfulness score (there is
nothing to judge); the abstention rate is reported separately.

The corpus is rebuilt in memory from ``--docs`` (default ``data/papers``) on every run.

Usage:  python scripts/evaluate.py [--docs FOLDER] [--golden PATH] [--output PATH] [--limit N] [--no-judge]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.core.errors import LLMGenerationError, LLMUnavailableError, ThesoraError  # noqa: E402
from app.generation.llm import BaseLLM  # noqa: E402
from app.generation.verifier import CITATION_RE  # noqa: E402

logger = logging.getLogger("thesora.evaluate")

EXIT_OK = 0
EXIT_FAILURE = 1
METRICS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SCORE_PATTERNS = (
    re.compile(r"SCORE\s*[:=]\s*([0-5](?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\b([0-5](?:\.\d+)?)\s*(?:/|out of)\s*5\b", re.IGNORECASE),
    re.compile(r"^\s*([0-5](?:\.\d+)?)\s*$"),
)
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Given CONTEXT and ANSWER, rate from 0 to 5 how fully every claim in the ANSWER "
    "is supported by the CONTEXT (0 = nothing supported or contradicted, 5 = every claim fully supported). "
    "Reply with exactly: SCORE: <number from 0 to 5>"
)
SAMPLE_GOLDEN = [
    {"question": "What is Reciprocal Rank Fusion and why is it used in hybrid retrieval?",
     "expected_answer": "Reciprocal Rank Fusion combines several ranked lists by summing 1/(k+rank) for each document, "
                        "so results from sparse and dense retrievers can be merged without calibrating their scores."},
    {"question": "Why do RAG systems rerank retrieved passages with a cross-encoder?",
     "expected_answer": "A cross-encoder reads the query and passage together, which scores relevance more accurately "
                        "than embedding similarity and improves the precision of the final context."},
    {"question": "How does retrieval-augmented generation reduce hallucination?",
     "expected_answer": "It grounds the language model in retrieved documents so answers are based on external evidence "
                        "instead of only the model's parametric memory."},
    {"question": "What is the difference between sparse and dense retrieval?",
     "expected_answer": "Sparse retrieval such as BM25 matches exact terms, while dense retrieval compares embedding "
                        "vectors to capture semantic similarity."},
    {"question": "Why should a RAG system abstain when evidence is weak?",
     "expected_answer": "When retrieved passages are not relevant, answering would invite fabricated claims, so the "
                        "system should say the documents lack enough information."},
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (already normalised or not) vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def split_sentences(text: str) -> list[str]:
    """Split text into sentences long enough to be meaningful."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if len(s.strip()) >= config.EVAL_MIN_SENTENCE_CHARS]


def parse_judge_score(reply: str) -> Optional[float]:
    """Parse a 0-5 score out of a judge reply; None if nothing parseable."""
    for pattern in _SCORE_PATTERNS:
        match = pattern.search(reply)
        if match:
            value = float(match.group(1))
            if 0.0 <= value <= config.FAITHFULNESS_MAX_SCORE:
                return value
    return None


def context_precision(similarities: Sequence[float], threshold: float = config.CONTEXT_PRECISION_SIM_THRESHOLD) -> float:
    """Average precision of ranked chunks judged relevant by similarity to the expected answer."""
    hits, precisions = 0, []
    for rank, similarity in enumerate(similarities, start=1):
        if similarity >= threshold:
            hits += 1
            precisions.append(hits / rank)
    return float(np.mean(precisions)) if precisions else 0.0


def context_recall(sentence_vectors: np.ndarray, chunk_vectors: np.ndarray,
                   threshold: float = config.CONTEXT_RECALL_SIM_THRESHOLD) -> float:
    """Fraction of expected-answer sentences covered by at least one chunk."""
    if len(sentence_vectors) == 0 or len(chunk_vectors) == 0:
        return 0.0
    covered = sum(1 for s in sentence_vectors if max(cosine(s, c) for c in chunk_vectors) >= threshold)
    return covered / len(sentence_vectors)


def judge_faithfulness(llm: BaseLLM, answer: str, contexts: Sequence[str]) -> Optional[float]:
    """Ask the local LLM to rate faithfulness; returns a value in [0, 1] or None."""
    context = "\n\n".join(contexts)[: config.JUDGE_CONTEXT_CHARS]
    clean_answer = CITATION_RE.sub("", answer)
    prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{clean_answer}\n\nRate the faithfulness."
    try:
        reply = llm.generate(JUDGE_SYSTEM_PROMPT, prompt, config.JUDGE_MAX_TOKENS)
    except (LLMUnavailableError, LLMGenerationError) as exc:
        logger.warning("Faithfulness judge failed: %s", exc)
        return None
    score = parse_judge_score(reply)
    if score is None:
        logger.warning("Unparseable judge reply skipped: %r", reply[:80])
        return None
    return score / config.FAITHFULNESS_MAX_SCORE


def load_golden(path: Path) -> list[dict[str, str]]:
    """Load and validate the golden set, creating a sample file if it is missing."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(SAMPLE_GOLDEN, indent=2), encoding="utf-8")
        logger.warning("Golden set not found; wrote a sample to %s. Edit it to match your corpus.", path)
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read golden set {path}: {exc}") from exc
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    for position, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(f"Item {position} in {path} needs a 'question'.")
        if not item.get("should_abstain") and not item.get("expected_answer"):
            raise ValueError(f"Item {position} in {path} needs 'expected_answer' (or 'should_abstain': true).")
    return items


def evaluate_item(assistant: Any, item: dict[str, str], use_judge: bool) -> dict[str, Any]:
    """Run one golden item and compute all metrics.

    Optional golden fields: ``should_abstain`` (the system must refuse; answer metrics are
    skipped) and ``expected_doc`` (a document id that must appear among the retrieved sources).
    """
    should_abstain = bool(item.get("should_abstain", False))
    expected_doc = item.get("expected_doc")
    answer = assistant.ask(item["question"])
    texts = [s.text for s in answer.sources]
    metrics: dict[str, Optional[float]] = {name: None for name in METRICS}
    if not should_abstain:
        embedder = assistant.embedder
        expected = item["expected_answer"]
        expected_vec = embedder.embed([expected])[0]
        chunk_vecs = embedder.embed(texts) if texts else np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        sentence_vecs = embedder.embed(split_sentences(expected) or [expected])
        metrics["context_precision"] = context_precision([cosine(expected_vec, v) for v in chunk_vecs])
        metrics["context_recall"] = context_recall(sentence_vecs, chunk_vecs)
        metrics["answer_relevancy"] = 0.0
        if not answer.abstained:
            generated_vec = embedder.embed([CITATION_RE.sub("", answer.answer)])[0]
            metrics["answer_relevancy"] = max(0.0, cosine(generated_vec, expected_vec))
            if use_judge:
                metrics["faithfulness"] = judge_faithfulness(assistant.llm, answer.answer, texts)
    return {
        "question": item["question"],
        "expected_answer": item.get("expected_answer", ""),
        "generated_answer": answer.answer,
        "abstained": answer.abstained,
        "should_abstain": should_abstain,
        "abstain_correct": (answer.abstained if should_abstain else None),
        "retrieval_hit": ((expected_doc in {s.doc_id for s in answer.sources}) if expected_doc else None),
        "confidence": answer.confidence,
        "unverified_citations": answer.unverified_citations,
        "metrics": {k: (None if v is None else round(float(v), 4)) for k, v in metrics.items()},
        "latency_s": answer.latency_s,
    }


def error_row(item: dict[str, str], message: str) -> dict[str, Any]:
    """Result row for a question whose generation failed (excluded from all averages)."""
    return {
        "question": item["question"],
        "expected_answer": item.get("expected_answer", ""),
        "generated_answer": "",
        "abstained": False,
        "should_abstain": bool(item.get("should_abstain", False)),
        "abstain_correct": None,
        "retrieval_hit": None,
        "confidence": 0.0,
        "unverified_citations": [],
        "metrics": {name: None for name in METRICS},
        "latency_s": 0.0,
        "error": message,
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each metric over rows where it is defined, plus abstention, retrieval and error stats."""
    scored = [r for r in rows if not r.get("error")]
    summary: dict[str, Any] = {}
    for name in METRICS:
        values = [r["metrics"][name] for r in scored if r["metrics"].get(name) is not None]
        summary[name] = round(float(np.mean(values)), 4) if values else None
    summary["abstention_rate"] = round(sum(1 for r in scored if r["abstained"]) / len(scored), 4) if scored else 0.0
    must_abstain = [r["abstain_correct"] for r in scored if r.get("abstain_correct") is not None]
    summary["abstention_accuracy"] = round(sum(must_abstain) / len(must_abstain), 4) if must_abstain else None
    hits = [r["retrieval_hit"] for r in scored if r.get("retrieval_hit") is not None]
    summary["retrieval_hit_rate"] = round(sum(hits) / len(hits), 4) if hits else None
    summary["errors"] = len(rows) - len(scored)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Evaluate Thesora on a golden QA set.")
    parser.add_argument("--docs", type=Path, default=config.PAPERS_DIR, help="Folder of PDFs to evaluate on")
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_QA_PATH)
    parser.add_argument("--output", type=Path, default=config.EVAL_RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N questions")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM faithfulness judge")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        golden = load_golden(args.golden)
        if args.limit > 0:
            golden = golden[: args.limit]
        from app.pipelines.rag import assistant_from_folder

        logger.info("Building an in-memory corpus from %s ...", args.docs)
        assistant, ingested = assistant_from_folder(args.docs)
        if len(assistant.store) == 0:
            print(f"Error: no usable PDFs in {args.docs}. Put your papers there or pass --docs.", file=sys.stderr)
            return EXIT_FAILURE
        logger.info("Corpus: %d documents, %d chunks", len(assistant.documents()), len(assistant.store))
        rows = []
        for number, item in enumerate(golden, start=1):
            logger.info("[%d/%d] %s", number, len(golden), item["question"])
            try:
                rows.append(evaluate_item(assistant, item, use_judge=not args.no_judge))
            except LLMUnavailableError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return EXIT_FAILURE
            except (LLMGenerationError, ThesoraError) as exc:
                logger.error("Question failed and was skipped: %s", exc)
                rows.append(error_row(item, str(exc)))
    except (ValueError, FileNotFoundError, ThesoraError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": getattr(assistant.llm, "model", "unknown"),
        "documents": [d["doc_id"] for d in assistant.documents()],
        "questions": len(rows),
        "aggregate": aggregate(rows),
        "per_question": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Wrote {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
