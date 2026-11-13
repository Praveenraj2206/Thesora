"""Check that Thesora's features work against your real papers (default: data/papers).

Run from the project root with the venv active (backend does not need to be
running for the in-process checks; it does for the API checks):

    python scripts/verify_backend.py                 # everything
    python scripts/verify_backend.py --skip-api      # no running backend needed
    python scripts/verify_backend.py --skip-eval     # skip the slower evaluation run

Feature checks
  F1 Hybrid retrieval   BM25 + dense lists are produced and RRF-fused correctly.
  F2 Reranking          Cross-encoder (not the fallback) orders the final top-k.
  F3 Citation check     A faithful claim verifies; a fabricated claim is flagged.
  F4 Abstention         Off-topic questions abstain WITHOUT calling the LLM; every
                        answer carries a confidence in [0, 1].
  F5 Evaluation         scripts/evaluate.py runs and writes valid results.json.
  API                   /health, sessions, upload limit (max 10 per request),
                        chat isolation (a new chat sees no old PDFs),
                        /ask (abstain + real answer), /synthesize and /refine.

F3/F4 use a stub LLM, so they prove the pipeline logic even if Ollama is down.
Exit code is 1 if any check FAILs (SKIP and INFO never fail the run).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from app import config  # noqa: E402
from app.generation.llm import BaseLLM  # noqa: E402
from app.retrieval.retriever import reciprocal_rank_fusion  # noqa: E402

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
OFF_TOPIC_QUESTION = "What is the capital of France?"
FABRICATED_CLAIM = "The method was trained on ten million satellite images using sixteen GPUs for several months [1]."
FAITHFUL_WORDS = 25
HTTP_UNAVAILABLE = 503
DEFAULT_TIMEOUT_S = 900
EVAL_TIMEOUT_S = 3600
EVAL_LIMIT = 2
FALLBACK_QUERY = "What is the main finding of the paper?"
SYNTH_TOPIC = "deja vu"
EXIT_OK, EXIT_FAILURE = 0, 1


@dataclass
class Result:
    """One line of the final report."""

    feature: str
    check: str
    status: str
    detail: str


class ScriptedLLM(BaseLLM):
    """Stub LLM returning a fixed reply (no Ollama needed)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 0) -> str:
        self.calls += 1
        return self.reply


def safe(feature: str, check: str, func: Callable[[], Result]) -> Result:
    """Run a check; any unexpected exception becomes a FAIL with the error text."""
    try:
        return func()
    except Exception as exc:  # report, never crash the whole run
        return Result(feature, check, FAIL, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------- F1
def check_hybrid(assistant: Any, query: str) -> Result:
    """F1: dense + BM25 lists exist and the candidates equal an independent RRF fusion."""
    retriever = assistant.retriever
    qvec = assistant.embedder.embed_query(query)
    dense = retriever.dense_search(qvec, config.DENSE_TOP_K)
    sparse = retriever.sparse_search(query, config.BM25_TOP_K)
    expected = reciprocal_rank_fusion(
        [[i for i, _ in dense], [i for i, _ in sparse]], top_k=config.FUSED_TOP_K
    )
    candidates = retriever._candidates(  # diagnostic use of the internal stage
        query, qvec, config.DENSE_TOP_K, config.BM25_TOP_K, config.FUSED_TOP_K
    )
    dense_ids, sparse_ids = {i for i, _ in dense}, {i for i, _ in sparse}
    problems = []
    if not dense:
        problems.append("dense search returned nothing")
    if not sparse:
        problems.append("BM25 returned nothing (query has no term overlap?)")
    if len(dense) > config.DENSE_TOP_K or len(sparse) > config.BM25_TOP_K:
        problems.append("a list is longer than its top_k")
    if not 0 < len(candidates) <= config.FUSED_TOP_K:
        problems.append(f"fused size {len(candidates)} not in 1..{config.FUSED_TOP_K}")
    if [c.store_index for c in candidates] != [i for i, _ in expected]:
        problems.append("fused order differs from an independent RRF computation")
    detail = (
        f"dense={len(dense)} bm25={len(sparse)} overlap={len(dense_ids & sparse_ids)} "
        f"fused={len(candidates)}; BM25 added {len({c.store_index for c in candidates} - dense_ids)} "
        f"chunk(s) dense search alone missed"
    )
    return Result("F1 Hybrid", "BM25 + dense + RRF fusion", FAIL if problems else PASS,
                  "; ".join(problems) if problems else detail)


# ----------------------------------------------------------------------------- F2
def check_rerank(assistant: Any, query: str) -> Result:
    """F2: final chunks come from the cross-encoder, sorted, in [0,1], at most top-k."""
    chunks = assistant.retriever.retrieve(query)
    if not chunks:
        return Result("F2 Rerank", "cross-encoder ordering", FAIL, "retrieve() returned no chunks")
    problems = []
    if any(c.relevance_source != "cross-encoder" for c in chunks):
        problems.append("using dense-cosine FALLBACK: the cross-encoder failed to load (check internet/HF cache)")
    relevances = [c.relevance for c in chunks]
    if relevances != sorted(relevances, reverse=True):
        problems.append("not sorted by relevance")
    if not all(0.0 <= r <= 1.0 for r in relevances):
        problems.append("relevance outside [0,1]")
    if len(chunks) > config.FINAL_TOP_K:
        problems.append(f"returned {len(chunks)} > FINAL_TOP_K={config.FINAL_TOP_K}")
    rrf_order = [c.store_index for c in sorted(chunks, key=lambda c: -c.rrf_score)]
    reordered = rrf_order != [c.store_index for c in chunks]
    detail = (f"{len(chunks)} chunks, relevance {relevances[0]:.2f}..{relevances[-1]:.2f}; "
              f"reranker {'changed' if reordered else 'kept'} the RRF order")
    return Result("F2 Rerank", "cross-encoder ordering", FAIL if problems else PASS,
                  "; ".join(problems) if problems else detail)


# ----------------------------------------------------------------------------- F3
def check_citations(assistant: Any, query: str) -> list[Result]:
    """F3: with a stub LLM, a faithful claim verifies and a fabricated one is flagged."""
    chunks = assistant.retriever.retrieve(query)
    if not chunks:
        return [Result("F3 Citations", "setup", FAIL, "no chunks retrieved for the query")]
    words = chunks[0].chunk.text.split()[:FAITHFUL_WORDS]
    original_llm = assistant.llm
    results = []
    try:
        assistant.llm = ScriptedLLM(" ".join(words) + " [1]")
        faithful = assistant.ask(query)
        ok = (not faithful.abstained and faithful.sources and faithful.sources[0].verified is True
              and 1 not in faithful.unverified_citations)
        results.append(Result("F3 Citations", "faithful claim verifies", PASS if ok else FAIL,
                              f"source[1].verified={faithful.sources[0].verified if faithful.sources else None}, "
                              f"match_score={faithful.sources[0].match_score if faithful.sources else None}"))
        assistant.llm = ScriptedLLM(FABRICATED_CLAIM)
        fake = assistant.ask(query)
        flagged = (not fake.abstained and fake.sources and fake.sources[0].verified is False
                   and fake.unverified_citations == [1] and bool(fake.message))
        results.append(Result("F3 Citations", "fabricated claim is flagged", PASS if flagged else FAIL,
                              f"unverified_citations={fake.unverified_citations}, message={fake.message!r}"))
    finally:
        assistant.llm = original_llm
    return results


# ----------------------------------------------------------------------------- F4
def check_abstention(assistant: Any, query: str) -> list[Result]:
    """F4: off-topic abstains without calling the LLM; confidence is always in [0,1]."""
    original_llm = assistant.llm
    results = []
    try:
        stub = ScriptedLLM("this must never be returned")
        assistant.llm = stub
        off = assistant.ask(OFF_TOPIC_QUESTION)
        problems = []
        if not off.abstained:
            problems.append(f"did not abstain (best relevance gave confidence {off.confidence})")
        if stub.calls:
            problems.append("LLM was called even though it should have abstained")
        if off.message != config.ABSTAIN_MESSAGE:
            problems.append(f"unexpected message {off.message!r}")
        if not 0.0 <= off.confidence <= 1.0:
            problems.append("confidence outside [0,1]")
        results.append(Result("F4 Abstain", "off-topic abstains, LLM not called", FAIL if problems else PASS,
                              "; ".join(problems) or f"abstained, confidence={off.confidence}, "
                              f"threshold={config.ABSTAIN_THRESHOLD}"))

        assistant.llm = ScriptedLLM(" ".join(assistant.retriever.retrieve(query)[0].chunk.text.split()[:FAITHFUL_WORDS]) + " [1]")
        on = assistant.ask(query)
        good = (not on.abstained) and 0.0 <= on.confidence <= 1.0
        results.append(Result("F4 Abstain", "on-topic answers with confidence", PASS if good else FAIL,
                              f"abstained={on.abstained}, confidence={on.confidence}"))

        assistant.llm = ScriptedLLM(config.INSUFFICIENT_MARKER)
        marker = assistant.ask(query)
        results.append(Result("F4 Abstain", "model 'INSUFFICIENT_CONTEXT' abstains",
                              PASS if marker.abstained else FAIL, f"abstained={marker.abstained}"))
    finally:
        assistant.llm = original_llm
    return results


# ----------------------------------------------------------------------------- F5
def check_evaluation(golden: Path, docs: Path, skip: bool) -> Result:
    """F5: run evaluate.py on a couple of questions and validate results.json."""
    if skip:
        return Result("F5 Evaluate", "evaluate.py writes results.json", SKIP, "--skip-eval given")
    out = config.EVAL_DIR / "verify_results.json"
    command = [sys.executable, str(Path(__file__).with_name("evaluate.py")), "--no-judge", "--docs", str(docs),
               "--limit", str(EVAL_LIMIT), "--golden", str(golden), "--output", str(out)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=EVAL_TIMEOUT_S)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        return Result("F5 Evaluate", "evaluate.py writes results.json", FAIL, " | ".join(tail))
    report = json.loads(out.read_text(encoding="utf-8"))
    aggregate = report["aggregate"]
    bad = [k for k, v in aggregate.items() if v is not None and not 0.0 <= v <= 1.0]
    if bad or not report["per_question"]:
        return Result("F5 Evaluate", "evaluate.py writes results.json", FAIL, f"invalid values: {bad}")
    return Result("F5 Evaluate", "evaluate.py writes results.json", PASS,
                  f"{report['questions']} questions: {json.dumps(aggregate)}")


# ----------------------------------------------------------------------------- API
def _pdf_payload(paths: list[Path]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (p.name, p.read_bytes(), "application/pdf")) for p in paths]


def api_checks(api: str, query: str, timeout: int, docs: Path) -> list[Result]:
    """Exercise the running FastAPI backend, including sessions and the upload limit."""
    results: list[Result] = []
    try:
        health = requests.get(f"{api}/health", timeout=10).json()
    except requests.exceptions.RequestException as exc:
        return [Result("API", "GET /health", FAIL, f"backend not reachable at {api}: {exc}")]
    ollama = health.get("ollama", {})
    results.append(Result("API", "GET /health", PASS if health.get("status") == "ok" else FAIL,
                          f"ollama reachable={ollama.get('reachable')}, model={ollama.get('model')}, "
                          f"installed={ollama.get('model_installed')}, limits={health.get('limits')}"))

    pdfs = sorted(docs.glob("*.pdf"))[: config.MAX_FILES_PER_REQUEST]
    if not pdfs:
        return results + [Result("API", "upload", FAIL, f"no PDFs in {docs}")]
    session = requests.post(f"{api}/sessions", timeout=10).json()["session_id"]
    base = f"{api}/sessions/{session}"

    too_many = requests.post(f"{base}/ingest", files=_pdf_payload([pdfs[0]] * (config.MAX_FILES_PER_REQUEST + 1)),
                             timeout=timeout)
    results.append(Result("API", f"upload of {config.MAX_FILES_PER_REQUEST + 1} files rejected",
                          PASS if too_many.status_code == 400 else FAIL,
                          f"HTTP {too_many.status_code}: {too_many.json().get('detail', '')[:90]}"))

    up = requests.post(f"{base}/ingest", files=_pdf_payload(pdfs), timeout=timeout)
    body = up.json()
    results.append(Result("API", f"upload {len(pdfs)} PDFs in one request", PASS if up.ok and body.get("ingested") else FAIL,
                          f"HTTP {up.status_code}, ingested={body.get('ingested')}, failed={body.get('failed')}, "
                          f"chunks={sum(r.get('chunks_added', 0) for r in body.get('results', []))}"))

    fresh = requests.post(f"{api}/sessions", timeout=10).json()["session_id"]
    fresh_docs = requests.get(f"{api}/sessions/{fresh}/documents", timeout=10).json()["documents"]
    fresh_ask = requests.post(f"{api}/sessions/{fresh}/ask", json={"question": query}, timeout=timeout).json()
    isolated = fresh_docs == [] and fresh_ask.get("abstained") and not fresh_ask.get("sources")
    results.append(Result("API", "new chat sees no old PDFs/references", PASS if isolated else FAIL,
                          f"documents={fresh_docs}, abstained={fresh_ask.get('abstained')}, "
                          f"sources={len(fresh_ask.get('sources', []))}"))
    requests.delete(f"{api}/sessions/{fresh}", timeout=10)

    off = requests.post(f"{base}/ask", json={"question": OFF_TOPIC_QUESTION}, timeout=timeout)
    body = off.json() if off.ok else {}
    results.append(Result("API", "POST /ask off-topic abstains", PASS if body.get("abstained") else FAIL,
                          f"HTTP {off.status_code}, abstained={body.get('abstained')}, confidence={body.get('confidence')}"))

    on = requests.post(f"{base}/ask", json={"question": query}, timeout=timeout)
    if on.status_code == HTTP_UNAVAILABLE:
        results.append(Result("API", "POST /ask real answer", SKIP, f"HTTP 503 clean error: {on.json().get('detail')}"))
    elif not on.ok:
        results.append(Result("API", "POST /ask real answer", FAIL, f"HTTP {on.status_code}: {on.text[:200]}"))
    else:
        data = on.json()
        sources = data.get("sources", [])
        shape_ok = all({"cited", "verified", "match_score"} <= set(s) for s in sources)
        leak = "</think>" in data.get("answer", "") or "<think>" in data.get("answer", "")
        problems = []
        if not shape_ok:
            problems.append("sources missing cited/verified/match_score")
        if leak:
            problems.append("reasoning leaked into the answer")
        verified = sum(1 for s in sources if s.get("verified") is True)
        cited = sum(1 for s in sources if s.get("cited"))
        results.append(Result("API", "POST /ask real answer", FAIL if problems else PASS,
                              "; ".join(problems) or f"abstained={data['abstained']}, confidence={data['confidence']}, "
                              f"{verified}/{cited} cited sources verified, {data['latency_s']}s"))

    syn = requests.post(f"{base}/synthesize", json={"topic": SYNTH_TOPIC}, timeout=timeout)
    if syn.status_code == HTTP_UNAVAILABLE:
        results.append(Result("API", "POST /synthesize", SKIP, f"HTTP 503 clean error: {syn.json().get('detail')}"))
    elif not syn.ok:
        results.append(Result("API", "POST /synthesize", FAIL, f"HTTP {syn.status_code}: {syn.text[:200]}"))
    else:
        data = syn.json()
        text = data.get("answer", "")
        has_table, has_gaps = "|" in text and "---" in text, "research gaps" in text.lower()
        papers = sorted({s["doc_id"] for s in data.get("sources", [])})
        status = PASS if data.get("abstained") is False and has_table and has_gaps else FAIL
        results.append(Result("API", "POST /synthesize table + gaps", status,
                              f"abstained={data.get('abstained')}, table={has_table}, gaps_section={has_gaps}, "
                              f"papers in sources={papers}"))

    rough = "brain parts deja vu"
    ref = requests.post(f"{base}/refine", json={"text": rough, "mode": "ask"}, timeout=timeout)
    if ref.status_code == HTTP_UNAVAILABLE:
        results.append(Result("API", "POST /refine", SKIP, f"HTTP 503 clean error: {ref.json().get('detail')}"))
    elif not ref.ok:
        results.append(Result("API", "POST /refine", FAIL, f"HTTP {ref.status_code}: {ref.text[:200]}"))
    else:
        data = ref.json()
        refined = data.get("refined", "")
        status = PASS if refined and refined.casefold() != rough and data.get("original") == rough else FAIL
        results.append(Result("API", "POST /refine sharper question", status, f"{rough!r} -> {refined[:80]!r}"))

    requests.delete(base, timeout=10)
    gone = requests.get(f"{base}/documents", timeout=10)
    results.append(Result("API", "ended chat is forgotten", PASS if gone.status_code == 404 else FAIL,
                          f"HTTP {gone.status_code} after DELETE"))
    return results


# ----------------------------------------------------------------------------- main
def default_query(golden: Path) -> str:
    """First golden-set question, so the checks use a question your corpus can answer."""
    try:
        return json.loads(golden.read_text(encoding="utf-8"))[0]["question"]
    except (OSError, ValueError, KeyError, IndexError):
        return FALLBACK_QUERY


def print_report(results: list[Result]) -> int:
    """Print an aligned table and return the exit code."""
    print("\n" + "=" * 100)
    for r in results:
        print(f"[{r.status:<4}] {r.feature:<13} {r.check:<38} {r.detail}")
    print("=" * 100)
    counts = {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, SKIP, INFO)}
    print("  ".join(f"{k}={v}" for k, v in counts.items()))
    return EXIT_FAILURE if counts[FAIL] else EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Verify Thesora's five features.")
    parser.add_argument("--api", default=config.API_URL, help="Backend URL for the API checks")
    parser.add_argument("--query", default=None, help="An answerable question (default: first golden question)")
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_QA_PATH)
    parser.add_argument("--docs", type=Path, default=config.PAPERS_DIR, help="Folder of PDFs to verify against")
    parser.add_argument("--skip-api", action="store_true", help="Skip checks that need the running backend/Ollama")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluate.py run")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="Per-request timeout (seconds)")
    args = parser.parse_args(argv)
    query = args.query or default_query(args.golden)
    print(f"Using query: {query}")

    results: list[Result] = []
    from app.pipelines.rag import assistant_from_folder

    print(f"Building an in-memory corpus from {args.docs} (loads models on first run)...")
    try:
        assistant, _ = assistant_from_folder(args.docs)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    if len(assistant.store) == 0:
        print(f"Error: no usable PDFs in {args.docs}.", file=sys.stderr)
        return EXIT_FAILURE
    print(f"Corpus: {len(assistant.documents())} documents, {len(assistant.store)} chunks.")

    results.append(safe("F1 Hybrid", "BM25 + dense + RRF fusion", lambda: check_hybrid(assistant, query)))
    results.append(safe("F2 Rerank", "cross-encoder ordering", lambda: check_rerank(assistant, query)))
    for name, func in (("F3 Citations", check_citations), ("F4 Abstain", check_abstention)):
        try:
            results.extend(func(assistant, query))
        except Exception as exc:
            results.append(Result(name, "checks", FAIL, f"{type(exc).__name__}: {exc}"))
    results.append(safe("F5 Evaluate", "evaluate.py writes results.json",
                        lambda: check_evaluation(args.golden, args.docs, args.skip_eval)))
    if args.skip_api:
        results.append(Result("API", "HTTP endpoints", SKIP, "--skip-api given"))
    else:
        results.extend(api_checks(args.api.rstrip("/"), query, args.timeout, args.docs))
    return print_report(results)


if __name__ == "__main__":
    sys.exit(main())