"""ResearchAssistant: orchestrates ingestion, retrieval, generation and verification.

One ResearchAssistant serves one chat session: it owns that chat's in-memory
VectorStore, while the heavy models (embedder, cross-encoder, LLM client) are
injected and shared between sessions.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

from app import config
from app.core.errors import PDFLoadError
from app.core.models import Answer, RetrievedChunk, Source
from app.embeddings.embedder import Embedder
from app.generation.llm import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    BaseLLM,
    OllamaLLM,
    build_ask_prompt,
    build_synthesis_prompt,
)
from app.generation.verifier import VerificationResult, extract_cited_numbers, verify_citations
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.retriever import HybridRetriever, Reranker, sigmoid
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MARKDOWN_PREFIX = "*#>`\"' \n"

NO_CITATIONS_NOTE = "The answer contains no citations, so none of it could be verified."


def is_insufficient(reply: str) -> bool:
    """True only when the model's reply *is* the insufficient-context marker.

    A mere mention of the marker inside an otherwise normal answer must not
    trigger abstention.
    """
    return reply.lstrip(_MARKDOWN_PREFIX).upper().startswith(config.INSUFFICIENT_MARKER)


def calibrate_relevance(item: RetrievedChunk) -> float:
    """Confidence-only relevance: a gentler sigmoid over the raw logit so scores do not saturate at 1.0.

    Chunks scored without a cross-encoder (dense fallback) keep their cosine relevance.
    """
    if item.rerank_logit is None:
        return item.relevance
    return sigmoid((item.rerank_logit - config.CONFIDENCE_LOGIT_MIDPOINT) / config.CONFIDENCE_LOGIT_SCALE)


def compute_confidence(relevances: Sequence[float], verified_ratio: float, coverage: Optional[float] = None) -> float:
    """Blend calibrated retrieval relevance, verified-source ratio and claim coverage into [0, 1].

    ``coverage`` defaults to ``verified_ratio`` for callers that do not track it.
    An answer with no verified citations can score at most half the relevance weight.
    """
    if not relevances:
        return 0.0
    if coverage is None:
        coverage = verified_ratio
    top = mean(sorted(relevances, reverse=True)[: config.CONFIDENCE_TOP_N])
    score = (
        config.CONFIDENCE_RELEVANCE_WEIGHT * top
        + config.CONFIDENCE_VERIFICATION_WEIGHT * verified_ratio
        + config.CONFIDENCE_COVERAGE_WEIGHT * coverage
    )
    return round(min(1.0, max(0.0, score)), 3)


import uuid  # noqa: E402
from dataclasses import replace  # noqa: E402

from app.ingestion.layout import table_chunks  # noqa: E402  (v2 N1)

from app.generation.viewer import (  # noqa: E402  (v2 N9 / N11)
    EXPLAIN_SYSTEM_PROMPT, build_explain_prompt, build_glossary, expand_query, glossary_hits, merge_glossary,
)
from app.generation.trust import (  # noqa: E402  (v2 batch A)
    REPAIR_SYSTEM_PROMPT, NLIChecker, drop_unsupported, extract_metadata, sentence_report, short_label,
    silent_papers, verify_table,
)


_CITE_NUM = re.compile(r"\[(\d+)\]")
_PART_SPLIT = re.compile(r"(?<=\?)\s+|;\s*|\s+as well as\s+|\s+and also\s+", re.IGNORECASE)


def split_question(question: str) -> list[str]:
    """Cheap heuristic split (no LLM): '?', ';', 'as well as', 'and also'; every part needs PART_MIN_WORDS words."""
    parts = [p.strip(" ,") for p in _PART_SPLIT.split(question.strip()) if p and p.strip(" ,")]
    if len(parts) < 2 or any(len(p.split()) < config.PART_MIN_WORDS for p in parts):
        return [question.strip()]
    return [p if p.endswith("?") else p + "?" for p in parts]


def _offsets(results) -> list[int]:
    out, offset = [], 0
    for r in results:
        out.append(offset)
        offset += 0 if r.abstained else len(r.sources)
    return out



def _merge_overlap(first: str, second: str) -> str:
    """Join two consecutive chunks, dropping the text they share (chunks overlap by CHUNK_OVERLAP)."""
    limit = min(len(first), len(second), config.CHUNK_OVERLAP + 50)
    for size in range(limit, 15, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + " " + second

class ResearchAssistant:
    """High-level API used by the FastAPI app, CLI scripts and the evaluator.

    All collaborators can be injected (used by tests); defaults are built lazily
    so importing this module never loads a model.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[BaseLLM] = None,
        reranker: Optional[Reranker] = None,
        chunker: Optional[DocumentChunker] = None,
        nli: Optional[NLIChecker] = None,
    ) -> None:
        self.nli = nli or NLIChecker()
        self.metadata: dict[str, dict[str, Any]] = {}  # N4, per paper
        self.history: list[dict[str, Any]] = []  # N4 notes export (memory only)
        self.pdf_bytes: dict[str, bytes] = {}  # N9: original PDF bytes, memory only, freed with the chat
        self.glossary: list[dict[str, Any]] = []  # N11
        self.answer_passages: dict[str, list[str]] = {}  # N11: passages behind each answer, by answer_id
        self.embedder = embedder or Embedder()
        self.store = store if store is not None else VectorStore()
        self.retriever = retriever or HybridRetriever(self.store, self.embedder, reranker)
        self.llm: BaseLLM = llm or OllamaLLM()
        self._chunker = chunker
        self._ingest_lock = threading.Lock()

    # ------------------------------------------------------------------ ingestion
    def ingest_pdf(self, path: str | Path, doc_id: Optional[str] = None) -> dict[str, Any]:
        """Ingest one PDF incrementally; documents already present are skipped."""
        pdf_path = Path(path)
        name = doc_id or pdf_path.name
        if self.store.has_document(name):
            return {"document": name, "status": "skipped", "pages": 0, "chunks_added": 0}
        pages = load_pdf(pdf_path, doc_id=name)
        self.metadata[name] = extract_metadata(pdf_path, name)
        self.pdf_bytes[name] = pdf_path.read_bytes()
        self.glossary = merge_glossary(self.glossary, build_glossary(pages, name))
        if self._chunker is None:
            self._chunker = DocumentChunker()
        chunks = self._chunker.chunk_pages(pages)
        if not chunks:
            raise PDFLoadError(f"{pdf_path.name} produced no usable text chunks.")
        sections: dict[int, str] = {}
        for c in chunks:
            sections.setdefault(c.page_number, c.section)
        chunks += table_chunks(pdf_path, name, len(chunks), sections)  # N1
        title = (self.metadata.get(name) or {}).get("title") or name
        for c in chunks:  # N2a: search text carries "Title › Section"; display/citation keep the original text
            c.search_text = f"{title} › {c.section}\n{c.text}" if config.CONTEXT_HEADERS else ""
        embeddings = self.embedder.embed([c.search_text or c.text for c in chunks])
        with self._ingest_lock:
            if self.store.has_document(name):  # lost a race with a concurrent upload
                return {"document": name, "status": "skipped", "pages": 0, "chunks_added": 0}
            self.store.add(chunks, embeddings)
        logger.info("Ingested %s: %d pages, %d chunks", name, len(pages), len(chunks))
        return {"document": name, "status": "ingested", "pages": len(pages), "chunks_added": len(chunks)}

    def ingest_folder(self, folder: str | Path) -> list[dict[str, Any]]:
        """Ingest every ``*.pdf`` in a folder; per-file failures are reported, not raised."""
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}. Put your PDFs there or pass --docs.")
        results: list[dict[str, Any]] = []
        for pdf in sorted(folder_path.glob("*.pdf")):
            try:
                results.append(self.ingest_pdf(pdf))
            except Exception as exc:  # keep batch going; report the failure
                logger.error("Failed to ingest %s: %s", pdf.name, exc)
                results.append({"document": pdf.name, "status": "failed", "error": str(exc)})
        return results

    def documents(self) -> list[dict[str, Any]]:
        """List ingested documents."""
        return self.store.documents()

    def warm_up(self) -> None:
        """Pre-load models so the first request is fast."""
        self.retriever.warm_up()

    # ------------------------------------------------------------------ helpers
    def _sources(self, chunks: Sequence[RetrievedChunk], cited: set[int]) -> list[Source]:
        return [
            Source(
                index=n,
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                page_number=item.chunk.page_number,
                section=item.chunk.section,
                text=item.chunk.text,
                relevance=round(item.relevance, 4),
                cited=n in cited,
                label=short_label(self.metadata.get(item.chunk.doc_id), item.chunk.page_number, item.chunk.section),
            )
            for n, item in enumerate(chunks, start=1)
        ]

    def _abstain(
        self, question: str, message: str, chunks: Sequence[RetrievedChunk], confidence: float, started: float, mode: str
    ) -> Answer:
        answer = Answer(
            question=question,
            answer="",
            abstained=True,
            confidence=round(confidence, 3),
            sources=self._sources(chunks, set()),
            message=message,
            latency_s=round(time.perf_counter() - started, 3),
            mode=mode,
        )
        self.history.append(answer.to_dict())
        return answer

    def _subject(self, doc_id: str) -> str:
        """Words naming a paper (file name, title, first author) - ignored when checking a claim's wording."""
        meta = self.metadata.get(doc_id) or {}
        authors = meta.get("authors") or []
        return " ".join([Path(doc_id).stem.replace("_", " "), meta.get("title") or "",
                         authors[0].split()[-1] if authors else ""])

    def _context_text(self, chunk, rank: int = 1) -> str:
        """N2c: for the top ``CONTEXT_EXPANSION_TOP`` passages, add the next (then previous) passage of the
        same section, up to CHUNK_PARENT_CHARS, so an answer cut at a chunk boundary is not lost.
        Verification uses exactly this text too."""
        if not config.CONTEXT_EXPANSION or rank > config.CONTEXT_EXPANSION_TOP or chunk.chunk_type != "text":
            return chunk.text
        same = {c.chunk_index: c for c in getattr(self.store, "_chunks", [])
                if c.doc_id == chunk.doc_id and c.section == chunk.section and c.chunk_type == "text"}
        text = chunk.text
        for neighbour, after in ((same.get(chunk.chunk_index + 1), True), (same.get(chunk.chunk_index - 1), False)):
            if neighbour is None:
                continue
            merged = _merge_overlap(text, neighbour.text) if after else _merge_overlap(neighbour.text, text)
            if len(merged) <= config.CHUNK_PARENT_CHARS:
                text = merged
        return text

    def _multi_query(self, question: str, chunks: list) -> list:
        """N2b: up to 2 reworded variants from one short LLM call; merged by best relevance. Silent on failure."""
        try:
            reply = self.llm.generate(
                "Rewrite the question in 2 different ways using other words. One per line. No numbering, no answers.",
                question, config.MULTI_QUERY_MAX_TOKENS)
        except Exception:
            return chunks
        merged = {c.chunk.chunk_id: c for c in chunks}
        for variant in [v.strip(" -•0123456789.") for v in reply.splitlines() if len(v.split()) >= 3][: config.MULTI_QUERY_VARIANTS]:
            for c in self.retriever.retrieve(variant):
                if c.chunk.chunk_id not in merged or c.relevance > merged[c.chunk.chunk_id].relevance:
                    merged[c.chunk.chunk_id] = c
        return sorted(merged.values(), key=lambda c: c.relevance, reverse=True)[: config.FINAL_TOP_K]

    def agree(self, topic: str) -> dict[str, Any]:
        """N12: each paper's best passage on the topic, an extractive position, and graded conflicts."""
        from app.generation.agree import compare, position

        topic = topic.strip()
        if not topic:
            raise ValueError("Topic must not be empty.")
        if len(self.store) == 0:
            return {"topic": topic, "rows": [], "conflicts": [], "nli_available": False, "message": config.NO_DOCUMENTS_MESSAGE}
        best: dict[str, Any] = {}
        for c in self.retriever.retrieve_across_documents(topic):
            if re.search(r"reference|bibliograph|works cited", c.chunk.section or "", re.IGNORECASE):
                continue  # a reference list quotes other papers, not this paper's position
            if c.chunk.doc_id not in best or c.relevance > best[c.chunk.doc_id].relevance:
                best[c.chunk.doc_id] = c
        rows = [{"doc_id": d, "label": short_label(self.metadata.get(d), c.chunk.page_number, c.chunk.section),
                 "page": c.chunk.page_number, "section": c.chunk.section, "relevance": round(c.relevance, 3),
                 "position": position(c.chunk.text, topic), "passage": c.chunk.text}
                for d, c in best.items() if c.relevance >= config.RELEVANCE_FLOOR]
        nli = self._nli()
        return {"topic": topic, "rows": rows, "conflicts": compare(rows, nli), "nli_available": nli is not None,
                "message": None if nli else "Meaning check is off: only number conflicts are detected."}

    def _remember(self, passages: list[str]) -> str:
        answer_id = uuid.uuid4().hex
        self.answer_passages[answer_id] = passages[: config.FINAL_TOP_K * 2]
        return answer_id

    def explain(self, answer_id: str, selected: str, follow_up: Optional[str] = None,
                previous: Optional[str] = None) -> dict[str, Any]:
        """N11: plain-English explanation of selected text, using only that answer's passages. Not evidence."""
        passages = self.answer_passages.get(answer_id)
        if passages is None:
            raise ValueError("That answer is no longer available in this chat. Ask the question again.")
        if not selected.strip():
            raise ValueError("Select or paste some text to explain.")
        hits = [dict(g) for g in glossary_hits(selected, self.glossary)]
        text = self.llm.generate(EXPLAIN_SYSTEM_PROMPT, build_explain_prompt(selected, passages, follow_up, previous),
                                 config.EXPLAIN_MAX_TOKENS)
        return {"explanation": text, "glossary": hits, "label": "Plain-English explanation — not evidence"}

    def _nli(self) -> Optional[NLIChecker]:
        return self.nli if self.nli.available else None

    # ------------------------------------------------------------------ ask
    def ask(self, question: str) -> Answer:
        """Answer a question; multi-part questions are answered part by part (N6), abstaining only where unsupported."""
        parts = split_question(question) if config.PARTIAL_ANSWERS else [question.strip()]
        if len(parts) < 2:
            return self._ask_one(question)
        started, mark = time.perf_counter(), len(self.history)
        results = [self._ask_one(p) for p in parts]
        del self.history[mark:]
        texts, sources, sentences, offset = [], [], [], 0
        for i, (part, res) in enumerate(zip(parts, results), start=1):
            if res.abstained:
                texts.append(f"**Part {i}:** {part} — Not found in your papers.")
                continue
            texts.append(f"**Part {i}:** " + _CITE_NUM.sub(lambda m: f"[{int(m.group(1)) + offset}]", res.answer))
            for src in res.sources:
                src.index += offset
                sources.append(src)
            for item in res.sentences:
                sentences.append({**item, "citations": [n + offset for n in item["citations"]]})
            offset += len(res.sources)
        answered = [r for r in results if not r.abstained]
        result = Answer(
            question=question.strip(), answer="\n\n".join(texts), abstained=not answered,
            confidence=round(mean(r.confidence for r in answered), 3) if answered else 0.0, sources=sources,
            unverified_citations=[n for r, o in zip(results, _offsets(results)) for n in (o + u for u in r.unverified_citations)],
            message=None if answered else config.ABSTAIN_MESSAGE, latency_s=round(time.perf_counter() - started, 3),
            mode="ask", sentences=sentences,
            silent_papers=sorted(set.intersection(*(set(r.silent_papers) for r in answered))) if answered else [],
            parts=[{"question": p, "answered": not r.abstained, "abstained": r.abstained} for p, r in zip(parts, results)],
            answer_id=self._remember([t for r in answered for t in self.answer_passages.get(r.answer_id or "", [])]),
        )
        self.history.append(result.to_dict())
        return result

    def _ask_one(self, question: str) -> Answer:
        """Answer a question with hybrid retrieval, reranking, verification and abstention."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        started = time.perf_counter()
        if len(self.store) == 0:
            return self._abstain(question, config.NO_DOCUMENTS_MESSAGE, [], 0.0, started, "ask")
        sparse = expand_query(question, self.glossary)
        chunks = self.retriever.retrieve(question, sparse_query=sparse) if sparse else self.retriever.retrieve(question)
        if config.MULTI_QUERY:
            chunks = self._multi_query(question, chunks)
        best = chunks[0].relevance if chunks else 0.0
        if not chunks or best < config.ABSTAIN_THRESHOLD:
            return self._abstain(question, config.ABSTAIN_MESSAGE, chunks, best, started, "ask")

        prompt_chunks = [replace(c, chunk=replace(c.chunk, text=self._context_text(c.chunk, n)))
                         for n, c in enumerate(chunks, start=1)] if config.CONTEXT_EXPANSION else chunks
        raw = self.llm.generate(SYSTEM_PROMPT, build_ask_prompt(question, prompt_chunks), config.ASK_MAX_TOKENS)
        if is_insufficient(raw):
            return self._abstain(question, config.ABSTAIN_MESSAGE, chunks, 0.0, started, "ask")

        texts = {n: self._context_text(item.chunk, n) for n, item in enumerate(chunks, start=1)}
        subjects = {n: self._subject(item.chunk.doc_id) for n, item in enumerate(chunks, start=1)}
        nli = self._nli()
        verification: VerificationResult = verify_citations(raw, texts, subjects)
        report = sentence_report(raw, texts, nli, subjects)
        repair = None
        if config.SELF_REPAIR and any(r["status"] == "unverified" for r in report):
            before = sum(r["status"] == "unverified" for r in report)
            try:
                retry = self.llm.generate(REPAIR_SYSTEM_PROMPT, build_ask_prompt(question, chunks), config.REPAIR_MAX_TOKENS)
                if not is_insufficient(retry):
                    raw, report = retry, sentence_report(retry, texts, nli, subjects)
            except Exception as exc:  # repair is best effort; keep the first answer
                logger.warning("Self-repair call failed: %s", exc)
            after = sum(r["status"] == "unverified" for r in report)
            raw, removed = drop_unsupported(raw, report)
            report = [r for r in report if r["status"] != "unverified"]
            repair = {"attempted": True, "removed": removed, "before_unverified": before, "after_unverified": after}
            if not raw.strip():
                return self._abstain(question, config.ABSTAIN_MESSAGE, chunks, 0.0, started, "ask")
            verification = verify_citations(raw, texts, subjects)
        sources = self._sources(chunks, set(verification.cited))
        for source in sources:
            check = verification.per_source.get(source.index)
            source.verified = bool(check and check.verified)
            source.match_score = check.match_score if check else None
            source.meaning = next((r["meaning"] for r in report if source.index in r["citations"] and r["meaning"]), None)
        message = None
        if not verification.cited:
            message = NO_CITATIONS_NOTE
        elif verification.unverified:
            message = "Some citations could not be verified against the source text: " + ", ".join(
                f"[{n}]" for n in verification.unverified
            )
        best_by_doc: dict[str, float] = {}
        for c in chunks:
            best_by_doc[c.chunk.doc_id] = max(best_by_doc.get(c.chunk.doc_id, 0.0), c.relevance)
        result = Answer(
            question=question,
            answer=raw,
            abstained=False,
            confidence=compute_confidence([calibrate_relevance(c) for c in chunks], verification.verified_ratio, verification.coverage),
            sources=sources,
            unverified_citations=verification.unverified,
            invalid_citations=verification.invalid,
            message=message,
            latency_s=round(time.perf_counter() - started, 3),
            mode="ask",
            sentences=report,
            silent_papers=silent_papers([d["doc_id"] for d in getattr(self.store, "documents", list)()], best_by_doc),
            repair=repair,
            nli_available=nli is not None,
            answer_id=self._remember([c.chunk.text for c in chunks]),
        )
        self.history.append(result.to_dict())
        return result

    # ------------------------------------------------------------------ synthesize
    def synthesize(self, topic: str) -> Answer:
        """Compare approaches/results across all ingested papers and list research gaps.

        Citation verification is not applied here (table cells combine several
        claims), so ``verified`` is None on every source.
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("Topic must not be empty.")
        started = time.perf_counter()
        if len(self.store) == 0:
            return self._abstain(topic, config.NO_DOCUMENTS_MESSAGE, [], 0.0, started, "synthesize")
        chunks = self.retriever.retrieve_across_documents(topic)
        best = max((c.relevance for c in chunks), default=0.0)
        if not chunks or best < config.ABSTAIN_THRESHOLD:
            return self._abstain(topic, config.ABSTAIN_MESSAGE, chunks, best, started, "synthesize")

        raw = self.llm.generate(SYNTHESIS_SYSTEM_PROMPT, build_synthesis_prompt(topic, chunks), config.SYNTH_MAX_TOKENS)
        if is_insufficient(raw):
            return self._abstain(topic, config.ABSTAIN_MESSAGE, chunks, 0.0, started, "synthesize")
        cited = {n for n in extract_cited_numbers(raw) if 1 <= n <= len(chunks)}
        invalid = sorted(n for n in extract_cited_numbers(raw) if not 1 <= n <= len(chunks))
        sources = self._sources(chunks, cited)
        nli = self._nli()
        passages = [{"index": n, "doc_id": c.chunk.doc_id, "page": c.chunk.page_number, "section": c.chunk.section,
                     "text": c.chunk.text} for n, c in enumerate(chunks, start=1)]
        result = Answer(
            question=topic,
            answer=raw,
            abstained=False,
            confidence=round(
                mean(sorted((calibrate_relevance(c) for c in chunks), reverse=True)[: config.CONFIDENCE_TOP_N]), 3
            ),
            sources=sources,
            invalid_citations=invalid,
            message=None if cited else NO_CITATIONS_NOTE,
            latency_s=round(time.perf_counter() - started, 3),
            mode="synthesize",
            table_cells=verify_table(raw, passages, nli),
            nli_available=nli is not None,
        )
        self.history.append(result.to_dict())
        return result


def assistant_from_folder(folder: str | Path = config.PAPERS_DIR) -> tuple[ResearchAssistant, list[dict[str, Any]]]:
    """Build a fresh assistant over every PDF in ``folder`` (used by the offline scripts).

    The corpus is rebuilt in memory on every run, so scripts always evaluate exactly
    the PDFs in the folder and never a stale index.
    """
    assistant = ResearchAssistant()
    results = assistant.ingest_folder(folder)
    return assistant, results
