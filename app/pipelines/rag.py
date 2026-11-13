"""ResearchAssistant: orchestrates ingestion, retrieval, generation and verification.

One ResearchAssistant serves one chat session: it owns that chat's in-memory
VectorStore, while the heavy models (embedder, cross-encoder, LLM client) are
injected and shared between sessions.
"""
from __future__ import annotations

import logging
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
    ) -> None:
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
        if self._chunker is None:
            self._chunker = DocumentChunker()
        chunks = self._chunker.chunk_pages(pages)
        if not chunks:
            raise PDFLoadError(f"{pdf_path.name} produced no usable text chunks.")
        embeddings = self.embedder.embed([c.text for c in chunks])
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
    @staticmethod
    def _sources(chunks: Sequence[RetrievedChunk], cited: set[int]) -> list[Source]:
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
            )
            for n, item in enumerate(chunks, start=1)
        ]

    def _abstain(
        self, question: str, message: str, chunks: Sequence[RetrievedChunk], confidence: float, started: float, mode: str
    ) -> Answer:
        return Answer(
            question=question,
            answer="",
            abstained=True,
            confidence=round(confidence, 3),
            sources=self._sources(chunks, set()),
            message=message,
            latency_s=round(time.perf_counter() - started, 3),
            mode=mode,
        )

    # ------------------------------------------------------------------ ask
    def ask(self, question: str) -> Answer:
        """Answer a question with hybrid retrieval, reranking, verification and abstention."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        started = time.perf_counter()
        if len(self.store) == 0:
            return self._abstain(question, config.NO_DOCUMENTS_MESSAGE, [], 0.0, started, "ask")
        chunks = self.retriever.retrieve(question)
        best = chunks[0].relevance if chunks else 0.0
        if not chunks or best < config.ABSTAIN_THRESHOLD:
            return self._abstain(question, config.ABSTAIN_MESSAGE, chunks, best, started, "ask")

        raw = self.llm.generate(SYSTEM_PROMPT, build_ask_prompt(question, chunks), config.ASK_MAX_TOKENS)
        if is_insufficient(raw):
            return self._abstain(question, config.ABSTAIN_MESSAGE, chunks, 0.0, started, "ask")

        verification: VerificationResult = verify_citations(
            raw, {n: item.chunk.text for n, item in enumerate(chunks, start=1)}
        )
        sources = self._sources(chunks, set(verification.cited))
        for source in sources:
            check = verification.per_source.get(source.index)
            source.verified = bool(check and check.verified)
            source.match_score = check.match_score if check else None
        message = None
        if not verification.cited:
            message = NO_CITATIONS_NOTE
        elif verification.unverified:
            message = "Some citations could not be verified against the source text: " + ", ".join(
                f"[{n}]" for n in verification.unverified
            )
        return Answer(
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
        )

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
        return Answer(
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
        )


def assistant_from_folder(folder: str | Path = config.PAPERS_DIR) -> tuple[ResearchAssistant, list[dict[str, Any]]]:
    """Build a fresh assistant over every PDF in ``folder`` (used by the offline scripts).

    The corpus is rebuilt in memory on every run, so scripts always evaluate exactly
    the PDFs in the folder and never a stale index.
    """
    assistant = ResearchAssistant()
    results = assistant.ingest_folder(folder)
    return assistant, results
