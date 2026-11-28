"""Hybrid retrieval: BM25 + dense search fused with RRF, then cross-encoder rerank."""
from __future__ import annotations

import logging
import math
import re
import threading
from typing import Any, Optional, Sequence

import numpy as np

from app import config
from app.core.models import RetrievedChunk
from app.embeddings.embedder import Embedder
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenisation used for BM25."""
    return _TOKEN_RE.findall(text.lower())


def sigmoid(value: float) -> float:
    """Numerically safe logistic function mapping logits to (0, 1)."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = config.RRF_K, top_k: Optional[int] = None
) -> list[tuple[int, float]]:
    """Fuse several ranked id lists: score(d) = sum over lists of 1 / (k + rank).

    Ranks start at 1. Ties break by id for determinism. Returns ``(id, score)``
    pairs, best first, truncated to ``top_k`` when given.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused[:top_k] if top_k is not None else fused


def apply_relevance_floor(
    ranked: Sequence[RetrievedChunk], floor: float, min_keep: int = config.MIN_CONTEXT_CHUNKS
) -> list[RetrievedChunk]:
    """Drop chunks whose relevance is below ``floor`` but always keep ``min_keep`` best.

    ``ranked`` must already be sorted best-first. Weak chunks only add noise to
    the LLM prompt and show up as misleading "sources" in the UI.
    """
    kept = [chunk for chunk in ranked if chunk.relevance >= floor]
    return kept if len(kept) >= min_keep else list(ranked[:min_keep])



_NUMBER_QUESTION_RE = re.compile(
    r"\b(accuracy|precision|recall|f1|ndcg|mrr|score|scores|rate|rates|percent|percentage|how many|how much|"
    r"number of|results?|performance|benchmark|table|cost|latency|size|compare[ds]?)\b|%",
    re.IGNORECASE,
)


def _with_table_slot(query: str, ranked: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """For questions asking for numbers, keep the best clearly relevant table in the last slot.

    Prose that names the paper and its benchmark can outscore the results table itself, and the
    table holds the numbers the question asks for."""
    top = ranked[:k]
    if (not config.TABLE_SLOT or len(top) < k or any(c.chunk.chunk_type == "table" for c in top)
            or not _NUMBER_QUESTION_RE.search(query)):
        return top
    table = next((c for c in ranked[k:] if c.chunk.chunk_type == "table"
                  and c.relevance >= config.TABLE_SLOT_MIN_RELEVANCE), None)
    return top[:-1] + [table] if table else top

class Reranker:
    """Lazy, thread-safe cross-encoder wrapper shared by every chat session.

    Loading fails at most once: afterwards ``model()`` returns None and callers
    fall back to dense cosine relevance instead of retrying on every request.
    """

    def __init__(self, model_name: str = config.RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[Any] = None
        self._failed = False
        self._lock = threading.Lock()

    def model(self) -> Optional[Any]:
        """Return the loaded CrossEncoder, or None if it cannot be loaded."""
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder %s", self._model_name)
                self._model = CrossEncoder(self._model_name, max_length=config.RERANKER_MAX_LENGTH, device="cpu")
            except Exception as exc:
                self._failed = True
                logger.warning("Cross-encoder unavailable (%s); falling back to dense cosine relevance.", exc)
            return self._model

    def logits(self, query: str, texts: Sequence[str]) -> list[float]:
        """Raw cross-encoder logits for ``(query, text)`` pairs. Raises if the model is unavailable."""
        model = self.model()
        if model is None:
            raise RuntimeError("cross-encoder unavailable")
        scores = model.predict(
            [(query, text) for text in texts], batch_size=config.RERANKER_BATCH_SIZE, show_progress_bar=False
        )
        return [float(score) for score in np.asarray(scores).reshape(-1)]


class HybridRetriever:
    _sparse_query: Optional[str] = None
    """Dense + BM25 retrieval, RRF fusion and cross-encoder reranking over one store.

    If the cross-encoder cannot be loaded, the retriever degrades gracefully to
    dense cosine similarity as the relevance score (and says so in each result).
    """

    def __init__(self, store: VectorStore, embedder: Embedder, reranker: Optional[Reranker] = None) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker or Reranker()
        self._bm25: Optional[Any] = None
        self._bm25_version = -1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ components
    def _ensure_bm25(self) -> Optional[Any]:
        """(Re)build the BM25 index whenever the store has changed."""
        with self._lock:
            if self._bm25_version == self._store.version and self._bm25 is not None:
                return self._bm25
            chunks = self._store.all_chunks()
            if not chunks:
                self._bm25, self._bm25_version = None, self._store.version
                return None
            from rank_bm25 import BM25Okapi

            corpus = [tokenize(chunk.search_text or chunk.text) or ["_"] for chunk in chunks]
            self._bm25 = BM25Okapi(corpus)
            self._bm25_version = self._store.version
            return self._bm25

    def warm_up(self) -> None:
        """Load the embedder, cross-encoder and BM25 index ahead of time."""
        self._embedder.warm_up()
        self._reranker.model()
        self._ensure_bm25()

    # ------------------------------------------------------------------ stages
    def dense_search(self, query_vector: np.ndarray, k: int = config.DENSE_TOP_K) -> list[tuple[int, float]]:
        """Top-k dense results as ``(row_id, cosine)``."""
        return self._store.search(query_vector, k)

    def sparse_search(self, query: str, k: int = config.BM25_TOP_K) -> list[tuple[int, float]]:
        """Top-k BM25 results as ``(row_id, bm25_score)``; empty if no term overlap."""
        bm25 = self._ensure_bm25()
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = np.asarray(bm25.get_scores(tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[: max(0, k)]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _candidates(
        self, query: str, query_vector: np.ndarray, dense_k: int, bm25_k: int, fused_k: int
    ) -> list[RetrievedChunk]:
        """Run dense + BM25 search and fuse with RRF into scored candidates."""
        dense = self.dense_search(query_vector, dense_k)
        sparse = self.sparse_search(self._sparse_query or query, bm25_k)
        dense_scores = dict(dense)
        sparse_scores = dict(sparse)
        fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]], top_k=fused_k)
        candidates: list[RetrievedChunk] = []
        for index, rrf_score in fused:
            dense_score = dense_scores.get(index)
            if dense_score is None:  # BM25-only hit: compute its cosine for fallback scoring
                dense_score = float(np.dot(query_vector.reshape(-1), self._store.reconstruct(index)))
            candidates.append(
                RetrievedChunk(
                    chunk=self._store.get_chunk(index),
                    store_index=index,
                    dense_score=dense_score,
                    bm25_score=sparse_scores.get(index),
                    rrf_score=rrf_score,
                )
            )
        return candidates

    def _score(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Assign ``relevance`` = sigmoid(cross-encoder logit), or cosine when the reranker is unavailable.

        The raw logit is kept on ``rerank_logit`` so confidence can use a calibrated, non-saturating score.
        """
        if not candidates:
            return []
        try:
            logits = self._reranker.logits(query, [c.chunk.text for c in candidates])
            for candidate, logit in zip(candidates, logits):
                candidate.rerank_logit = logit
                candidate.relevance = sigmoid(logit)
                candidate.relevance_source = "cross-encoder"
            return candidates
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed (%s); using dense cosine.", exc)
        for candidate in candidates:
            candidate.rerank_logit = None
            candidate.relevance = min(1.0, max(0.0, candidate.dense_score or 0.0))
            candidate.relevance_source = "dense-fallback"
        return candidates

    # ------------------------------------------------------------------ public API
    def retrieve(
        self,
        query: str,
        dense_k: int = config.DENSE_TOP_K,
        bm25_k: int = config.BM25_TOP_K,
        fused_k: int = config.FUSED_TOP_K,
        final_k: int = config.FINAL_TOP_K,
        min_relevance: Optional[float] = None,
        sparse_query: Optional[str] = None,
        scope: Optional[set[str]] = None,
        alt_queries: Sequence[str] = (),
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve + rerank. Returns up to ``final_k`` chunks, best first.

        Chunks below ``min_relevance`` (default ``RELEVANCE_FLOOR``) are dropped,
        keeping at least ``MIN_CONTEXT_CHUNKS``; pass 0.0 to disable the floor.
        ``scope``: only search these papers (the question names them), with a wider pool.
        ``alt_queries``: other phrasings (paper name removed, abbreviations spelled out);
        each passage keeps its best reranker score over all phrasings.
        """
        if len(self._store) == 0 or not query.strip():
            return []
        query_vector = self._embedder.embed_query(query)
        self._sparse_query = sparse_query  # N11: glossary long forms added to the BM25 query only
        try:
            if scope:
                candidates = self._per_document_candidates(query, query_vector, config.SCOPED_POOL, docs=scope)
                seen = {c.store_index for c in candidates}
                for index in self._table_indexes(scope):  # a named paper's tables always reach the reranker
                    if index not in seen:
                        seen.add(index)
                        candidates.append(RetrievedChunk(
                            chunk=self._store.get_chunk(index), store_index=index,
                            dense_score=float(np.dot(query_vector.reshape(-1), self._store.reconstruct(index))),
                            bm25_score=None, rrf_score=0.0))
                for alt in dict.fromkeys(q for q in alt_queries if q.strip() and q != query):
                    # the name-free wording finds passages the paper's name hides ("the threshold is 0.8")
                    self._sparse_query = None
                    for extra in self._per_document_candidates(alt, self._embedder.embed_query(alt),
                                                               config.SCOPED_POOL, docs=scope):
                        if extra.store_index not in seen:
                            seen.add(extra.store_index)
                            candidates.append(extra)
            else:
                candidates = self._candidates(query, query_vector, dense_k, bm25_k, fused_k)
        finally:
            self._sparse_query = None
        scored = self._score(query, candidates)
        self._rescore_with(alt_queries, scored)
        scored.sort(key=lambda c: c.relevance, reverse=True)
        top = _with_table_slot(query, scored, final_k)
        floor = config.RELEVANCE_FLOOR if min_relevance is None else min_relevance
        return apply_relevance_floor(top, floor)

    def _table_indexes(self, docs: set[str]) -> list[int]:
        """Store rows of the table passages of ``docs`` (papers have few tables; cached per store size)."""
        size = len(self._store)
        if getattr(self, "_tables_cache_size", -1) != size:
            self._tables_by_doc: dict[str, list[int]] = {}
            for i in range(size):
                chunk = self._store.get_chunk(i)
                if chunk.chunk_type == "table":
                    self._tables_by_doc.setdefault(chunk.doc_id, []).append(i)
            self._tables_cache_size = size
        return [i for d in docs for i in self._tables_by_doc.get(d, [])][: config.SCOPED_MAX_TABLES]

    def _rescore_with(self, alt_queries: Sequence[str], scored: list[RetrievedChunk]) -> None:
        """Keep each passage's best cross-encoder score over several phrasings of the question."""
        if not scored or not alt_queries or scored[0].relevance_source != "cross-encoder":
            return
        for alt in dict.fromkeys(q for q in alt_queries if q.strip()):
            try:
                logits = self._reranker.logits(alt, [c.chunk.text for c in scored])
            except Exception as exc:  # the first scores stay valid
                logger.warning("Alternative-query rerank failed (%s).", exc)
                return
            for candidate, logit in zip(scored, logits):
                if candidate.rerank_logit is None or logit > candidate.rerank_logit:
                    candidate.rerank_logit, candidate.relevance = logit, sigmoid(logit)

    def _per_document_candidates(
        self, query: str, query_vector: np.ndarray, pool: int, docs: Optional[set[str]] = None
    ) -> list[RetrievedChunk]:
        """Dense + BM25 search run separately inside each paper, fused per paper with RRF.

        A single shared pool lets a few long papers crowd the others out; ranking within each
        paper guarantees every uploaded paper contributes its own best passages.
        """
        total = len(self._store)
        dense_all = self.dense_search(query_vector, total)
        sparse_all = self.sparse_search(self._sparse_query or query, total)
        doc_of = {i: self._store.get_chunk(i).doc_id for i, _ in dense_all}
        for i, _ in sparse_all:
            doc_of.setdefault(i, self._store.get_chunk(i).doc_id)
        dense_by_doc: dict[str, list[tuple[int, float]]] = {}
        sparse_by_doc: dict[str, list[tuple[int, float]]] = {}
        for i, score in dense_all:
            bucket = dense_by_doc.setdefault(doc_of[i], [])
            if len(bucket) < pool:
                bucket.append((i, score))
        for i, score in sparse_all:
            bucket = sparse_by_doc.setdefault(doc_of[i], [])
            if len(bucket) < pool:
                bucket.append((i, score))
        candidates: list[RetrievedChunk] = []
        for doc_id in dict.fromkeys(list(dense_by_doc) + list(sparse_by_doc)):
            if docs is not None and doc_id not in docs:
                continue
            dense, sparse = dense_by_doc.get(doc_id, []), sparse_by_doc.get(doc_id, [])
            dense_scores, sparse_scores = dict(dense), dict(sparse)
            # keep every candidate of both lists: a passage only keyword search finds (a results table) must reach the reranker
            fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]], top_k=None)
            for index, rrf_score in fused:
                dense_score = dense_scores.get(index)
                if dense_score is None:
                    dense_score = float(np.dot(query_vector.reshape(-1), self._store.reconstruct(index)))
                candidates.append(RetrievedChunk(chunk=self._store.get_chunk(index), store_index=index,
                                                 dense_score=dense_score, bm25_score=sparse_scores.get(index),
                                                 rrf_score=rrf_score))
        return candidates

    def retrieve_across_documents(
        self,
        query: str,
        per_doc: int = config.SYNTH_CHUNKS_PER_DOC,
        max_total: int = config.SYNTH_MAX_CHUNKS,
        pool: int = config.SYNTH_DOC_POOL,
    ) -> list[RetrievedChunk]:
        """Retrieve the best chunks per paper so every uploaded paper is covered.

        Candidates are searched inside each paper, reranked, and picked round-robin
        (best paper first). Each paper keeps at least its best passage so it gets a row;
        further passages must pass the relevance floor.
        """
        if len(self._store) == 0 or not query.strip():
            return []
        query_vector = self._embedder.embed_query(query)
        scored = self._score(query, self._per_document_candidates(query, query_vector, pool))
        scored.sort(key=lambda c: c.relevance, reverse=True)
        by_doc: dict[str, list[RetrievedChunk]] = {}
        for candidate in scored:
            bucket = by_doc.setdefault(candidate.chunk.doc_id, [])
            keep = not bucket and config.SYNTH_COVER_ALL_PAPERS
            if len(bucket) < per_doc and (keep or candidate.relevance >= config.SYNTH_MIN_CHUNK_RELEVANCE):
                bucket.append(candidate)
        selected: list[RetrievedChunk] = []
        for depth in range(per_doc):
            for bucket in by_doc.values():  # insertion order == best document first
                if depth < len(bucket) and len(selected) < max_total:
                    selected.append(bucket[depth])
        return selected
