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

            corpus = [tokenize(chunk.text) or ["_"] for chunk in chunks]
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
        sparse = self.sparse_search(query, bm25_k)
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
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve + rerank. Returns up to ``final_k`` chunks, best first.

        Chunks below ``min_relevance`` (default ``RELEVANCE_FLOOR``) are dropped,
        keeping at least ``MIN_CONTEXT_CHUNKS``; pass 0.0 to disable the floor.
        """
        if len(self._store) == 0 or not query.strip():
            return []
        query_vector = self._embedder.embed_query(query)
        candidates = self._candidates(query, query_vector, dense_k, bm25_k, fused_k)
        scored = self._score(query, candidates)
        scored.sort(key=lambda c: c.relevance, reverse=True)
        floor = config.RELEVANCE_FLOOR if min_relevance is None else min_relevance
        return apply_relevance_floor(scored[:final_k], floor)

    def retrieve_across_documents(
        self,
        query: str,
        per_doc: int = config.SYNTH_CHUNKS_PER_DOC,
        max_total: int = config.SYNTH_MAX_CHUNKS,
        pool: int = config.SYNTH_CANDIDATE_POOL,
    ) -> list[RetrievedChunk]:
        """Retrieve the best chunks per paper so every relevant document is covered.

        A wide candidate pool is fused and reranked, irrelevant chunks are dropped,
        then chunks are picked round-robin across documents (best paper first).
        """
        if len(self._store) == 0 or not query.strip():
            return []
        query_vector = self._embedder.embed_query(query)
        candidates = self._candidates(query, query_vector, pool, pool, pool)
        scored = [
            c for c in self._score(query, candidates) if c.relevance >= config.SYNTH_MIN_CHUNK_RELEVANCE
        ]
        scored.sort(key=lambda c: c.relevance, reverse=True)
        by_doc: dict[str, list[RetrievedChunk]] = {}
        for candidate in scored:
            bucket = by_doc.setdefault(candidate.chunk.doc_id, [])
            if len(bucket) < per_doc:
                bucket.append(candidate)
        selected: list[RetrievedChunk] = []
        for depth in range(per_doc):
            for bucket in by_doc.values():  # insertion order == best document first
                if depth < len(bucket) and len(selected) < max_total:
                    selected.append(bucket[depth])
        return selected
