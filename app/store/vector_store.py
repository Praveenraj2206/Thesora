"""In-memory FAISS vector store with aligned chunk metadata.

Uses ``IndexFlatIP`` over normalised vectors (exact cosine search). Each chat
session owns its own store and nothing is written to disk, so a new chat (or a
browser refresh) can never retrieve PDFs or references from an earlier chat.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional, Sequence

import numpy as np

from app import config
from app.core.errors import StoreError
from app.core.models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS index plus chunk metadata, aligned by row id.

    ``version`` increases on every mutation so dependants (the BM25 index) know
    when to rebuild.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self._dim = dim
        self._index: Optional[Any] = None
        self._chunks: list[DocumentChunk] = []
        self._lock = threading.RLock()
        self.version = 0

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[DocumentChunk]:
        """Return a snapshot list of all chunks, aligned with FAISS row ids."""
        with self._lock:
            return list(self._chunks)

    def get_chunk(self, index: int) -> DocumentChunk:
        """Return the chunk at FAISS row ``index``."""
        with self._lock:
            return self._chunks[index]

    def has_document(self, doc_id: str) -> bool:
        """True if any chunk belongs to ``doc_id``."""
        with self._lock:
            return any(chunk.doc_id == doc_id for chunk in self._chunks)

    def documents(self) -> list[dict[str, Any]]:
        """Summarise ingested documents (id, chunk count, page count) in ingestion order."""
        with self._lock:
            summary: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
            for chunk in self._chunks:
                entry = summary.setdefault(chunk.doc_id, {"doc_id": chunk.doc_id, "chunks": 0, "_pages": set()})
                entry["chunks"] += 1
                entry["_pages"].add(chunk.page_number)
            return [
                {"doc_id": e["doc_id"], "chunks": e["chunks"], "pages": len(e["_pages"])}
                for e in summary.values()
            ]

    # ------------------------------------------------------------------ mutation
    def _new_index(self) -> Any:
        import faiss

        return faiss.IndexFlatIP(self._dim)

    def add(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> None:
        """Append chunks and their vectors (incremental add)."""
        if len(chunks) == 0:
            return
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(chunks), self._dim):
            raise StoreError(
                f"Embedding shape {vectors.shape} does not match {len(chunks)} chunks x {self._dim} dims."
            )
        with self._lock:
            if self._index is None:
                self._index = self._new_index()
            self._index.add(vectors)
            self._chunks.extend(chunks)
            self.version += 1

    def clear(self) -> None:
        """Forget every chunk and vector."""
        with self._lock:
            self._index = None
            self._chunks = []
            self.version += 1

    # ------------------------------------------------------------------ search
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row_id, cosine_score)`` pairs, best first."""
        with self._lock:
            if self._index is None or not self._chunks or k <= 0:
                return []
            query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
            scores, ids = self._index.search(query, min(k, len(self._chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def reconstruct(self, index: int) -> np.ndarray:
        """Return the stored vector for FAISS row ``index``."""
        with self._lock:
            if self._index is None:
                raise StoreError("Vector store is empty.")
            return np.asarray(self._index.reconstruct(int(index)), dtype=np.float32)
