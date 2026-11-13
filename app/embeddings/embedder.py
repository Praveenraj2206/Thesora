"""Sentence-embedding wrapper producing L2-normalised float32 vectors."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

import numpy as np

from app import config
from app.core.errors import EmbeddingError

logger = logging.getLogger(__name__)


def embedding_dimension(model: object) -> int:
    """Vector size of a sentence-transformers model, across library versions.

    Newer releases renamed ``get_sentence_embedding_dimension`` to
    ``get_embedding_dimension``; try both.
    """
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, name, None)
        if callable(getter):
            return int(getter())
    raise EmbeddingError("The embedding model does not report its vector dimension.")


class Embedder:
    """Lazy-loading all-MiniLM-L6-v2 embedder (CPU).

    Vectors are normalised, so inner product equals cosine similarity.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    def _load(self) -> object:
        """Load the model on first use (thread-safe)."""
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self._model_name)
                    model = SentenceTransformer(self._model_name, device="cpu")
                except Exception as exc:
                    raise EmbeddingError(
                        f"Could not load embedding model '{self._model_name}': {exc}"
                    ) from exc
                dim = embedding_dimension(model)
                if dim != config.EMBEDDING_DIM:
                    raise EmbeddingError(
                        f"Embedding model produces {dim}-d vectors but EMBEDDING_DIM={config.EMBEDDING_DIM}."
                    )
                self._model = model
            return self._model

    def warm_up(self) -> None:
        """Force the model to load now instead of on the first request."""
        self._load()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts into an ``(n, EMBEDDING_DIM)`` normalised float32 array."""
        if not texts:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        model = self._load()
        try:
            vectors = model.encode(  # type: ignore[attr-defined]
                list(texts),
                batch_size=config.EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a ``(1, EMBEDDING_DIM)`` array."""
        return self.embed([text])
