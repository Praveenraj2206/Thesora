"""Chat sessions: one isolated, in-memory corpus per chat.

Each session owns its own ``ResearchAssistant`` (and therefore its own FAISS +
BM25 index). The expensive models (embedder, cross-encoder) and the LLM client
are loaded once and shared. Sessions live only in memory: deleting a session,
letting it idle past ``SESSION_TTL_MIN`` or restarting the server forgets its
PDFs and references for good, so a new chat always starts empty.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app import config
from app.core.errors import SessionNotFoundError
from app.embeddings.embedder import Embedder
from app.generation.llm import BaseLLM, OllamaLLM
from app.ingestion.chunker import DocumentChunker
from app.pipelines.rag import ResearchAssistant
from app.retrieval.retriever import Reranker

logger = logging.getLogger(__name__)

SECONDS_PER_MINUTE = 60


@dataclass
class Session:
    """One chat: its assistant plus bookkeeping timestamps (monotonic seconds)."""

    session_id: str
    assistant: ResearchAssistant
    created_at: float
    last_used: float
    lock: threading.Lock = field(default_factory=threading.Lock)

    def summary(self) -> dict[str, Any]:
        """Public description of the session."""
        return {
            "session_id": self.session_id,
            "documents": self.assistant.documents(),
            "chunks": len(self.assistant.store),
            "max_files_per_request": config.MAX_FILES_PER_REQUEST,
        }


class SessionManager:
    """Thread-safe registry of chat sessions with idle expiry and a size cap."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[BaseLLM] = None,
        ttl_s: float = config.SESSION_TTL_MIN * SECONDS_PER_MINUTE,
        max_sessions: int = config.MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()
        self.llm: BaseLLM = llm or OllamaLLM()
        self._ttl_s = ttl_s
        self._max_sessions = max(1, max_sessions)
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._chunker: Optional[DocumentChunker] = None

    # ------------------------------------------------------------------ internals
    def _shared_chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    def _purge_expired(self, now: float) -> None:
        """Drop sessions idle longer than the TTL (caller holds the lock)."""
        expired = [sid for sid, s in self._sessions.items() if now - s.last_used > self._ttl_s]
        for sid in expired:
            self._sessions.pop(sid).assistant.store.clear()
            logger.info("Session %s expired", sid)

    # ------------------------------------------------------------------ public API
    def create(self) -> Session:
        """Start a new, empty chat session."""
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            while len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                self._sessions.pop(oldest.session_id).assistant.store.clear()
                logger.info("Session %s evicted (MAX_SESSIONS=%d)", oldest.session_id, self._max_sessions)
            session_id = uuid.uuid4().hex
            assistant = ResearchAssistant(
                embedder=self.embedder, reranker=self.reranker, llm=self.llm, chunker=self._shared_chunker()
            )
            session = Session(session_id, assistant, created_at=now, last_used=now)
            self._sessions[session_id] = session
        logger.info("Session %s created", session_id)
        return session

    def get(self, session_id: str) -> Session:
        """Return a live session and refresh its idle timer.

        Raises:
            SessionNotFoundError: unknown, deleted or expired session.
        """
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(
                    "This chat session has ended or does not exist (sessions reset on refresh, after "
                    f"{config.SESSION_TTL_MIN} idle minutes, or when the server restarts). Start a new chat."
                )
            session.last_used = now
            return session

    def delete(self, session_id: str) -> bool:
        """End a session and free its index. Returns False if it did not exist."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.assistant.store.clear()
        logger.info("Session %s deleted", session_id)
        return True

    def count(self) -> int:
        """Number of live (non-expired) sessions."""
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._sessions)

    def warm_up(self) -> None:
        """Load the shared embedding model and cross-encoder ahead of the first request."""
        self.embedder.warm_up()
        self.reranker.model()
        self._shared_chunker()
