"""Exception hierarchy shared across Thesora modules."""
from __future__ import annotations


class ThesoraError(Exception):
    """Base class for all expected, user-presentable Thesora errors."""


class PDFLoadError(ThesoraError):
    """A PDF could not be opened or contained no extractable text."""


class EmbeddingError(ThesoraError):
    """The embedding model failed to load or produced invalid output."""


class StoreError(ThesoraError):
    """The vector store is inconsistent, corrupt or could not be saved."""


class LLMUnavailableError(ThesoraError):
    """The LLM backend (Ollama) cannot be reached or lacks the model."""


class LLMGenerationError(ThesoraError):
    """The LLM backend replied, but with an error or an empty answer."""


class SessionNotFoundError(ThesoraError):
    """The chat session id is unknown, deleted or expired."""


class UploadError(ThesoraError):
    """An upload request was rejected (too many files, wrong type, too large, empty)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
