"""Plain dataclasses that flow between Thesora's pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PageText:
    """Text of one PDF page plus the section structure detected on it.

    ``headings`` holds ``(character_offset, heading_text)`` pairs that index into
    ``text``; ``start_section`` is the section already open when the page began.
    """

    doc_id: str
    page_number: int
    text: str
    start_section: str = ""
    headings: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A retrievable text chunk with its provenance metadata."""

    chunk_id: str
    doc_id: str
    page_number: int
    section: str
    text: str
    chunk_index: int


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever together with all stage scores.

    ``relevance`` is the final score in [0, 1] used for abstention: the
    sigmoid-normalised cross-encoder score, or cosine similarity when the
    cross-encoder is unavailable (``relevance_source`` records which).
    """

    chunk: DocumentChunk
    store_index: int
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    relevance: float = 0.0
    relevance_source: str = "cross-encoder"
    rerank_logit: Optional[float] = None


@dataclass
class Source:
    """A citation target shown to the user.

    ``verified`` is True/False when citation verification ran and None when it
    did not apply (e.g. synthesis). Sources that were retrieved but never cited
    have ``cited=False`` and ``verified=False``.
    """

    index: int
    chunk_id: str
    doc_id: str
    page_number: int
    section: str
    text: str
    relevance: float
    cited: bool = False
    verified: Optional[bool] = None
    match_score: Optional[float] = None


@dataclass
class Answer:
    """Final response of the assistant for /ask and /synthesize."""

    question: str
    answer: str
    abstained: bool
    confidence: float
    sources: list[Source] = field(default_factory=list)
    unverified_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    message: Optional[str] = None
    latency_s: float = 0.0
    mode: str = "ask"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return asdict(self)
