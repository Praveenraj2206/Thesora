"""Chunking: split page text into overlapping chunks, keeping provenance."""
from __future__ import annotations

import re
from typing import Sequence

from app import config
from app.core.models import DocumentChunk, PageText


_HYPHEN_BREAK_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*(?=[a-z])")


def dehyphenate(text: str) -> str:
    """Rejoin words split by an end-of-line hyphen ("hippo-\ncampus" -> "hippocampus").

    Only lower-case continuations are merged, so "Gtr1p-\nGtr2p" is left alone.
    Without this, PDF column text makes terms like "hippocampus" unsearchable by
    BM25 because the token never appears whole.
    """
    return _HYPHEN_BREAK_RE.sub(r"\1", text)


class DocumentChunker:
    """Splits pages into chunks with a RecursiveCharacterTextSplitter.

    Each chunk keeps its document id, page number and the section heading that
    was in effect at the chunk's starting character.
    """

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    @staticmethod
    def _section_at(page: PageText, start_index: int) -> str:
        """Return the last heading at or before ``start_index`` on the page."""
        section = page.start_section
        for offset, heading in page.headings:
            if offset <= start_index:
                section = heading
            else:
                break
        return section

    def chunk_pages(self, pages: Sequence[PageText]) -> list[DocumentChunk]:
        """Chunk every page, returning chunks with globally unique ids."""
        chunks: list[DocumentChunk] = []
        for page in pages:
            documents = self._splitter.create_documents([page.text])
            for position, document in enumerate(documents):
                text = dehyphenate(document.page_content).strip()
                if len(text) < config.MIN_CHUNK_CHARS:
                    continue
                start = max(0, int(document.metadata.get("start_index", 0)))
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{page.doc_id}:p{page.page_number}:c{position}",
                        doc_id=page.doc_id,
                        page_number=page.page_number,
                        section=self._section_at(page, start),
                        text=text,
                        chunk_index=position,
                    )
                )
        return chunks
