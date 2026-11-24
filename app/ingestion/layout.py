"""N1: table chunks and two-column detection (PyMuPDF / PyMuPDF4LLM family, AGPL). Falls back silently."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app import config
from app.core.models import DocumentChunk

logger = logging.getLogger(__name__)


def table_chunks(path: str | Path, doc_id: str, start_index: int, sections: dict[int, str]) -> list[DocumentChunk]:
    """Each detected table becomes one citable chunk (markdown rows), with page and section."""
    if config.PDF_LAYOUT_MODE != "auto":
        return []
    chunks: list[DocumentChunk] = []
    try:
        import pymupdf

        with pymupdf.open(str(path)) as pdf:
            for page in pdf:
                for table in page.find_tables().tables:
                    if table.row_count < config.TABLE_MIN_ROWS:
                        continue
                    md = table.to_markdown().strip()
                    if not md:
                        continue
                    caption = _caption(page, table.bbox)
                    if caption:  # "Table 2: Evaluation on LitQA" makes the table findable by what it is about
                        md = f"{caption}\n\n{md}"
                    n = page.number + 1
                    chunks.append(DocumentChunk(chunk_id=f"{doc_id}:p{n}:t{len(chunks)}", doc_id=doc_id, page_number=n,
                                                section=sections.get(n, ""), text=md,
                                                chunk_index=start_index + len(chunks), chunk_type="table"))
    except Exception as exc:  # layout extras are optional; text chunks are already ingested
        logger.warning("Table extraction skipped for %s: %s", doc_id, exc)
    return chunks


_CAPTION_RE = re.compile(r"^\s*Table\s+\d+\s*[:.]", re.IGNORECASE)


def _caption(page: Any, bbox: Any) -> str:
    """The 'Table N: ...' text block closest above or below a table (empty if none is near)."""
    top, bottom = bbox[1], bbox[3]
    best, best_dist = "", config.TABLE_CAPTION_MAX_GAP
    for block in page.get_text("blocks"):
        text = " ".join(str(block[4]).split())
        if not _CAPTION_RE.match(text):
            continue
        dist = min(abs(block[3] - top), abs(block[1] - bottom))
        if dist < best_dist:
            best, best_dist = text, dist
    return best[: config.TABLE_CAPTION_MAX_CHARS]


def column_report(path: str | Path) -> list[dict[str, Any]]:
    """Per page: how many text blocks sit wholly in the left / right half (two-column hint)."""
    import pymupdf

    report = []
    with pymupdf.open(str(path)) as pdf:
        for page in pdf:
            mid = page.rect.width / 2
            blocks = [b for b in page.get_text("blocks") if b[4].strip()]
            left = sum(1 for b in blocks if b[2] < mid)
            right = sum(1 for b in blocks if b[0] > mid)
            total = max(1, len(blocks))
            two_col = left / total >= config.COLUMN_GAP_RATIO and right / total >= config.COLUMN_GAP_RATIO
            report.append({"page": page.number + 1, "blocks": len(blocks), "left": left, "right": right,
                           "two_column": two_col})
    return report
