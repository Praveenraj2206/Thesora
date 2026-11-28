"""N1: table chunks and two-column detection (PyMuPDF / PyMuPDF4LLM family, AGPL). Falls back silently."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app import config
from app.core.models import DocumentChunk

logger = logging.getLogger(__name__)


_TABLE_CACHE: dict[str, dict[int, list[tuple[Any, str]]]] = {}


def _looks_like_prose(table: Any) -> bool:
    """Ruled boxes around ordinary text (a journal's front page, a boxed abstract) are found as 'tables'
    too. A real table fills several cells per row; text in a box fills about one cell per row."""
    try:
        rows = table.extract()
    except Exception:
        return False
    filled = [sum(1 for c in row if c and c.strip()) for row in rows]
    if not any(filled):
        return True
    return sum(filled) / max(1, len([f for f in filled if f])) <= config.TABLE_MIN_CELLS_PER_ROW


def _may_have_table(page: Any) -> bool:
    """Cheap pre-check: a ruled table needs at least TABLE_MIN_RULES horizontal lines (or thin bars)."""
    rules = 0
    try:
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item[0] == "l" and abs(item[1].y - item[2].y) < 1.0 and abs(item[1].x - item[2].x) > 20:
                    rules += 1
                elif item[0] == "re" and item[1].width > 20:
                    rules += 1 if item[1].height < 2.0 else 2  # a thin bar, or a box: its top and bottom edges
            if rules >= config.TABLE_MIN_RULES:
                return True
    except Exception:
        return True
    return False


def detect_tables(document: Any, path: str | Path) -> dict[int, list[Any]]:
    """Find tables once per page (bbox + markdown with caption); the loader uses the boxes to keep table
    cells out of the prose, and ``table_chunks`` reuses the markdown instead of searching again."""
    found: dict[int, list[tuple[Any, str]]] = {}
    if config.PDF_LAYOUT_MODE == "auto":
        for page in document:
            if not _may_have_table(page):
                continue  # no ruled lines: find_tables (the slow part of ingest) cannot find a ruled table here
            try:
                tables = page.find_tables().tables
            except Exception as exc:  # table detection is optional
                logger.info("Table detection failed on page %s: %s", page.number + 1, exc)
                continue
            for table in tables:
                if table.row_count < config.TABLE_MIN_ROWS or _looks_like_prose(table):
                    continue
                md = (table.to_markdown() or "").strip()
                if not md:
                    continue
                caption = _caption(page, table.bbox)
                found.setdefault(page.number, []).append((table.bbox, f"{caption}\n\n{md}" if caption else md))
    _TABLE_CACHE[str(Path(path).resolve())] = found
    return {n: [bbox for bbox, _ in items] for n, items in found.items()}


def table_chunks(path: str | Path, doc_id: str, start_index: int, sections: dict[int, str]) -> list[DocumentChunk]:
    """Each detected table becomes one citable chunk (markdown rows headed by its caption), with page and section."""
    if config.PDF_LAYOUT_MODE != "auto":
        return []
    chunks: list[DocumentChunk] = []
    try:
        found = _TABLE_CACHE.pop(str(Path(path).resolve()), None)
        if found is None:
            import pymupdf

            with pymupdf.open(str(path)) as pdf:
                detect_tables(pdf, path)
            found = _TABLE_CACHE.pop(str(Path(path).resolve()), {})
        for page_index in sorted(found):
            for _, md in found[page_index]:
                n = page_index + 1
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
