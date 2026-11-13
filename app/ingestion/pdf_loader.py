"""PDF loading: per-page text extraction with section-heading detection.

PyMuPDF is the primary backend (it exposes font sizes, which drive heading
detection). pypdf is used as a text-only fallback if PyMuPDF fails.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app import config
from app.core.errors import PDFLoadError
from app.core.models import PageText

logger = logging.getLogger(__name__)

_BOLD_FLAG = 16
_WHITESPACE_RE = re.compile(r"\s+")
_NON_HEADING_RE = re.compile(r"^[\d\W_]+$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+\S")
_KNOWN_HEADINGS = frozenset(
    {
        "abstract", "introduction", "background", "related work", "methodology", "methods",
        "method", "approach", "experiments", "evaluation", "results", "discussion",
        "conclusion", "conclusions", "future work", "limitations", "references",
        "acknowledgements", "acknowledgments", "appendix",
    }
)


@dataclass(frozen=True)
class _Line:
    """One extracted text line with its dominant font properties."""

    text: str
    size: float
    bold: bool


_BLOCK_BREAK = _Line("", 0.0, False)


def is_heading_line(text: str, size: float, bold: bool, body_size: float) -> bool:
    """Decide whether a line is a section heading.

    A line is a heading when it is short, not just a number/page marker, does not
    end like a sentence, and either uses a clearly larger font than body text or
    is a bold numbered/upper-case/well-known heading.
    """
    candidate = text.strip()
    if not config.HEADING_MIN_CHARS <= len(candidate) <= config.HEADING_MAX_CHARS:
        return False
    if _NON_HEADING_RE.match(candidate) or candidate.endswith((".", ",", ";")):
        return False
    if body_size > 0 and size >= body_size * config.HEADING_SIZE_RATIO:
        return True
    if bold and len(candidate) <= config.HEADING_BOLD_MAX_CHARS:
        if _NUMBERED_HEADING_RE.match(candidate) or candidate.isupper():
            return True
        if candidate.lower().strip(" :") in _KNOWN_HEADINGS:
            return True
    return False


def _body_font_size(pages_lines: list[list[_Line]]) -> float:
    """Return the most common font size, weighted by text length."""
    counter: Counter[float] = Counter()
    for lines in pages_lines:
        for line in lines:
            if line.text:
                counter[round(line.size, 1)] += len(line.text)
    return counter.most_common(1)[0][0] if counter else 0.0


def _build_pages(pages_lines: list[list[_Line]], doc_id: str) -> list[PageText]:
    """Assemble PageText objects, tracking headings and the running section."""
    body_size = _body_font_size(pages_lines)
    pages: list[PageText] = []
    current_section = ""
    for number, lines in enumerate(pages_lines, start=1):
        start_section = current_section
        pieces: list[str] = []
        headings: list[tuple[int, str]] = []
        offset = 0
        previous_was_heading = False
        for line in lines:
            is_heading = bool(line.text) and is_heading_line(line.text, line.size, line.bold, body_size)
            if is_heading and previous_was_heading and headings:
                # A heading wrapped over several lines: merge it into one title.
                start, earlier = headings[-1]
                current_section = f"{earlier} {line.text}"
                headings[-1] = (start, current_section)
            elif is_heading:
                headings.append((offset, line.text))
                current_section = line.text
            previous_was_heading = is_heading
            pieces.append(line.text)
            offset += len(line.text) + 1  # +1 for the joining newline
        text = "\n".join(pieces)
        if text.strip():
            pages.append(PageText(doc_id, number, text, start_section, headings))
    return pages


def _extract_lines_pymupdf(page: object) -> list[_Line]:
    """Extract ordered lines (with font info) from one PyMuPDF page."""
    layout = page.get_text("dict", sort=True)  # type: ignore[attr-defined]
    lines: list[_Line] = []
    for block in layout.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = [s for s in raw_line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = _WHITESPACE_RE.sub(" ", "".join(s["text"] for s in spans)).strip()
            size = max(float(s.get("size", 0.0)) for s in spans)
            bold = all(
                bool(int(s.get("flags", 0)) & _BOLD_FLAG) or "bold" in str(s.get("font", "")).lower()
                for s in spans
            )
            lines.append(_Line(text, size, bold))
        lines.append(_BLOCK_BREAK)
    return lines


def _load_with_pymupdf(path: Path, doc_id: str) -> list[PageText]:
    """Load a PDF with PyMuPDF."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # older PyMuPDF releases only expose "fitz"
        import fitz

    document = fitz.open(str(path))
    try:
        if document.needs_pass:
            raise PDFLoadError(f"{path.name} is password-protected.")
        pages_lines = [_extract_lines_pymupdf(page) for page in document]
    finally:
        document.close()
    return _build_pages(pages_lines, doc_id)


def _load_with_pypdf(path: Path, doc_id: str) -> list[PageText]:
    """Fallback loader: plain text per page, no heading detection."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFLoadError(f"{path.name} is password-protected.")
        pages: list[PageText] = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(doc_id, number, text))
        return pages
    except PDFLoadError:
        raise
    except Exception as exc:  # pypdf raises many unrelated exception types
        raise PDFLoadError(f"Could not read {path.name}: {exc}") from exc


def load_pdf(path: str | Path, doc_id: Optional[str] = None) -> list[PageText]:
    """Load a PDF into per-page text with section metadata.

    Raises:
        PDFLoadError: if the file is missing, unreadable, encrypted, or has no
            extractable text (for example a scanned PDF; OCR is not supported).
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PDFLoadError(f"PDF not found: {pdf_path}")
    name = doc_id or pdf_path.name
    try:
        pages = _load_with_pymupdf(pdf_path, name)
    except PDFLoadError:
        raise
    except Exception as exc:  # PyMuPDF missing or failed: try the fallback backend
        logger.warning("PyMuPDF failed on %s (%s); falling back to pypdf.", pdf_path.name, exc)
        pages = _load_with_pypdf(pdf_path, name)
    if not pages:
        raise PDFLoadError(
            f"{pdf_path.name} has no extractable text (scanned PDF? OCR is not supported)."
        )
    return pages
