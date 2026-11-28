"""PDF loading: per-page text extraction with section-heading detection.

PyMuPDF is the primary backend (it exposes font sizes, which drive heading
detection). pypdf is used as a text-only fallback if PyMuPDF fails.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app import config
from app.core.errors import PDFLoadError
from app.core.models import PageText

logger = logging.getLogger(__name__)

_BOLD_FLAG = 16
_WHITESPACE_RE = re.compile(r"\s+")
_NON_HEADING_RE = re.compile(r"^[\d\W_]+$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.?)\s+[A-Z]")
_SECTION_NUMBER_RE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z](\.\d+)*)$")
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
_REFERENCES_RE = re.compile(r"^(\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$", re.IGNORECASE)


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
    if not re.search(r"[A-Za-z]{3,}", candidate):  # "C D", "(a)": figure panel letters, not a section
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
    in_references = False
    for number, lines in enumerate(pages_lines, start=1):
        start_section = current_section
        pieces: list[str] = []
        headings: list[tuple[int, str]] = []
        offset = 0
        previous_was_heading = False
        for line in lines:
            is_heading = bool(line.text) and is_heading_line(line.text, line.size, line.bold, body_size)
            if config.PDF_SKIP_REFERENCES:
                if is_heading:
                    in_references = bool(_REFERENCES_RE.match(line.text.strip()))
                if in_references:
                    continue  # the reference list names other papers; it is not this paper's content
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


def _span_chars(span: dict) -> list[tuple[str, float, float]]:
    """(character, x0, x1) for one rawdict span; falls back to a single pseudo-char for 'dict' spans."""
    chars = span.get("chars")
    if chars is None:
        x0, _, x1, _ = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
        return [(span.get("text", ""), float(x0), float(x1))]
    return [(c.get("c", ""), float(c["bbox"][0]), float(c["bbox"][2])) for c in chars]


# LaTeX often draws an accent as its own glyph before the letter ("L ´ala"): join it to the letter.
_SPACING_ACCENTS = {"´": "\u0301", "`": "\u0300", "¨": "\u0308", "ˆ": "\u0302", "˜": "\u0303",
                    "¸": "\u0327", "ˇ": "\u030c", "˚": "\u030a", "˝": "\u030b", "¯": "\u0304"}


def line_text(raw_line: dict) -> str:
    """Text of one PyMuPDF line with word spaces rebuilt from the gaps between letters.

    PyMuPDF only inserts a space when the gap between two glyphs is at least ~0.15 em, so the
    tightly justified lines of two-column papers come out glued ("Priortoprocessingaquery").
    Here a space is added wherever the horizontal gap exceeds ``PDF_SPACE_GAP_EM`` x font size.
    """
    out: list[str] = []
    prev_x1: Optional[float] = None
    accent = ""
    for span in raw_line.get("spans", []):
        size = float(span.get("size", 0.0)) or 1.0
        for ch, x0, x1 in _span_chars(span):
            if not ch:
                continue
            if ch in _SPACING_ACCENTS:  # keep it for the next letter; no space around it
                if out and out[-1] == " ":
                    out.pop()
                accent = _SPACING_ACCENTS[ch]
                continue
            if accent and not ch.isspace():
                out.append(unicodedata.normalize("NFC", ch + accent))
                accent, prev_x1 = "", x1
                continue
            if prev_x1 is not None and out and not out[-1].isspace() and not ch.isspace():
                if x0 - prev_x1 > config.PDF_SPACE_GAP_EM * size:
                    out.append(" ")
            out.append(ch)
            prev_x1 = x1
    return _WHITESPACE_RE.sub(" ", "".join(out)).strip()


_SUPERSCRIPT_FLAG = 1
_SPACE_BEFORE_PUNCT_RE = re.compile(r"(?<=\w) +([,.;:!?)\]])")
_MARKER_RE = re.compile(r"^[\d,–\-∗*†‡§¶]+$")


def _without_markers(spans: list[dict]) -> list[dict]:
    """Drop footnote and citation markers printed as superscripts ("latest²", "hallucinations²,³"):
    small raised spans made only of digits, commas, dashes or footnote symbols."""
    if not config.PDF_DROP_SUPERSCRIPT_MARKERS or len(spans) < 2:
        return spans
    biggest = max(float(s.get("size", 0.0)) for s in spans)
    kept = []
    for span in spans:
        text = "".join(c.get("c", "") for c in span.get("chars", [])) or span.get("text", "")
        raised = bool(int(span.get("flags", 0)) & _SUPERSCRIPT_FLAG) or float(span.get("size", 0.0)) < biggest * 0.8
        if raised and _MARKER_RE.match(text.strip() or "x"):
            continue
        kept.append(span)
    return kept or spans


def _extract_lines_pymupdf(page: object, raw: Optional[dict] = None, page_index: int = 0,
                           layout: Optional[Any] = None) -> list[_Line]:
    """Ordered lines (with font info) of one page's useful text blocks (see ``clean_layout``)."""
    from app.ingestion.clean_layout import DocLayout, clean_blocks

    raw = raw if raw is not None else page.get_text("rawdict")  # type: ignore[attr-defined]
    layout = layout or DocLayout(body_size=0.0)
    lines: list[_Line] = []
    for block in clean_blocks(page, raw, page_index, layout):
        for raw_line in block.get("lines", []):
            spans = [s for s in raw_line.get("spans", []) if any(not c.get("c", " ").isspace() for c in s.get("chars", []))
                     or s.get("text", "").strip()]
            if not spans:
                continue
            spans = _without_markers(spans)
            text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", line_text({"spans": spans}))
            if not text:
                continue
            size = max(float(s.get("size", 0.0)) for s in spans)
            bold = all(
                bool(int(s.get("flags", 0)) & _BOLD_FLAG) or "bold" in str(s.get("font", "")).lower()
                for s in spans
            )
            previous = lines[-1] if lines else None
            if previous is not None and previous.text and _SECTION_NUMBER_RE.match(previous.text) \
                    and text[:1].isupper() and abs(previous.size - size) < 0.6:
                # "2" + "Pipeline", "A" + "Python Package Usage": one heading, as printed
                lines[-1] = _Line(f"{previous.text} {text}", size, bold and previous.bold)
                continue
            lines.append(_Line(text, size, bold))
        lines.append(_BLOCK_BREAK)
    return lines


def _load_with_pymupdf(path: Path, doc_id: str) -> list[PageText]:
    """Load a PDF with PyMuPDF: useful text only, two columns read column by column."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # older PyMuPDF releases only expose "fitz"
        import fitz
    from app.ingestion.clean_layout import document_layout
    from app.ingestion.layout import detect_tables

    document = fitz.open(str(path))
    try:
        if document.needs_pass:
            raise PDFLoadError(f"{path.name} is password-protected.")
        raws = [page.get_text("rawdict") for page in document]
        layout = document_layout(document, raws)
        layout.tables = detect_tables(document, path)
        pages_lines = [_extract_lines_pymupdf(page, raw, i, layout)
                       for i, (page, raw) in enumerate(zip(document, raws))]
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
