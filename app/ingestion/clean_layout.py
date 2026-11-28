"""Keep only a paper's useful text, in reading order (two columns: left column, then right column).

Scientific PDFs carry a lot of text that is not content. On page 1 of the survey papers it showed up as:
the sideways arXiv stamp, running headers and footers ("Proceedings of ...", "Nature | Vol 650", page
numbers), the author / affiliation / e-mail block, submission dates, tiny labels inside figures, and
footnotes such as "*Corresponding authors". Searching and quoting that text makes answers worse, so each
block of text is kept or dropped by simple layout rules:

* rotated text (arXiv stamp, margin notes)                     -> dropped
* text much smaller than the body font (figure labels)         -> dropped (``PDF_TINY_FONT_RATIO``)
* small text inside a figure (image or drawing area)           -> dropped
* blocks in the top / bottom page margin, or repeated on many pages (running headers, page numbers)
                                                               -> dropped (titles excepted)
* page 1: everything between the title and the abstract / first paragraph (authors, affiliations,
  dates)                                                       -> dropped, the title is kept
* short e-mail, URL, "corresponding author" footnotes          -> dropped
* text inside a detected table                                 -> dropped (the table is its own passage)

Every rule can be switched off with ``PDF_CLEAN=0``. Captions ("Figure 1: ...", "Table 2: ...") are kept.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from app import config

_URL_RE = re.compile(r"(https?://|www\.|\b[\w-]+\.(org|com|ai|io|edu|net)\b|\b\w+/\w+[-\w]*)", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"(correspond|equal contribution|core contributor|e-?mail|@)", re.IGNORECASE)
_ABSTRACT_RE = re.compile(r"\babstract\b", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table)\s*\d+", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


@dataclass
class DocLayout:
    """Facts about the whole document that the per-page rules need."""

    body_size: float
    repeated: set[str] = field(default_factory=set)  # normalised header/footer texts seen on many pages
    tables: dict[int, list[Any]] = field(default_factory=dict)  # page index -> table bboxes


def block_text(block: dict) -> str:
    return " ".join(
        "".join(ch.get("c", "") for span in line.get("spans", []) for ch in span.get("chars", []))
        or "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    ).strip()


def _block_size(block: dict) -> float:
    sizes = Counter()
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes[round(float(span.get("size", 0.0)), 1)] += max(1, len(span.get("chars", [])) or len(span.get("text", "")))
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _horizontal(line: dict) -> bool:
    direction = line.get("dir", (1, 0))
    return abs(direction[0] - 1) < 0.01 and abs(direction[1]) < 0.01


def _norm_margin_text(text: str) -> str:
    return _DIGITS_RE.sub("#", " ".join(text.lower().split()))[:80]


def document_layout(doc: Any, pages_raw: list[dict]) -> DocLayout:
    """Body font size (most common, weighted by characters) and texts repeated in page margins."""
    sizes: Counter = Counter()
    margin_texts: Counter = Counter()
    for raw, page in zip(pages_raw, doc):
        height = page.rect.height
        for block in raw.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(float(span.get("size", 0.0)), 1)] += len(span.get("chars", []))
            y0, y1 = block["bbox"][1], block["bbox"][3]
            if y1 < height * config.PDF_REPEAT_BAND or y0 > height * (1 - config.PDF_REPEAT_BAND):
                margin_texts[_norm_margin_text(block_text(block))] += 1
    min_pages = max(3, int(len(pages_raw) * config.PDF_REPEAT_MIN_SHARE))
    repeated = {t for t, n in margin_texts.items() if n >= min_pages and t}
    body = sizes.most_common(1)[0][0] if sizes else 0.0
    return DocLayout(body_size=body, repeated=repeated)


def _figure_areas(page: Any, raw: dict) -> list[Any]:
    """Areas covered by images or clusters of vector drawings (figures, diagrams)."""
    import pymupdf

    areas = [pymupdf.Rect(b["bbox"]) for b in raw.get("blocks", []) if b.get("type") == 1]
    try:
        areas += [r for r in page.cluster_drawings() if r.width * r.height >= config.PDF_FIGURE_MIN_AREA]
    except Exception:  # older PyMuPDF without cluster_drawings
        pass
    return [r + (-3, -3, 3, 3) for r in areas]


def _is_prose(text: str) -> bool:
    """Sentence-like text of any size (prompts, examples, footnoted remarks): 15+ words, mostly lower-case."""
    words = text.split()
    return len(words) >= config.PDF_PROSE_MIN_WORDS and sum(w[:1].islower() for w in words) / len(words) >= 0.5


def _is_paragraph(text: str, size: float, body: float) -> bool:
    """Running prose: long, mostly lower-case words (an author list is long but capitalised)."""
    words = text.split()
    if len(words) < config.PDF_PARAGRAPH_MIN_WORDS or size < body * 0.85:
        return False
    lower = sum(w[:1].islower() for w in words) / len(words)
    return lower >= 0.5


def _front_matter_end(blocks: list[dict], body: float) -> Optional[float]:
    """Page 1: the y where the content starts (an 'Abstract' heading or the first paragraph)."""
    starts = []
    for block in blocks:
        text = block_text(block)
        size = _block_size(block)
        first_words = " ".join(text.split()[:4])
        if (_ABSTRACT_RE.search(first_words) and len(text.split()) <= 6) or _ABSTRACT_RE.match(text) \
                or _is_paragraph(text, size, body):
            starts.append(block["bbox"][1])
        elif _ABSTRACT_RE.search(text) and len(text.split()) <= 6:  # "Dec 8, 2023 ABSTRACT"
            starts.append(block["bbox"][1])
    return min(starts) if starts else None


def _title_blocks(blocks: list[dict], height: float) -> set[int]:
    """Indexes of the title block(s): the largest font in the top half of page 1."""
    candidates = [(i, _block_size(b)) for i, b in enumerate(blocks)
                  if b["bbox"][1] < height * 0.5 and sum(c.isalpha() for c in block_text(b)) >= 6
                  and not block_text(b).lower().startswith(("arxiv", "proceedings"))]
    if not candidates:
        return set()
    top = max(size for _, size in candidates)
    return {i for i, size in candidates if size >= top - 0.5}


def _strip_rotated(block: dict) -> dict:
    return {**block, "lines": [line for line in block.get("lines", []) if _horizontal(line)]}


def clean_blocks(page: Any, raw: dict, page_index: int, layout: DocLayout,
                 dropped: Optional[list[tuple[str, str]]] = None) -> list[dict]:
    """The page's useful text blocks, in reading order. ``dropped`` (optional) collects (reason, text)."""

    def drop(reason: str, text: str) -> None:
        if dropped is not None:
            dropped.append((reason, text))

    import pymupdf

    width, height = page.rect.width, page.rect.height
    body = layout.body_size or 10.0
    blocks = [_strip_rotated(b) for b in raw.get("blocks", []) if b.get("type", 0) == 0]
    blocks = [b for b in blocks if b["lines"] and block_text(b)]
    if not config.PDF_CLEAN:
        return order_columns(blocks, width)

    figures = _figure_areas(page, raw)
    tables = layout.tables.get(page_index, [])
    titles = _title_blocks(blocks, height) if page_index == 0 else set()
    content_y = _front_matter_end([b for i, b in enumerate(blocks) if i not in titles], body) if page_index == 0 else None

    kept = []
    for i, block in enumerate(blocks):
        text = block_text(block)
        size = _block_size(block)
        rect = pymupdf.Rect(block["bbox"])
        words = text.split()
        if sum(ch.isalnum() for ch in text) < 2:
            drop("symbols", text)
            continue  # stray symbols from figures ("…", quote marks, bullets)
        if i in titles:
            kept.append(block)
            continue
        if _CAPTION_RE.match(text):  # figure / table captions describe content: keep
            kept.append(block)
            continue
        if size < body * config.PDF_TINY_FONT_RATIO and not _is_prose(text):
            drop("tiny font", text)
            continue  # figure labels, tiny print
        if size < body * 0.95 and len(words) < config.PDF_FIGURE_LABEL_MAX_WORDS and any(
                rect.intersects(f) and (rect & f).get_area() >= 0.6 * rect.get_area() for f in figures):
            drop("figure label", text)
            continue  # short label inside a figure (boxed prompts and examples are longer and kept)
        if not _is_prose(text) and any(
                rect.intersects(t) and (rect & t).get_area() >= 0.6 * rect.get_area() for t in tables):
            drop("table cell", text)
            continue  # table cells: the table is its own passage (prose is never dropped this way)
        in_margin = rect.y1 < height * config.PDF_MARGIN_BAND or rect.y0 > height * (1 - config.PDF_MARGIN_BAND)
        if (in_margin and size < body * 1.3) or _norm_margin_text(text) in layout.repeated:
            drop("header/footer", text)
            continue  # running header / footer / page number
        if content_y is not None and (rect.y1 <= content_y + 1 or (
                rect.y0 < content_y - 1 and len(words) <= 8 and not _is_paragraph(text, size, body))):
            drop("front matter", text)
            continue  # page 1 front matter: authors, affiliations, dates, badges
        if page_index == 0 and len(words) >= 8 and rect.y0 > height * 0.75 and size < body * 0.95 \
                and sum(w[:1].islower() for w in words) / len(words) < 0.35:
            drop("affiliations", text)
            continue  # page 1 affiliation list printed as a footnote
        if len(words) <= config.PDF_FOOTNOTE_MAX_WORDS and (
                (_FOOTNOTE_RE.search(text) and size < body * 0.95)
                or (_URL_RE.search(text) and (size < body * 0.95 or sum(1 for w in words if _URL_RE.search(w)) * 3 >= len(words)))):
            drop("footnote/link", text)
            continue  # e-mail / "corresponding author" / link lines and URL footnotes
        kept.append(block)
    return order_columns(kept, width)


def order_columns(blocks: list[dict], width: float) -> list[dict]:
    """Reading order: top to bottom; where the page has two columns, the whole left column, then the
    right column. A block spanning the middle (title, abstract, wide figure, single-column paper) closes
    the current two-column band."""
    mid = width / 2
    gap = config.PDF_COLUMN_TOLERANCE

    def kind(block: dict) -> str:
        x0, _, x1, _ = block["bbox"]
        if x1 <= mid + gap:
            return "left"
        if x0 >= mid - gap:
            return "right"
        return "full"

    ordered: list[dict] = []
    left: list[dict] = []
    right: list[dict] = []

    def flush() -> None:
        ordered.extend(sorted(left, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(right, key=lambda b: b["bbox"][1]))
        left.clear()
        right.clear()

    for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        k = kind(block)
        if k == "full":
            flush()
            ordered.append(block)
        else:
            (left if k == "left" else right).append(block)
    flush()
    return ordered
