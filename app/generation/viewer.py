"""N9 PDF side panel and N11 explain + glossary. PDFs stay in memory only (bytes kept per chat)."""
from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional, Sequence

from app import config

logger = logging.getLogger(__name__)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ABBR_OK = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,9}")

EXPLAIN_SYSTEM_PROMPT = (
    "You explain research text in plain English for a student. Use ONLY the given passages and the selected text. "
    "Answer in four short sections with these headings: What, Why, How, Example. "
    "Start the Example section with: Illustrative example — not from the papers. "
    "If the passages do not explain something, say the papers do not say. Keep it under 180 words."
)


def _open(pdf_bytes: bytes):
    import fitz

    return fitz.open(stream=pdf_bytes, filetype="pdf")


def highlight(pdf_bytes: bytes, page: int, passage: str) -> dict[str, Any]:
    """Rectangles for ``passage`` on 1-based ``page``, in PDF points with a top-left origin
    (the coordinate system streamlit-pdf-viewer annotations use). Full text first, then sentence by sentence."""
    with _open(pdf_bytes) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise ValueError(f"Page {page} does not exist in this paper (it has {pdf.page_count} pages).")
        pg = pdf[page - 1]
        text = " ".join(passage.split())
        rects = pg.search_for(text) if len(text) < 400 else []
        if not rects:
            for sentence in _SENT_RE.split(text):
                if len(sentence) >= config.HIGHLIGHT_MIN_SENTENCE_CHARS:
                    rects += pg.search_for(sentence)
        annotations = [{"page": page, "x": r.x0, "y": r.y0, "width": r.width, "height": r.height,
                        "color": config.HIGHLIGHT_COLOR} for r in rects]
        return {"page": page, "found": bool(annotations), "annotations": annotations,
                "page_width": pg.rect.width, "page_height": pg.rect.height}


def page_png(pdf_bytes: bytes, page: int, passage: Optional[str] = None) -> bytes:
    """Fallback: the page rendered as PNG (passage highlighted when found)."""
    import fitz

    with _open(pdf_bytes) as pdf:
        pg = pdf[max(1, min(page, pdf.page_count)) - 1]
        if passage:
            for a in highlight(pdf_bytes, page, passage)["annotations"]:
                pg.add_highlight_annot(fitz.Rect(a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]))
        return pg.get_pixmap(matrix=fitz.Matrix(config.PAGE_PNG_ZOOM, config.PAGE_PNG_ZOOM)).tobytes("png")


# ----------------------------------------------------------------------------- N11 glossary
_NLP: Any = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        from unv_scispacy_abbr import AbbreviationDetector  # noqa: F401  (registers the factory)

        _NLP = spacy.blank("en")
        _NLP.add_pipe("abbreviation_detector")
    return _NLP


def build_glossary(pages: Sequence[Any], doc_id: str) -> list[dict[str, Any]]:
    """Abbreviation -> long form -> first page, using the Schwartz-Hearst detector on spacy.blank('en')."""
    if not config.GLOSSARY_ENABLED:
        return []
    found: dict[str, dict[str, Any]] = {}
    try:
        nlp = _nlp()
        for page in pages:
            for abbr in nlp(page.text)._.abbreviations:
                key = abbr.text.strip()
                if _ABBR_OK.fullmatch(key) and sum(ch.isupper() for ch in key) >= 2 and key not in found:
                    found[key] = {"abbr": key, "long_form": " ".join(abbr._.long_form.text.split()), "page": page.page_number,
                                  "doc_id": doc_id}
                if len(found) >= config.GLOSSARY_MAX_TERMS:
                    break
    except Exception as exc:  # glossary is optional
        logger.warning("Glossary unavailable for %s: %s", doc_id, exc)
    return list(found.values())


def glossary_hits(text: str, glossary: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    words = set(_WORD.findall(text))
    return [g for g in glossary if g["abbr"] in words]


def expand_query(query: str, glossary: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """BM25-only query with known long forms appended; None when nothing to add."""
    extra = [g["long_form"] for g in glossary_hits(query, glossary) if g["long_form"].lower() not in query.lower()]
    return f"{query} {' '.join(dict.fromkeys(extra))}" if extra else None


def build_explain_prompt(selected: str, passages: Sequence[str], follow_up: Optional[str], previous: Optional[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    prompt = f"Passages:\n\n{numbered}\n\nSelected text: {selected[: config.EXPLAIN_MAX_SELECTION_CHARS]}"
    if previous:
        prompt += f"\n\nYour previous explanation: {previous[: config.EXPLAIN_MAX_SELECTION_CHARS]}"
    prompt += f"\n\nQuestion: {follow_up}" if follow_up else "\n\nExplain the selected text."
    return prompt
