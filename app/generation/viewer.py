"""N9 PDF side panel and N11 explain + glossary. PDFs stay in memory only (bytes kept per chat)."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Mapping, Optional, Sequence

from app import config

logger = logging.getLogger(__name__)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ABBR_OK = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,9}")

EXPLAIN_SYSTEM_PROMPT = (
    "You help a student understand an answer that came from research papers. Use ONLY the given passages. "
    "First write a short plain-English explanation of the selected text: 3 to 5 sentences, no headings, no lists, "
    "explaining any technical term simply. If the passages do not explain something, say the papers do not say. "
    "Then suggest five short follow-up questions the student could ask next; each must be answerable "
    "from these passages and follow naturally from the original question. "
    'Reply only with JSON: {"explanation": "...", "follow_ups": ["...", "...", "...", "...", "..."]}'
)


def _open(pdf_bytes: bytes):
    import fitz

    return fitz.open(stream=pdf_bytes, filetype="pdf")


def _fold(text: str) -> str:
    """Letters and digits only, lower-case, accents and ligatures folded ("ﬁ" -> "fi", "á" -> "a")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if ch.isalnum() and not unicodedata.combining(ch)).lower()


def _page_chars(pg: Any) -> tuple[str, list[tuple[Any, tuple[int, int]]]]:
    """The page as one folded string, with each folded character's rectangle and (block, line) key."""
    import fitz

    folded: list[str] = []
    boxes: list[tuple[Any, tuple[int, int]]] = []
    for b_i, block in enumerate(pg.get_text("rawdict").get("blocks", [])):
        for l_i, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                for c in span.get("chars", []):
                    for ch in _fold(c["c"]):
                        folded.append(ch)
                        boxes.append((fitz.Rect(c["bbox"]), (b_i, l_i)))
    return "".join(folded), boxes


def _match_rects(pg: Any, passage: str) -> tuple[list[Any], float]:
    """Rectangles (one per text line) covering the passage, and the share of the passage found.

    Matching ignores spaces, hyphens, punctuation, ligatures and accents, so a word split over two lines
    or a sentence continued in the next column still matches. Each sentence is looked up on its own; a
    sentence not found whole is matched in pieces of HIGHLIGHT_PIECE_CHARS letters."""
    import fitz

    haystack, boxes = _page_chars(pg)
    by_line: dict[tuple[int, int], Any] = {}
    total = matched = 0

    def mark(start: int, length: int) -> None:
        for rect, key in boxes[start:start + length]:
            by_line[key] = by_line[key] | rect if key in by_line else fitz.Rect(rect)

    for sentence in _SENT_RE.split(passage):
        needle = _fold(sentence)
        if len(needle) < config.HIGHLIGHT_MIN_SENTENCE_CHARS:
            continue
        total += len(needle)
        at = haystack.find(needle)
        if at >= 0:
            mark(at, len(needle))
            matched += len(needle)
            continue
        step = config.HIGHLIGHT_PIECE_CHARS
        for i in range(0, len(needle) - step + 1, step):
            at = haystack.find(needle[i:i + step])
            if at >= 0:
                mark(at, step)
                matched += step
    return list(by_line.values()), (matched / total if total else 0.0)


def highlight(pdf_bytes: bytes, page: int, passage: str) -> dict[str, Any]:
    """Rectangles for ``passage`` on 1-based ``page``, in PDF points with a top-left origin."""
    with _open(pdf_bytes) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise ValueError(f"Page {page} does not exist in this paper (it has {pdf.page_count} pages).")
        pg = pdf[page - 1]
        rects, coverage = _match_rects(pg, passage)
        found = bool(rects) and coverage >= config.HIGHLIGHT_MIN_COVERAGE
        annotations = [{"page": page, "x": r.x0, "y": r.y0, "width": r.width, "height": r.height,
                        "color": config.HIGHLIGHT_COLOR} for r in rects] if found else []
        return {"page": page, "found": found, "coverage": round(coverage, 2), "annotations": annotations,
                "page_width": pg.rect.width, "page_height": pg.rect.height}


def page_png(pdf_bytes: bytes, page: int, passage: Optional[str] = None, zoom: Optional[float] = None) -> bytes:
    """The page rendered as PNG (sharp at ``zoom`` x 72 dpi); with ``passage`` the passage is highlighted in it."""
    import fitz

    with _open(pdf_bytes) as pdf:
        pg = pdf[max(1, min(page, pdf.page_count)) - 1]
        if passage:
            for a in highlight(pdf_bytes, page, passage)["annotations"]:
                pg.add_highlight_annot(fitz.Rect(a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]))
        scale = zoom or config.PAGE_PNG_ZOOM
        return pg.get_pixmap(matrix=fitz.Matrix(scale, scale)).tobytes("png")


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
                    long_form = re.sub(r"(\w)- (\w)", r"\1-\2", " ".join(abbr._.long_form.text.split()))
                    found[key] = {"abbr": key, "long_form": long_form, "page": page.page_number,
                                  "doc_id": doc_id}
                if len(found) >= config.GLOSSARY_MAX_TERMS:
                    break
    except Exception as exc:  # glossary is optional
        logger.warning("Glossary unavailable for %s: %s", doc_id, exc)
    text = " ".join(getattr(p, "text", "") for p in pages)
    for entry in found.values():  # how often the paper uses the term: the sidebar shows the most used first
        entry["count"] = len(re.findall(rf"\b{re.escape(entry['abbr'])}s?\b", text))
    return list(found.values())


def _abbr_key(abbr: str) -> str:
    """'LLMs' and 'LLM' are the same term."""
    return abbr[:-1] if len(abbr) > 2 and abbr.endswith("s") and abbr[:-1].isupper() else abbr


def merge_glossary(existing: list[dict[str, Any]], new: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add one paper's abbreviations to a chat glossary, one entry per term (not one per paper).

    The first paper's long form and page are kept; ``papers`` lists every paper that defines the term.
    Entries stay sorted alphabetically.
    """
    by_key = {_abbr_key(g["abbr"]): g for g in existing}
    for entry in new:
        key = _abbr_key(entry["abbr"])
        if key in by_key:
            papers = by_key[key].setdefault("papers", [by_key[key]["doc_id"]])
            if entry["doc_id"] not in papers:
                papers.append(entry["doc_id"])
            by_key[key]["count"] = by_key[key].get("count", 0) + entry.get("count", 0)
        else:
            by_key[key] = {**entry, "abbr": key, "papers": [entry["doc_id"]]}
    return sorted(by_key.values(), key=lambda g: g["abbr"].lower())


def glossary_hits(text: str, glossary: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    words = {_abbr_key(w) for w in _WORD.findall(text)}
    return [g for g in glossary if _abbr_key(g["abbr"]) in words]


def expand_query(query: str, glossary: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """BM25-only query with known long forms appended; None when nothing to add."""
    extra = [g["long_form"] for g in glossary_hits(query, glossary) if g["long_form"].lower() not in query.lower()]
    return f"{query} {' '.join(dict.fromkeys(extra))}" if extra else None


def spell_out(query: str, glossary: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The question with known abbreviations written out ("LFS" -> "LFS (Logic Fidelity Score)") for the
    reranker, which does not know paper-specific abbreviations. None when nothing to spell out."""
    out = query
    for g in glossary_hits(query, glossary):
        if g["long_form"].lower() in query.lower():
            continue
        out = re.sub(rf"\b{re.escape(g['abbr'])}s?\b", lambda m, g=g: f"{m.group(0)} ({g['long_form']})", out, count=1)
    return out if out != query else None


def build_explain_prompt(selected: str, passages: Sequence[str], follow_up: Optional[str], previous: Optional[str],
                         question: str = "") -> str:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    prompt = f"Passages:\n\n{numbered}\n\n"
    if question:
        prompt += f"Original question: {question}\n\n"
    prompt += f"Selected text: {selected[: config.EXPLAIN_MAX_SELECTION_CHARS]}"
    if previous:
        prompt += f"\n\nYour previous explanation: {previous[: config.EXPLAIN_MAX_SELECTION_CHARS]}"
    prompt += f"\n\nStudent's question: {follow_up}" if follow_up else "\n\nExplain the selected text."
    return prompt


SUGGEST_SYSTEM_PROMPT = (
    "You suggest what a researcher could ask next about their own uploaded papers. Use ONLY the given paper "
    "titles and passages. Write six short, specific questions that follow on from the researcher's "
    "question and that these passages clearly answer; vary them (a detail, a number or result, a comparison). "
    "Do not repeat questions already asked. "
    'Reply only with JSON: {"questions": ["...", "...", "...", "...", "...", "..."]}'
)


def build_suggest_prompt(question: str, passages: Sequence[str], titles: Sequence[str], asked: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {p[:900]}" for i, p in enumerate(passages, start=1))
    prompt = "Papers:\n" + "\n".join(f"- {t}" for t in titles) + f"\n\nPassages:\n\n{numbered}\n\n"
    if asked:
        prompt += "Already asked:\n" + "\n".join(f"- {q}" for q in asked) + "\n\n"
    return prompt + f"Researcher's question: {question}\n\nSuggest six questions to ask next."


def parse_questions(raw: str, limit: int = 3) -> list[str]:
    """Questions from {"questions": [...]} or, if the model ignored the format, from lines ending in '?'."""
    import json

    match = re.search(r"\{.*\}?", raw, re.S)  # the closing brace may be cut off
    if match:
        try:
            items = json.loads(match.group(0)).get("questions") or []
            questions = [" ".join(str(q).split()) for q in items if str(q).strip()]
            if questions:
                return list(dict.fromkeys(q if q.endswith("?") else q + "?" for q in questions))[:limit]
        except (ValueError, AttributeError):  # almost-JSON: take every quoted question
            quoted = [" ".join(_unescape(q).split()) for q in re.findall(r'"((?:[^"\\]|\\.)*\?)"', match.group(0))]
            if quoted:
                return list(dict.fromkeys(quoted))[:limit]
    return list(dict.fromkeys(m.group(1).strip() for line in raw.splitlines()
                              if (m := _QUESTION_LINE.match(line))))[:limit]


_QUESTION_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*(.+\?)\s*$")


_JSON_FIELD = r'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"'


def _unescape(text: str) -> str:
    import json

    try:
        return json.loads(f'"{text}"')
    except ValueError:
        return text.replace('\\"', '"')


def parse_explanation(raw: str, limit: int = 3) -> tuple[str, list[str]]:
    """(explanation, up to ``limit`` follow-up questions) from the model's reply.

    Valid JSON is read directly. Small local models sometimes return almost-JSON (a missing bracket or
    comma), so the "explanation" string and every quoted question are then recovered one by one. If there is
    no JSON at all, lines ending in '?' are the questions and the rest is the explanation. Raw JSON is never
    shown to the user."""
    import json

    text = raw.strip()
    explanation, items = "", []
    match = re.search(r"\{.*\}?", text, re.S)
    if match and '"' in match.group(0):
        blob = match.group(0)
        try:
            data = json.loads(blob)
            explanation = str(data.get("explanation", ""))
            items = data.get("follow_ups") or data.get("questions") or data.get("follow_up_questions") or []
        except (ValueError, AttributeError):
            found = re.search(_JSON_FIELD.format(name="explanation"), blob, re.S)
            explanation = _unescape(found.group(1)) if found else ""
            tail = blob[found.end():] if found else blob
            items = [_unescape(q) for q in re.findall(r'"((?:[^"\\]|\\.)*\?)"', tail)]
        questions = [" ".join(str(q).split()) for q in items if str(q).strip()]
        questions = list(dict.fromkeys(q if q.endswith("?") else q + "?" for q in questions))[:limit]
        return " ".join(explanation.split()), questions
    lines = [line for line in text.splitlines() if line.strip()]
    questions = [m.group(1).strip() for line in lines if (m := _QUESTION_LINE.match(line))]
    body = [line.strip() for line in lines if not _QUESTION_LINE.match(line)
            and not re.match(r"^\s*(follow[- ]?up|questions?)\b.*:\s*$", line, re.I)]
    return " ".join(body).strip(), list(dict.fromkeys(questions))[:limit]