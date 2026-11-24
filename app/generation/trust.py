"""Batch A trust features: NLI meaning check (N8), per-sentence status + silent papers (N5),
table-cell verification (N3), self-repair (N10) and paper metadata / BibTeX / notes (N4).

Everything here protects the promise "never claim anything the papers don't say": it only
labels, removes or exports claims; it never adds new ones.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from app import config
from app.generation.verifier import CITATION_RE, _segments, extract_cited_numbers, score_claim

logger = logging.getLogger(__name__)

PASS, WEAK, FAIL = "PASS", "WEAK", "FAIL"
MEANING_LABEL = {PASS: "entailed", WEAK: "weak", FAIL: "contradicted"}
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_NOT_STATED = re.compile(r"^\s*(not stated|n/?a|-)\s*\.?\s*$", re.IGNORECASE)
REPAIR_SYSTEM_PROMPT = (
    "Rewrite the answer using ONLY the numbered passages. Keep a [n] citation on every sentence. "
    "Omit anything you cannot support from the passages. Do not add new information."
)


# ----------------------------------------------------------------------------- N8 NLI
class NLIChecker:
    """Lazy local NLI cross-encoder. ``available`` is False when the model cannot load (offline, disabled)."""

    def __init__(self, model_name: str = config.NLI_MODEL, enabled: bool = config.NLI_ENABLED) -> None:
        self.model_name, self.enabled = model_name, enabled
        self._model: Any = None
        self._failed = not enabled
        self._lock = threading.Lock()
        self._labels: list[str] = []

    def _load(self) -> Any:
        if self._failed:
            return None
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
                    id2label = getattr(self._model.model.config, "id2label", {}) or {}
                    self._labels = [str(id2label.get(i, "")).lower() for i in range(len(id2label))]
                except Exception as exc:  # offline, missing model, bad install: fall back to words only
                    logger.warning("NLI model unavailable (%s); using word check only.", exc)
                    self._failed = True
        return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    def probabilities(self, premise: str, hypothesis: str) -> Optional[dict[str, float]]:
        model = self._load()
        if model is None:
            return None
        import numpy as np

        logits = np.asarray(model.predict([(premise, hypothesis)]))[0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        labels = self._labels or ["contradiction", "entailment", "neutral"]
        return {labels[i]: float(p) for i, p in enumerate(probs)}

    def judge(self, premise: str, hypothesis: str) -> Optional[str]:
        probs = self.probabilities(premise, hypothesis)
        if probs is None:
            return None
        return classify(probs)


def classify(probs: Mapping[str, float]) -> str:
    """PASS / WEAK / FAIL from NLI probabilities; contradiction always fails."""
    if probs.get("contradiction", 0.0) >= config.NLI_FAIL_CONTRADICTION:
        return FAIL
    if probs.get("entailment", 0.0) >= config.NLI_PASS:
        return PASS
    return WEAK


def best_premise(passage: str, claim: str, n: int = config.NLI_PREMISE_SENTENCES) -> str:
    """The 1–2 passage sentences with the most word overlap with the claim (keeps NLI input short)."""
    claim_words = set(_WORD_RE.findall(claim.lower()))
    sentences = [s for s in _SENT_RE.split(" ".join(passage.split())) if s.strip()]
    ranked = sorted(sentences, key=lambda s: len(claim_words & set(_WORD_RE.findall(s.lower()))), reverse=True)
    return " ".join(ranked[:n]) or passage


# ----------------------------------------------------------------------------- N5 sentences
def sentence_report(answer: str, chunk_texts: Mapping[int, str], nli: Optional[NLIChecker] = None,
                    subjects: Optional[Mapping[int, str]] = None) -> list[dict[str, Any]]:
    """One entry per answer sentence: text, citations, status (verified|weak|unverified|uncited), score, meaning."""
    report, subjects = [], subjects or {}
    for segment in _segments(answer):
        text = CITATION_RE.sub("", segment).strip(" -*•|#>\t")
        if not text:
            continue
        cites = [n for n in extract_cited_numbers(segment) if n in chunk_texts]
        if not cites:
            report.append({"text": text, "citations": [], "status": "uncited", "score": 0.0, "meaning": None})
            continue
        best_ok, best_score, best_n = False, 0.0, cites[0]
        for n in cites:
            ok, score = score_claim(text, chunk_texts[n], subjects.get(n, ""))
            if (ok, score) > (best_ok, best_score):
                best_ok, best_score, best_n = ok, score, n
        meaning = nli.judge(best_premise(chunk_texts[best_n], text), text) if nli is not None else None
        if meaning == FAIL or not best_ok and best_score < config.SENTENCE_WEAK_SCORE:
            status = "unverified"
        elif best_ok and meaning in (None, PASS):
            status = "verified"
        else:
            status = "weak"
        report.append({"text": text, "citations": cites, "status": status, "score": round(best_score, 1),
                       "meaning": MEANING_LABEL.get(meaning) if meaning else None})
    return report


def silent_papers(all_docs: Sequence[str], relevances_by_doc: Mapping[str, float]) -> list[str]:
    """Papers with no passage at or above the relevance floor for this question."""
    return [d for d in all_docs if relevances_by_doc.get(d, 0.0) < config.RELEVANCE_FLOOR]


# ----------------------------------------------------------------------------- N10 repair
def drop_unsupported(answer: str, report: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    """Remove sentences whose status is unverified; returns (new_answer, removed_count)."""
    bad = {r["text"] for r in report if r["status"] == "unverified"}
    kept, removed = [], 0
    for segment in _segments(answer):
        text = CITATION_RE.sub("", segment).strip(" -*•|#>\t")
        if text in bad:
            removed += 1
            continue
        kept.append(segment.strip())
    return " ".join(kept).strip(), removed


# ----------------------------------------------------------------------------- N3 table cells
def _claims_in_cell(cell: str) -> list[str]:
    text = CITATION_RE.sub("", cell)
    parts = re.split(r"[;]|(?<=[.!?])\s+", text)
    return [p.strip(" .") for p in parts if len(p.strip().split()) >= 2]


def verify_table(answer: str, passages: Sequence[Mapping[str, Any]], nli: Optional[NLIChecker] = None) -> list[dict[str, Any]]:
    """Verify each comparison-table cell against that row's own paper passages.

    ``passages``: [{index, doc_id, page, section, text}]. Returns one entry per cell:
    {row, paper, column, text, status: supported|partly|not_found|not_stated, evidence}.
    """
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    cells: list[dict[str, Any]] = []
    for r, line in enumerate(lines[2:], start=1):
        cols = [c.strip() for c in line.strip("|").split("|")]
        if not cols:
            continue
        paper = cols[0]
        own = [p for p in passages if p["doc_id"] in paper or Path(p["doc_id"]).stem in paper] or list(passages)
        for c, cell in enumerate(cols[1:], start=1):
            column = header[c] if c < len(header) else f"Column {c}"
            if _NOT_STATED.match(CITATION_RE.sub("", cell)):
                cells.append({"row": r, "paper": paper, "column": column, "text": cell, "status": "not_stated", "evidence": []})
                continue
            claims = _claims_in_cell(cell)
            supported, evidence = 0, []
            for claim in claims:
                best = max(((score_claim(claim, p["text"]), p) for p in own), key=lambda t: t[0], default=None)
                if best is None:
                    continue
                (ok, score), p = best
                if ok and nli is not None and nli.judge(best_premise(p["text"], claim), claim) == FAIL:
                    ok = False
                if ok:
                    supported += 1
                    evidence.append({"passage": p["index"], "page": p["page"], "section": p["section"], "score": round(score, 1)})
            status = "not_found" if not claims or supported == 0 else ("supported" if supported == len(claims) else "partly")
            cells.append({"row": r, "paper": paper, "column": column, "text": cell, "status": status, "evidence": evidence})
    return cells


# ----------------------------------------------------------------------------- N4 metadata
_ARXIV_ID_RE = re.compile(r"arXiv:\s*(\d{2})(\d{2})\.\d{4,5}", re.IGNORECASE)
_HEADER_JUNK_RE = re.compile(
    r"^(arxiv\b|a\s*p\s*re\s*print\b|preprint\b|proceedings\b|article$|nature\b.*\|"
    r"|vol(ume)?\b|©|\(c\)|copyright|https?://|www\.|doi\b|accepted\b|published\b|received\b"
    r"|under review|journal\b|page \d|\d+$)",
    re.IGNORECASE,
)
_ABSTRACT_RE = re.compile(r"^(abstract|a\s?b\s?s\s?t\s?r\s?a\s?c\s?t|1\.?\s+introduction|introduction)\b", re.IGNORECASE)
_NAME_RE = re.compile(r"^[A-Z][\w'’.\-]*(?:\s+[A-Z][\w'’.\-]*){1,3}$")
_NOT_NAME_WORDS = frozenset(
    "university institute department school laboratory lab college center centre inc llc ltd "
    "corporation research foundation hospital abstract introduction preprint article house group team "
    "independent researcher program programme project company department faculty academy "
    "united states kingdom republic county province city hampshire york jersey paper working conceptual draft".split()
)
_AUTHOR_MARKS_RE = re.compile(r"[*∗†‡§¶#✉\d]")
_BIG_GAP = "\u2003"  # marks a wide horizontal gap between two author names on one line


def _page_lines(page: Any) -> list[dict[str, Any]]:
    """Horizontal text lines of one page: text (spaces rebuilt), size, top y; wide gaps kept as a marker."""
    from app.ingestion.pdf_loader import line_text

    lines = []
    for block in page.get_text("rawdict", sort=True).get("blocks", []):
        for raw in block.get("lines", []):
            direction = raw.get("dir", (1, 0))
            if abs(direction[0] - 1) > 0.01:  # rotated text: arXiv side stamps, margin notes
                continue
            spans = [sp for sp in raw.get("spans", []) if any(not c["c"].isspace() for c in sp.get("chars", []))]
            if not spans:
                continue
            pieces, prev_x1 = [], None
            for sp in spans:
                x0 = sp["chars"][0]["bbox"][0] if sp.get("chars") else sp["bbox"][0]
                if prev_x1 is not None and x0 - prev_x1 > config.METADATA_NAME_GAP_EM * float(sp.get("size", 10)):
                    pieces.append(_BIG_GAP)
                pieces.append(line_text({"spans": [sp]}))
                prev_x1 = sp["chars"][-1]["bbox"][2] if sp.get("chars") else sp["bbox"][2]
            text = re.sub(r"\s*\u2003\s*", _BIG_GAP, re.sub(r"[ \t]+", " ", " ".join(pieces).strip()))
            if text.strip(_BIG_GAP):
                lines.append({"text": text, "size": max(float(sp.get("size", 0)) for sp in spans),
                              "y": float(raw["bbox"][1]), "x": float(raw["bbox"][0])})
    lines.sort(key=lambda ln: (round(ln["y"] / 3), ln["x"]))  # reading order: top to bottom, then left to right
    return lines


def _plain(text: str) -> str:
    return " ".join(text.replace(_BIG_GAP, " ").split())


def _detect_year(all_text: str, header_text: str) -> Optional[int]:
    """arXiv identifier first (2601.xxxxx -> 2026), then the latest year in the page-1 header
    (journal line, conference dates, ©, preprint date), then the most frequent year on pages 1-2."""
    import datetime

    latest = datetime.date.today().year + 1
    arxiv = _ARXIV_ID_RE.search(all_text)
    if arxiv and 7 <= int(arxiv.group(1)) and 1 <= int(arxiv.group(2)) <= 12:
        year = 2000 + int(arxiv.group(1))
        if year <= latest:
            return year
    header_years = [int(y) for y in _YEAR_RE.findall(header_text) if int(y) <= latest]
    if header_years:
        return max(header_years)
    years = [y for y in _YEAR_RE.findall(all_text) if int(y) <= latest]
    return int(max(set(years), key=years.count)) if years else None


def _clean_title(title: str, stem: str) -> str:
    title = _plain(title)
    if title[:1] in ":;-–—" and stem:  # title starts after a logo (e.g. "[logo]: Organized ...")
        title = f"{stem}{title if title.startswith(':') else ': ' + title.lstrip(';-–— ')}"
    return title.strip(" :;-–—")


def _authors_from(lines: Sequence[Mapping[str, Any]]) -> list[str]:
    authors: list[str] = []
    for line in lines:
        text = line["text"]
        if _ABSTRACT_RE.match(_plain(text)) or len(_plain(text)) > config.METADATA_AUTHOR_MAX_CHARS:
            break
        for part in re.split(r",|;|&|\band\b|" + _BIG_GAP, text):
            name = " ".join(_AUTHOR_MARKS_RE.sub(" ", part).split()).strip(" .")
            words = {w.lower().strip(".,") for w in name.split()}
            if _NAME_RE.match(name) and not words & _NOT_NAME_WORDS and name not in authors:
                authors.append(name)
        if len(authors) >= 8:
            break
    return authors[:8]


def extract_metadata(pdf_path: str | Path, doc_id: str) -> dict[str, Any]:
    """Title, authors, year and DOI from page 1 (best effort; every field is marked auto-detected).

    Skips rotated arXiv stamps and running headers such as "A PREPRINT", falls back to the PDF's
    own title field and then the file name; the year comes from the arXiv id or the page-1 header.
    """
    stem = Path(doc_id).stem.replace("_", " ").strip()
    meta: dict[str, Any] = {"doc_id": doc_id, "title": stem, "authors": [], "year": None, "doi": None,
                            "auto": True, "edited": False}
    try:
        import fitz

        with fitz.open(str(pdf_path)) as pdf:
            if not pdf.page_count:
                return meta
            first = pdf[0]
            lines = _page_lines(first)
            abstract_at = next((i for i, ln in enumerate(lines) if _ABSTRACT_RE.match(_plain(ln["text"]))), None)
            limit_y = first.rect.height * config.METADATA_HEADER_FRACTION
            head = lines[:abstract_at] if abstract_at is not None else [ln for ln in lines if ln["y"] <= limit_y]
            candidates = [i for i, ln in enumerate(head)
                          if not _HEADER_JUNK_RE.match(_plain(ln["text"])) and "@" not in ln["text"]
                          and sum(ch.isalpha() for ch in ln["text"]) >= 3]
            if candidates:
                top = max(head[i]["size"] for i in candidates)
                first_i = next(i for i in candidates if head[i]["size"] >= top - 0.5)
                last_i = first_i
                while (last_i + 1 < len(head) and head[last_i + 1]["size"] >= top - 0.5
                       and head[last_i + 1]["y"] - head[last_i]["y"] <= config.METADATA_TITLE_LINE_GAP * top):
                    last_i += 1
                title = _clean_title(" ".join(head[i]["text"] for i in range(first_i, last_i + 1)), stem)
                if len(title.split()) >= 3:
                    meta["title"] = title[: config.METADATA_TITLE_MAX_CHARS]
                meta["authors"] = _authors_from(head[last_i + 1:last_i + 1 + config.METADATA_AUTHOR_LINES])
            info = pdf.metadata or {}
            info_title = (info.get("title") or "").strip()
            if meta["title"] == stem and len(info_title.split()) >= 3 and not re.match(
                    r"(microsoft word|untitled|\S+\.(docx?|tex|pdf))", info_title, re.IGNORECASE):
                meta["title"] = info_title
            if info.get("author") and not meta["authors"]:
                meta["authors"] = [a.strip() for a in re.split(r",|;| and ", info["author"]) if a.strip()][:8]
            all_text = " ".join(pdf[i].get_text() for i in range(min(2, pdf.page_count)))
            header_text = " ".join(_plain(ln["text"]) for ln in head) + " " + " ".join(
                _plain(ln["text"]) for ln in lines if _HEADER_JUNK_RE.match(_plain(ln["text"])))
            meta["year"] = _detect_year(all_text, header_text)
            doi = _DOI_RE.search(all_text)
            meta["doi"] = doi.group(0).rstrip(".,;)") if doi else None
    except Exception as exc:  # metadata is best effort; the file name stays the title
        logger.info("Metadata extraction failed for %s: %s", doc_id, exc)
    return meta


def short_label(meta: Optional[Mapping[str, Any]], page: int, section: str) -> str:
    """'Smith et al., 2023 · p.4 · Results' (falls back to the title or file name)."""
    if not meta:
        who = "Unknown"
    else:
        authors = meta.get("authors") or []
        surname = authors[0].split()[-1] if authors else (meta.get("title") or meta.get("doc_id", ""))[:40]
        who = f"{surname} et al." if len(authors) > 1 else surname
        if meta.get("year"):
            who += f", {meta['year']}"
    parts = [who, f"p.{page}"] + ([section] if section else [])
    return " · ".join(parts)


def bibtex(metas: Sequence[Mapping[str, Any]]) -> str:
    entries = []
    for m in metas:
        authors = m.get("authors") or []
        key = re.sub(r"\W", "", (authors[0].split()[-1] if authors else Path(m["doc_id"]).stem)) + str(m.get("year") or "")
        fields = [f"  title = {{{m.get('title') or m['doc_id']}}}"]
        if authors:
            fields.append(f"  author = {{{' and '.join(authors)}}}")
        if m.get("year"):
            fields.append(f"  year = {{{m['year']}}}")
        if m.get("doi"):
            fields.append(f"  doi = {{{m['doi']}}}")
        fields.append(f"  note = {{{'Auto-detected by Thesora; check before use' if not m.get('edited') else 'Checked by user'}}}")
        entries.append(f"@article{{{key or 'paper'},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + ("\n" if entries else "")


def notes_markdown(history: Sequence[Mapping[str, Any]]) -> str:
    """Research notes: questions, answers and only the verified citations. No LLM calls."""
    lines = ["# Thesora research notes", ""]
    for turn in history:
        lines.append(f"## {turn['question']}")
        if turn.get("abstained"):
            lines += ["", "_Not answered from the papers._", ""]
            continue
        lines += ["", turn.get("answer", ""), ""]
        verified = [s for s in turn.get("sources", []) if s.get("verified")]
        if verified:
            lines.append("**Verified sources:**")
            lines += [f"- [{s['index']}] {s.get('label') or s['doc_id']}" for s in verified]
            lines.append("")
    return "\n".join(lines)
