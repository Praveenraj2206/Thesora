"""Readable display of a source passage: paragraphs instead of PDF line breaks, words re-joined across
line-end hyphens, headings in bold, and table passages shown as tables."""
from __future__ import annotations

import html
import re
from typing import Optional

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z](\.\d+)*)?\s*[A-Z][^.!?]{0,80}$")


def is_table(text: str) -> bool:
    return "|---" in text or "| ---" in text


def _join_lines(paragraph: str) -> str:
    paragraph = _HYPHEN_BREAK.sub(r"\1\2", paragraph)
    return " ".join(paragraph.split())


def paragraphs(text: str) -> list[tuple[str, str]]:
    """[(kind, text)] with kind 'heading' or 'para'. Blank lines in the passage separate PDF text blocks."""
    out: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        joined = _join_lines(block)
        if not joined:
            continue
        words = joined.split()
        kind = "heading" if len(words) <= 8 and _HEADING.match(joined) and not joined.endswith((",", ";")) else "para"
        if kind == "para" and out and out[-1][0] == "para" and not out[-1][1].endswith((".", "!", "?", ":", "”", '"')):
            out[-1] = ("para", f"{out[-1][1]} {joined}")  # a sentence continued in the next column / block
        else:
            out.append((kind, joined))
    return out


def _mark(para: str, marks: list[str]) -> str:
    """Escape ``para`` and wrap the parts that are evidence sentences in <mark>. A sentence that continues
    in the next passage is marked from its start to the end of this paragraph."""
    spans: list[tuple[int, int]] = []
    for sentence in marks:
        sentence = " ".join(sentence.split())
        head = sentence[: min(60, len(sentence))]
        start = para.find(head)
        if start < 0:
            continue
        spans.append((start, min(len(para), start + len(sentence))))
    out, at = [], 0
    for start, end in sorted(spans):
        if start < at:
            continue
        out += [html.escape(para[at:start]), f"<mark>{html.escape(para[start:end])}</mark>"]
        at = end
    out.append(html.escape(para[at:]))
    return "".join(out)


def readable_html(text: str, limit: int, marks: Optional[list[str]] = None) -> str:
    """HTML for a prose passage, cut after ``limit`` characters at a word boundary; ``marks`` (the answer's
    evidence sentences) are highlighted."""
    parts, used = [], 0
    for kind, para in paragraphs(text):
        if used >= limit:
            parts.append('<p class="th-src-more">…</p>')
            break
        if used + len(para) > limit:
            para = para[: max(0, limit - used)].rsplit(" ", 1)[0] + " …"
        used += len(para)
        tag = "strong" if kind == "heading" else "span"
        body = _mark(para, marks or []) if kind == "para" else html.escape(para)
        parts.append(f'<p class="th-src-{kind}"><{tag}>{body}</{tag}></p>')
    return "".join(parts)


def table_markdown(text: str) -> tuple[str, str]:
    """(caption, markdown table) for a table passage."""
    caption, _, table = text.partition("\n\n") if not text.lstrip().startswith("|") else ("", "", text)
    return caption.strip(), (table or text).strip()
