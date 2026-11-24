"""Streamlit front-end for Thesora. All retrieval and generation logic lives in the FastAPI backend.

Each browser tab gets its own backend chat session. A page refresh (or "New chat") starts a fresh
session, so earlier PDFs and references are never reused. The look (light "paper" / dark "ink"
themes) is defined in ``ui/styles.py``; the theme choice is kept in the URL (``?theme=dark``) so it
survives a refresh.
"""
from __future__ import annotations

import html
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

# `streamlit run ui/streamlit_app.py` only puts ui/ on sys.path; add the project root so the
# UI shares app/config.py (and its env overrides) with the backend instead of duplicating it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from ui.styles import DARK, LIGHT, PANEL_CSS, build_css  # noqa: E402

try:  # N9 PDF viewer component; falls back to a page image when missing
    from streamlit_pdf_viewer import pdf_viewer
except Exception:  # pragma: no cover
    pdf_viewer = None

DEFAULT_API_URL = config.API_URL
REQUEST_TIMEOUT_S = config.UI_REQUEST_TIMEOUT_S
MAX_FILES_PER_REQUEST = config.MAX_FILES_PER_REQUEST
HEALTH_TIMEOUT_S = config.UI_HEALTH_TIMEOUT_S
SNIPPET_CHARS = config.UI_SNIPPET_CHARS
PROGRESS_POLL_S = config.UI_PROGRESS_POLL_S
PERCENT = 100
HTTP_NOT_FOUND = 404
HTTP_UNAVAILABLE = 503

MODE_ASK_LABEL = "Ask a question"
MODE_COMPARE_LABEL = "Compare papers"
MODE_AGREE_LABEL = "Agree?"
MODES = {MODE_ASK_LABEL: "ask", MODE_COMPARE_LABEL: "compare", MODE_AGREE_LABEL: "agree"}
_CITE_RE = re.compile(r"\[(\d+)\]")

Reply = tuple[Optional[Any], Optional[str], Optional[int]]


# ============================================================================ backend calls
def _request(base_url: str, method: str, path: str, **kwargs: Any) -> Reply:
    """Call the backend; returns ``(json, error_message, status)``. Safe to run in a thread."""
    url = f"{base_url.rstrip('/')}{path}"
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_S)
    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the Thesora backend at {base_url}. Is uvicorn running?", None
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond (CPU generation can be slow).", None
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}", None
    try:
        body = response.json()
    except ValueError:
        return None, f"Backend returned HTTP {response.status_code} with a non-JSON body.", response.status_code
    if not response.ok:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        prefix = "LLM unavailable: " if response.status_code == HTTP_UNAVAILABLE else ""
        return body, f"{prefix}{detail}", response.status_code
    return body, None, response.status_code


def call_api(method: str, path: str, label: Optional[str] = None, **kwargs: Any) -> Reply:
    """Backend call; with ``label`` it shows a live elapsed-seconds indicator while waiting."""
    base_url = st.session_state.api_url
    if label is None:
        return _request(base_url, method, path, **kwargs)
    placeholder = st.empty()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_request, base_url, method, path, **kwargs)
        while not future.done():
            elapsed = time.perf_counter() - started
            placeholder.markdown(notice("info", "…", label, f"{elapsed:.0f}s elapsed"), unsafe_allow_html=True)
            time.sleep(PROGRESS_POLL_S)
    placeholder.empty()
    return future.result()


def new_chat(delete_old: bool = True) -> None:
    """Start a fresh backend session and clear the conversation."""
    old = st.session_state.get("session_id")
    if delete_old and old:
        call_api("DELETE", f"/sessions/{old}", timeout=HEALTH_TIMEOUT_S)
    data, err, _ = call_api("POST", "/sessions", timeout=HEALTH_TIMEOUT_S)
    st.session_state.session_id = None if err else data["session_id"]
    st.session_state.session_error = err
    st.session_state.history = []
    st.session_state.ingest_results = []
    st.session_state.refine_result = None
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def session_path(suffix: str = "") -> str:
    return f"/sessions/{st.session_state.session_id}{suffix}"


def handle_expired(status: Optional[int]) -> bool:
    """If the backend forgot this chat (404), start a new one. Returns True when that happened."""
    if status == HTTP_NOT_FOUND:
        new_chat(delete_old=False)
        st.session_state.expired_notice = True
        return True
    return False


# ============================================================================ HTML helpers
def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def notice(kind: str, icon: str, title: str, body: str = "") -> str:
    """A calm inline notice (kind: info / good / warn / bad)."""
    body_html = f'<div class="th-notice-body">{esc(body)}</div>' if body else ""
    return (
        f'<div class="th-notice {kind}"><div class="th-notice-icon">{esc(icon)}</div>'
        f'<div><div class="th-notice-title">{esc(title)}</div>{body_html}</div></div>'
    )


def pill(kind: str, text: str) -> str:
    return f'<span class="th-pill {kind}"><span class="th-dot"></span>{esc(text)}</span>'


def with_citation_badges(answer: str, unverified: set[int]) -> str:
    """Escape the model's text and turn ``[n]`` markers into citation badges."""
    safe = html.escape(answer, quote=False).replace("&gt;", ">")  # keep markdown quotes; no raw tags survive

    def badge(match: re.Match) -> str:
        number = int(match.group(1))
        extra = " unverified" if number in unverified else ""
        return f'<span class="cite{extra}">{number}</span>'

    return _CITE_RE.sub(badge, safe)


def confidence_meter(confidence: float) -> str:
    value = min(1.0, max(0.0, confidence))
    if value >= config.UI_CONFIDENCE_HIGH:
        kind, word = "good", "High"
    elif value >= config.UI_CONFIDENCE_MEDIUM:
        kind, word = "warn", "Medium"
    else:
        kind, word = "bad", "Low"
    return (
        f'<div class="th-conf"><div class="th-conf-top"><span>Confidence · {word}</span>'
        f'<b>{value * PERCENT:.0f}%</b></div><div class="th-conf-track">'
        f'<div class="th-conf-fill {kind}" style="width:{value * PERCENT:.0f}%"></div></div></div>'
    )


def card() -> Any:
    """A bordered container styled as a card (see `.th-card-mark` in styles.py)."""
    box = st.container(border=True)
    box.markdown('<span class="th-card-mark"></span>', unsafe_allow_html=True)
    return box


# ============================================================================ rendering
def render_question(prompt: str, mode: str) -> None:
    label = {"compare": "Compare", "agree": "Agree?"}.get(mode, "Question")
    st.markdown(
        f'<div class="th-q"><div class="th-q-bubble"><span class="th-q-mode">{label}</span>{esc(prompt)}</div></div>',
        unsafe_allow_html=True,
    )


def sources_title(result: dict[str, Any], count: int) -> str:
    return f"Closest passages · {count}" if result["abstained"] else f"Sources · {count}"


def render_result(result: dict[str, Any]) -> None:
    """One answer card: notices, answer with citation badges, confidence meter and source cards."""
    unverified = set(result.get("unverified_citations") or [])
    mode = result.get("mode", "ask")
    with card():
        title = "Comparison" if mode == "synthesize" else "Answer"
        head, menu = st.columns([12, 1], vertical_alignment="center")
        head.markdown(
            f'<div class="th-answer-head"><span class="th-answer-label">{title}</span>'
            f'<span class="th-meta">{result.get("latency_s", 0)} s</span></div>',
            unsafe_allow_html=True,
        )
        if not result["abstained"]:
            with menu.popover("⋯", help="Checks and explain"):
                render_answer_menu(result)
        if result["abstained"]:
            st.markdown(
                notice("warn", "?", "Not answered from your papers", result.get("message") or config.ABSTAIN_MESSAGE),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(with_citation_badges(result["answer"], unverified), unsafe_allow_html=True)
            if unverified:
                numbers = ", ".join(f"[{n}]" for n in sorted(unverified))
                st.markdown(
                    notice("warn", "!", f"Check citation {numbers}",
                           "These claims could not be matched word-for-word to the cited passage. "
                           "Open the source below before relying on them."),
                    unsafe_allow_html=True,
                )
            elif result.get("message"):
                st.markdown(notice("info", "i", "Note", result["message"]), unsafe_allow_html=True)
        if result.get("invalid_citations"):
            numbers = ", ".join(f"[{n}]" for n in result["invalid_citations"])
            st.markdown(notice("bad", "×", f"Invalid citation {numbers}", "The model cited a passage that does not exist."),
                        unsafe_allow_html=True)
        render_trust_extras(result)
        st.markdown(confidence_meter(float(result.get("confidence", 0.0))), unsafe_allow_html=True)
        render_sources(result, result.get("sources", []))


def _key() -> str:
    st.session_state._btn = st.session_state.get("_btn", 0) + 1
    return f"btn{st.session_state._btn}"


SOURCE_BADGE = {True: "✓ Verified", None: "Cited", False: "! Unverified"}


def _source_badge(source: dict[str, Any]) -> str:
    if source["verified"] is None:
        return "Cited" if source["cited"] else "Context"
    if not source["cited"]:
        return "Not cited"
    return SOURCE_BADGE[bool(source["verified"])]


def _open_panel(**panel: Any) -> None:
    st.session_state.panel = panel
    st.rerun()


def render_sources(result: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    """One compact dropdown per source; inside: the passage, then 📄 Open page and 💡 Explain."""
    if not sources:
        return
    st.markdown(f'<div class="th-sources-title">{sources_title(result, len(sources))}</div>', unsafe_allow_html=True)
    for src in sources:
        rel = min(1.0, max(0.0, float(src.get("relevance", 0.0))))
        label = f"[{src['index']}]  {src.get('label') or src['doc_id']}  ·  {_source_badge(src)}  ·  rel {rel:.2f}"
        with st.expander(label):
            text = " ".join(src["text"].split())
            st.markdown(f'<div class="th-src-text" style="padding:0;border:0">'
                        f'{esc(text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "…")}</div>',
                        unsafe_allow_html=True)
            if src.get("meaning"):
                st.caption(f"Meaning check: {src['meaning']}")
            open_col, explain_col, _ = st.columns([1, 1, 2])
            if open_col.button("📄 Open page", key=_key(), help="Show this page with the passage highlighted"):
                _open_panel(tab="citation", doc_id=src["doc_id"], page=src["page_number"], passage=src["text"],
                            label=src.get("label") or src["doc_id"])
            if result.get("answer_id") and explain_col.button("💡 Explain", key=_key(), help="Explain this passage"):
                _open_panel(tab="explain", answer_id=result["answer_id"], text=text[:1500])


def render_answer_menu(result: dict[str, Any]) -> None:
    """The ⋯ menu: sentence check, table check and explain-the-answer, shown only when asked."""
    sentences = result.get("sentences") or []
    if sentences:
        st.markdown("**Sentence check**")
        for item in sentences:
            st.markdown(f"{STATUS_ICON.get(item['status'], item['status'])} · {esc(item['text'])}")
        if result.get("nli_available") is False:
            st.caption("Checked by wording only (meaning check is off).")
    cells = result.get("table_cells") or []
    if cells:
        st.markdown("**Table check: is each cell in the paper?**")
        for cell in cells:
            ev = ", ".join(f"[{e['passage']}] p.{e['page']}" for e in cell["evidence"])
            st.markdown(f"{STATUS_ICON.get(cell['status'], cell['status'])} · {esc(cell['paper'])} · {esc(cell['column'])}"
                        + (f" — {ev}" if ev else ""))
    if result.get("answer_id") and st.button("💡 Explain this answer", key=_key(), use_container_width=True):
        _open_panel(tab="explain", answer_id=result["answer_id"],
                    text=re.sub(r"\[\d+\]", "", result.get("answer", ""))[:1500])
    if not (sentences or cells or result.get("answer_id")):
        st.caption("No extra checks for this answer.")


def render_panel(panel: dict[str, Any]) -> None:
    """Right-hand panel (about 30%): Citation tab (PDF page with highlight) and Explain tab."""
    if st.button("✕ Close panel", key=_key(), use_container_width=True):
        st.session_state.pop("panel", None)
        st.rerun()
    names = ["Citation", "Explain"] if panel.get("tab") == "citation" else ["Explain", "Citation"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Citation"]:
        if panel.get("doc_id"):
            render_citation(panel)
        else:
            st.caption("Click a 📄 source button under an answer to open its page here.")
    with tabs["Explain"]:
        if panel.get("answer_id"):
            render_explain(panel)
        else:
            st.caption("Click 💡 Explain under an answer to get a plain-English explanation here.")


def render_citation(panel: dict[str, Any]) -> None:
    base = f"{st.session_state.api_url}{session_path('/documents/')}{requests.utils.quote(panel['doc_id'])}"
    st.markdown(f"**{esc(panel['label'])}**")
    hl, err, _ = call_api("POST", session_path("/highlight"),
                          json={"doc_id": panel["doc_id"], "page": panel["page"], "passage": panel["passage"]})
    if err:
        st.markdown(notice("bad", "×", "Could not open the page", err), unsafe_allow_html=True)
        return
    if not hl["found"]:
        st.caption("The exact passage could not be located on this page, so it is shown without a highlight.")
    try:
        if pdf_viewer is None or config.PDF_PANEL_MODE == "image":
            raise RuntimeError("image mode")
        pdf = requests.get(f"{base}/pdf", timeout=REQUEST_TIMEOUT_S).content
        pdf_viewer(pdf, width=config.VIEWER_WIDTH_PX, zoom_level="auto", annotations=hl["annotations"], pages_to_render=[panel["page"]],
                   key=f"pdf_{panel['doc_id']}_{panel['page']}")
    except Exception:
        png = requests.get(f"{base}/page/{panel['page']}.png", params={"passage": panel["passage"][:1500]},
                           timeout=REQUEST_TIMEOUT_S).content
        st.image(png, use_container_width=True)


def render_explain(panel: dict[str, Any]) -> None:
    text = st.text_area("Text to explain (edit to select just part of it)", value=panel.get("text", ""), height=140,
                        key=f"exp_text_{panel['answer_id']}")
    follow = st.text_input("Follow-up question (optional)", key=f"exp_follow_{panel['answer_id']}")
    if st.button("💡 Explain", type="primary", key=_key()):
        data, err, _ = call_api("POST", session_path("/explain"), label="Explaining",
                                json={"answer_id": panel["answer_id"], "selected_text": text or "",
                                      "follow_up": follow or None, "previous": panel.get("last")})
        panel["result"] = {"error": err} if err else data
        if data and not err:
            panel["last"] = data["explanation"]
    result = panel.get("result")
    if result and result.get("error"):
        st.markdown(notice("bad", "×", "Could not explain", result["error"]), unsafe_allow_html=True)
    elif result:
        for g in result.get("glossary", []):
            st.markdown(f"**From the paper:** {esc(g['abbr'])} = {esc(g['long_form'])}, p.{g['page']}")
        st.caption(result["label"])
        st.markdown(result["explanation"])


STATUS_ICON = {"verified": "✓ Verified", "weak": "~ Weak", "unverified": "! Unverified", "uncited": "○ Uncited",
               "supported": "✓ Supported", "partly": "~ Partly supported", "not_found": "✗ Not found in text",
               "not_stated": "– Not stated"}


def render_trust_extras(result: dict[str, Any]) -> None:
    """Always visible but compact: part chips, repair note and silent papers. Detailed checks live in the ⋯ menu."""
    for part in result.get("parts") or []:
        icon = "✓ Answered" if part["answered"] else "✗ Not found in your papers"
        st.markdown(f'<div class="th-chip">{icon} · {esc(part["question"])}</div>', unsafe_allow_html=True)
    repair = result.get("repair") or {}
    if repair.get("removed"):
        n = repair["removed"]
        st.caption(f"ⓘ {n} claim{'s' if n != 1 else ''} removed because it could not be verified against the papers.")
    if result.get("silent_papers"):
        st.caption("No evidence from: " + ", ".join(result["silent_papers"]) + " — silence is not disagreement.")


def render_agree(data: dict[str, Any]) -> None:
    """N12: Paper / Position / Passage table, conflict badges with both quotes (icon + text)."""
    with card():
        st.markdown('<span class="th-answer-label">Do the papers agree?</span>', unsafe_allow_html=True)
        if data.get("message"):
            st.caption(data["message"])
        if not data.get("rows"):
            st.markdown(notice("warn", "?", "No paper discusses this topic", config.ABSTAIN_MESSAGE), unsafe_allow_html=True)
            return
        st.markdown("| Paper | Position (quoted) | Passage |\n| --- | --- | --- |\n" + "\n".join(
            f"| {r['label']} | {r['position'].replace('|', '/')} | p.{r['page']} {r['section']} |" for r in data["rows"]))
        show_all = st.toggle("Show wording-only differences", key=_key())
        shown = [c for c in data.get("conflicts", []) if show_all or c["severity"] > 1]
        if not shown:
            st.markdown("**✓ No conflicts found** between the papers' statements.")
        for c in shown:
            icon = {3: "⚠️ Contradiction", 2: "⚠️ Numbers differ", 1: "~ Wording differs"}[c["severity"]]
            with st.expander(f"{icon} · {c['papers'][0]} vs {c['papers'][1]}"):
                st.markdown(f"{esc(c['reason'])}\n\n> {esc(c['quotes'][0])}\n\n> {esc(c['quotes'][1])}")


def render_turn(turn: dict[str, Any]) -> None:
    render_question(turn["prompt"], turn.get("mode", "ask"))
    if turn.get("mode") == "agree" and not turn.get("error"):
        render_agree(turn["data"])
        return
    if turn.get("error"):
        with card():
            st.markdown(notice("bad", "×", "Something went wrong", turn["error"]), unsafe_allow_html=True)
    else:
        render_result(turn["data"])


def render_empty_state(has_documents: bool) -> None:
    first = "Done — your papers are ready." if has_documents else "Drop up to 10 PDFs in the sidebar and press Ingest."
    st.markdown(
        '<div class="th-steps">'
        f'<div class="th-step"><div class="th-step-n">01</div><div class="th-step-t">Add papers</div>'
        f'<div class="th-step-d">{esc(first)}</div></div>'
        '<div class="th-step"><div class="th-step-n">02</div><div class="th-step-t">Ask or compare</div>'
        '<div class="th-step-d">Ask a precise question, or switch to Compare papers for a side-by-side table '
        'and research gaps.</div></div>'
        '<div class="th-step"><div class="th-step-n">03</div><div class="th-step-t">Check the evidence</div>'
        '<div class="th-step-d">Every claim carries a citation. Open a source to read the exact passage it came '
        'from.</div></div></div>'
        '<div class="th-legend">'
        + pill("good", "Verified: claim matches its passage")
        + pill("warn", "Unverified: check the passage")
        + pill("", "Abstains when the papers don't say")
        + "</div>",
        unsafe_allow_html=True,
    )


# ============================================================================ sidebar
def render_ingest_results(results: list[dict[str, Any]]) -> None:
    """One summary line for the files that worked, plus one notice per skipped or failed file."""
    ingested = [r for r in results if r["status"] == "ingested"]
    if ingested:
        passages = sum(r.get("chunks_added", 0) for r in ingested)
        names = ", ".join(r["document"] for r in ingested)
        title = f"{len(ingested)} paper{'s' if len(ingested) != 1 else ''} indexed · {passages} passages"
        st.markdown(notice("good", "✓", title, names), unsafe_allow_html=True)
    for row in results:
        if row["status"] == "skipped":
            st.markdown(notice("info", "i", row["document"], "Already in this chat, so it was skipped."),
                        unsafe_allow_html=True)
        elif row["status"] == "failed":
            st.markdown(notice("bad", "×", row["document"], row["error"]), unsafe_allow_html=True)


def sidebar() -> list[dict[str, Any]]:
    """Brand, backend status, New chat, upload (max 10 per request) and this chat's documents."""
    documents: list[dict[str, Any]] = []
    with st.sidebar:
        st.markdown(
            '<div class="th-brand"><div class="th-logo">T</div><div><div class="th-wordmark">Thesora</div>'
            '<div class="th-tagline">Evidence-grounded research assistant</div></div></div>',
            unsafe_allow_html=True,
        )
        health, error, _ = call_api("GET", "/health", timeout=HEALTH_TIMEOUT_S)
        if error:
            status = pill("bad", "Backend offline")
        elif not health["ollama"]["reachable"]:
            status = pill("warn", "Ollama not running")
        elif not health["ollama"]["model_installed"]:
            status = pill("warn", f"Model missing: {health['ollama']['model']}")
        else:
            status = pill("good", f"Ready · {health['ollama']['model']}")
        st.markdown(f'<div style="margin:.7rem 0 .9rem">{status}</div>', unsafe_allow_html=True)
        if error and "too long" in error:
            st.markdown(notice("warn", "!", "Backend is starting or busy",
                               "It is running but did not answer in time. On the first start it loads the "
                               "search models (up to a minute on a laptop). Wait until terminal 1 is quiet, "
                               "then press Retry."), unsafe_allow_html=True)
            if st.button("↻  Retry", use_container_width=True):
                st.rerun()
        elif error:
            st.markdown(notice("bad", "×", "Cannot reach the backend", error), unsafe_allow_html=True)
        elif not health["ollama"]["reachable"]:
            st.markdown(notice("warn", "!", "Start Ollama", "Open the Ollama app or run `ollama serve`."),
                        unsafe_allow_html=True)
        elif not health["ollama"]["model_installed"]:
            st.markdown(notice("warn", "!", "Pull the model", f"Run: ollama pull {health['ollama']['model']}"),
                        unsafe_allow_html=True)

        if st.button("＋  New chat", use_container_width=True):
            new_chat()
            st.rerun()
        if st.session_state.session_error and not error:
            new_chat(delete_old=False)  # backend is reachable again: start the chat that failed earlier
        if st.session_state.session_error:
            st.markdown(notice("bad", "×", "Could not start a chat", st.session_state.session_error),
                        unsafe_allow_html=True)
            return documents

        st.markdown('<div class="th-eyebrow">Library</div>', unsafe_allow_html=True)
        files = st.file_uploader(
            f"Add PDFs · up to {MAX_FILES_PER_REQUEST} per upload",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        for f in files or []:
            if f.size > config.LARGE_FILE_WARN_MB * 1024 * 1024:
                st.markdown(notice("warn", "!", f"{f.name}: large file ({f.size / 1048576:.0f} MB)",
                                   "It may use more memory and take longer to read."), unsafe_allow_html=True)
        too_many = bool(files) and len(files) > MAX_FILES_PER_REQUEST
        if too_many:
            st.markdown(
                notice("bad", "×", f"{len(files)} files selected",
                       f"Remove some: at most {MAX_FILES_PER_REQUEST} PDFs per upload."),
                unsafe_allow_html=True,
            )
        if st.button("Ingest papers", type="primary", use_container_width=True, disabled=not files or too_many):
            payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
            data, err, status = call_api("POST", session_path("/ingest"), label="Reading and indexing PDFs", files=payload)
            if not handle_expired(status):
                st.session_state.ingest_results = (data or {}).get("results", [])
                if err and not data:
                    st.session_state.ingest_results = [{"document": "Upload", "status": "failed", "error": err}]
        render_ingest_results(st.session_state.ingest_results)

        docs, _, status = call_api("GET", session_path("/documents"), timeout=HEALTH_TIMEOUT_S)
        if handle_expired(status):
            return documents
        documents = (docs or {}).get("documents", [])
        st.markdown('<div class="th-eyebrow">In this chat</div>', unsafe_allow_html=True)
        if documents:
            cards = "".join(
                f'<div class="th-doc"><div class="th-doc-icon">PDF</div><div><div class="th-doc-name">'
                f'{esc(d["doc_id"])}</div><div class="th-doc-meta">{d["pages"]} pages · {d["chunks"]} passages'
                f"</div></div></div>"
                for d in documents
            )
            total = sum(d["chunks"] for d in documents)
            st.markdown(
                cards + f'<div class="th-stat">{len(documents)} paper{"s" if len(documents) != 1 else ""} · '
                f"{total} passages</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="th-empty">No papers yet. Every new chat starts empty.</div>',
                        unsafe_allow_html=True)
        if documents:
            gl, _, _ = call_api("GET", session_path("/glossary"), timeout=HEALTH_TIMEOUT_S)
            terms = (gl or {}).get("glossary", [])
            with st.expander(f"Glossary ({len(terms)})"):
                for g in terms:
                    st.markdown(f"**{esc(g['abbr'])}** — {esc(g['long_form'])} · {esc(g['doc_id'])} p.{g['page']}")
                if not terms:
                    st.caption("No abbreviations defined in these papers.")
            try:
                bib_resp = requests.get(f"{st.session_state.api_url}{session_path('/bibtex')}", timeout=HEALTH_TIMEOUT_S)
                notes_resp = requests.get(f"{st.session_state.api_url}{session_path('/export')}", timeout=HEALTH_TIMEOUT_S)
                st.download_button("Download .bib", bib_resp.text, file_name="thesora.bib", use_container_width=True)
                st.download_button("Download notes (.md)", notes_resp.text, file_name="thesora_notes.md",
                                   use_container_width=True)
            except requests.exceptions.RequestException:
                st.caption("Exports are unavailable while the backend is unreachable.")
        st.markdown(
            f'<div class="th-footer">Max {MAX_FILES_PER_REQUEST} PDFs per upload · {config.MAX_UPLOAD_MB} MB each.<br>'
            "Papers live only in this chat's memory; nothing is saved to disk.</div>",
            unsafe_allow_html=True,
        )
    return documents


# ============================================================================ prompt refiner (prompt-master)
def refiner_panel(mode: str) -> None:
    """Optional helper: rewrite a rough question into a sharper one. Nothing is sent unless the user chooses."""
    if st.session_state.pop("refine_edit", None) is not None:
        st.session_state.refine_text = st.session_state.refine_result["refined"]
        st.session_state.refine_result = None
    with card():
        st.markdown(
            '<div class="th-refine-label">✨ Prompt refiner</div>'
            '<div class="th-refine-why" style="margin:.2rem 0 .5rem">Type a rough question; Thesora rewrites it into a '
            "precise one that retrieves better. You choose what to send.</div>",
            unsafe_allow_html=True,
        )
        target = "topic to compare" if mode == "compare" else "question"
        st.text_area(f"Your rough {target}", key="refine_text", height=80, label_visibility="collapsed",
                     placeholder=f"e.g. {'methods and results' if mode == 'compare' else 'brain parts deja vu'}")
        # Always enabled: Streamlit commits typed text only when the box loses focus, so a disabled
        # button would swallow the first click.
        if st.button("✨ Refine", type="primary"):
            if not st.session_state.refine_text.strip():
                st.session_state.refine_result = {"error": "Type a rough question in the box first."}
            else:
                data, err, status = call_api(
                    "POST", session_path("/refine"), label="Refining your question",
                    json={"text": st.session_state.refine_text, "mode": "compare" if mode == "agree" else mode},
                )
                if not handle_expired(status):
                    st.session_state.refine_result = data if not err else {"error": err}
        result = st.session_state.refine_result
        if result and result.get("error"):
            st.markdown(notice("bad", "×", "Could not refine", result["error"]), unsafe_allow_html=True)
        elif result:
            st.markdown(
                f'<div class="th-refine"><div class="th-refine-label">Suggested {target}</div>'
                f'<div class="th-refine-text">{esc(result["refined"])}</div>'
                f'<div class="th-refine-why">{esc(result["rationale"])}</div></div>',
                unsafe_allow_html=True,
            )
            use, edit, original = st.columns(3)
            if use.button("Use suggestion", type="primary", use_container_width=True):
                st.session_state.pending_prompt = result["refined"]
                st.session_state.refine_result = None
                st.rerun()
            if edit.button("Edit it", use_container_width=True):
                st.session_state.refine_edit = True
                st.rerun()
            if original.button("Send my original", use_container_width=True):
                st.session_state.pending_prompt = result["original"]
                st.session_state.refine_result = None
                st.rerun()


# ============================================================================ page
def init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("ingest_results", [])
    st.session_state.setdefault("refine_result", None)
    st.session_state.setdefault("refine_text", "")
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = st.query_params.get("theme", config.UI_DEFAULT_THEME) == DARK
    if "session_id" not in st.session_state:  # first load or browser refresh: brand-new chat
        new_chat(delete_old=False)


def run_prompt(prompt: str, mode: str) -> None:
    render_question(prompt, mode)
    if mode == "agree":
        data, err, status = call_api("POST", session_path("/agree"), label="Comparing positions", json={"topic": prompt})
        if handle_expired(status):
            st.rerun()
        st.session_state.history.append({"prompt": prompt, "mode": mode, "error": err} if err else
                                        {"prompt": prompt, "mode": mode, "data": data})
        if err:
            st.markdown(notice("bad", "×", "Something went wrong", err), unsafe_allow_html=True)
        else:
            render_agree(data)
        return
    if mode == "compare":
        data, err, status = call_api("POST", session_path("/synthesize"), label="Comparing papers",
                                     json={"topic": prompt})
    else:
        data, err, status = call_api("POST", session_path("/ask"), label="Retrieving, reranking and verifying",
                                     json={"question": prompt})
    if handle_expired(status):
        st.rerun()
    turn = {"prompt": prompt, "mode": mode, "error": err} if err else {"prompt": prompt, "mode": mode, "data": data}
    st.session_state.history.append(turn)
    if err:
        with card():
            st.markdown(notice("bad", "×", "Something went wrong", err), unsafe_allow_html=True)
    else:
        render_result(data)


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Thesora · Research assistant", page_icon="📚", layout="wide")
    init_state()
    theme = DARK if st.session_state.dark_mode else LIGHT
    st.markdown(build_css(theme), unsafe_allow_html=True)
    documents = sidebar()

    hero, toggle = st.columns([5, 1], vertical_alignment="center")
    hero.markdown(
        '<div class="th-hero"><h1>Ask your papers</h1>'
        "<p>Answers come only from the PDFs in this chat, with every claim cited and checked.</p></div>",
        unsafe_allow_html=True,
    )
    toggle.toggle("🌙 Dark", key="dark_mode")
    st.query_params["theme"] = DARK if st.session_state.dark_mode else LIGHT
    if st.session_state.pop("expired_notice", False):
        st.markdown(notice("warn", "!", "That chat had ended", "The backend restarted, so a new empty chat was started. "
                           "Upload your PDFs again."), unsafe_allow_html=True)

    modebar = st.container(key="th_modebar")  # stays pinned at the top while the chat scrolls
    controls, refine_toggle = modebar.columns([3, 2], vertical_alignment="center")
    mode_label = controls.segmented_control("Mode", list(MODES), default=MODE_ASK_LABEL, key="mode_label",
                                            label_visibility="collapsed") or MODE_ASK_LABEL
    mode = MODES[mode_label]
    show_refiner = refine_toggle.toggle("✨ Prompt refiner", key="show_refiner",
                                        help="Optional: rewrite a rough question into a precise one before sending.")

    # The chat box is pinned to the bottom wherever it is created; reading it first lets the welcome
    # cards disappear as soon as the first question is sent.
    placeholder = {"ask": "Ask a question about your papers…", "compare": "Topic to compare across papers…",
                   "agree": "Topic to check: do the papers agree?"}[mode]
    prompt = st.chat_input(placeholder, disabled=not st.session_state.session_id)
    pending = st.session_state.pop("pending_prompt", None) or prompt

    st.session_state._btn = 0
    panel = st.session_state.get("panel")
    if panel:
        st.markdown(PANEL_CSS, unsafe_allow_html=True)  # wider page while the side panel is open
    main_col, side_col = st.columns([7, 3]) if panel else (st.container(), None)
    with main_col:
        if not st.session_state.history and not pending:
            render_empty_state(bool(documents))
        for turn in st.session_state.history:
            render_turn(turn)
        if pending:
            run_prompt(pending, mode)
        if show_refiner:
            refiner_panel(mode)
    if side_col is not None:
        with side_col, st.container(key="th_sidepanel"):  # follows the reader down the chat
            render_panel(panel)


main()
