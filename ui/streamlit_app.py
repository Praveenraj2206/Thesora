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
from ui.styles import DARK, LIGHT, build_css  # noqa: E402

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
MODES = {MODE_ASK_LABEL: "ask", MODE_COMPARE_LABEL: "compare"}
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


def source_card(source: dict[str, Any]) -> str:
    if source["verified"] is True:
        badge = '<span class="th-badge good">✓ Verified</span>'
    elif source["verified"] is None:
        badge = '<span class="th-badge muted">Cited</span>' if source["cited"] else '<span class="th-badge muted">Context</span>'
    elif source["cited"]:
        badge = '<span class="th-badge warn">! Unverified</span>'
    else:
        badge = '<span class="th-badge muted">Not cited</span>'
    location = f"Page {source['page_number']}"
    if source.get("section"):
        location += f" · {source['section']}"
    relevance = min(1.0, max(0.0, float(source.get("relevance", 0.0))))
    text = " ".join(source["text"].split())
    if len(text) > SNIPPET_CHARS:
        text = text[:SNIPPET_CHARS] + "…"
    return (
        '<details class="th-src"><summary>'
        f'<span class="th-src-n">{source["index"]}</span>'
        f'<span class="th-src-main"><div class="th-src-doc">{esc(source["doc_id"])}</div>'
        f'<div class="th-src-loc">{esc(location)}</div></span>'
        f'{badge}'
        f'<span class="th-src-rel"><div class="th-src-rel-track"><div class="th-src-rel-fill" '
        f'style="width:{relevance * PERCENT:.0f}%"></div></div>'
        f'<div class="th-src-rel-label">rel {relevance:.2f}</div></span>'
        '<span class="th-chev">▶</span></summary>'
        f'<div class="th-src-text">{esc(text)}</div></details>'
    )


def card() -> Any:
    """A bordered container styled as a card (see `.th-card-mark` in styles.py)."""
    box = st.container(border=True)
    box.markdown('<span class="th-card-mark"></span>', unsafe_allow_html=True)
    return box


# ============================================================================ rendering
def render_question(prompt: str, mode: str) -> None:
    label = "Compare" if mode == "compare" else "Question"
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
        st.markdown(
            f'<div class="th-answer-head"><span class="th-answer-label">{title}</span>'
            f'<span class="th-meta">{result.get("latency_s", 0)} s</span></div>',
            unsafe_allow_html=True,
        )
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
        st.markdown(confidence_meter(float(result.get("confidence", 0.0))), unsafe_allow_html=True)
        sources = result.get("sources", [])
        if sources:
            st.markdown(
                f'<div class="th-sources-title">{sources_title(result, len(sources))}</div>'
                + "".join(source_card(s) for s in sources),
                unsafe_allow_html=True,
            )


def render_turn(turn: dict[str, Any]) -> None:
    render_question(turn["prompt"], turn.get("mode", "ask"))
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
        if error:
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
                    json={"text": st.session_state.refine_text, "mode": mode},
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

    controls, refine_toggle = st.columns([3, 2], vertical_alignment="center")
    mode_label = controls.segmented_control("Mode", list(MODES), default=MODE_ASK_LABEL, key="mode_label",
                                            label_visibility="collapsed") or MODE_ASK_LABEL
    mode = MODES[mode_label]
    show_refiner = refine_toggle.toggle("✨ Prompt refiner", key="show_refiner",
                                        help="Optional: rewrite a rough question into a precise one before sending.")

    # The chat box is pinned to the bottom wherever it is created; reading it first lets the welcome
    # cards disappear as soon as the first question is sent.
    placeholder = "Ask a question about your papers…" if mode == "ask" else "Topic to compare across papers…"
    prompt = st.chat_input(placeholder, disabled=not st.session_state.session_id)
    pending = st.session_state.pop("pending_prompt", None) or prompt

    if not st.session_state.history and not pending:
        render_empty_state(bool(documents))
    for turn in st.session_state.history:
        render_turn(turn)
    if pending:
        run_prompt(pending, mode)
    if show_refiner:
        refiner_panel(mode)


main()
