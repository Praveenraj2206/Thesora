"""Basic Streamlit front-end for Thesora. All logic lives in the FastAPI backend.

Each browser tab gets its own backend chat session. A page refresh (or the
"New chat" button) starts a fresh session, so earlier PDFs and references are
never reused.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

# `streamlit run ui/streamlit_app.py` only puts ui/ on sys.path; add the project root so the
# UI shares app/config.py (and its env overrides) with the backend instead of duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

DEFAULT_API_URL = config.API_URL
REQUEST_TIMEOUT_S = config.UI_REQUEST_TIMEOUT_S
MAX_FILES_PER_REQUEST = config.MAX_FILES_PER_REQUEST
HEALTH_TIMEOUT_S = config.UI_HEALTH_TIMEOUT_S
SNIPPET_CHARS = config.UI_SNIPPET_CHARS
PERCENT = 100
PROGRESS_POLL_S = config.UI_PROGRESS_POLL_S
HTTP_NOT_FOUND = 404
HTTP_UNAVAILABLE = 503

Reply = tuple[Optional[Any], Optional[str], Optional[int]]


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
            placeholder.info(f"⏳ {label} {time.perf_counter() - started:.0f}s elapsed")
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
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def session_path(suffix: str = "") -> str:
    return f"/sessions/{st.session_state.session_id}{suffix}"


def handle_expired(status: Optional[int]) -> bool:
    """If the backend forgot this chat (404), start a new one. Returns True when that happened."""
    if status == HTTP_NOT_FOUND:
        new_chat(delete_old=False)
        st.warning("That chat had ended on the server, so a new empty chat was started. Upload your PDFs again.")
        return True
    return False


def render_result(result: dict[str, Any]) -> None:
    """Show an answer: abstention banner, text, unverified citations, confidence and sources."""
    unverified = result.get("unverified_citations") or []
    if result["abstained"]:
        st.warning(f"**Abstained.** {result.get('message') or ''}")
    else:
        st.markdown(result["answer"])
        if unverified:
            st.warning("Citations that could not be verified: " + ", ".join(f"[{n}]" for n in unverified))
        elif result.get("message"):
            st.info(result["message"])
    confidence = float(result.get("confidence", 0.0))
    st.progress(min(1.0, max(0.0, confidence)), text=f"Confidence: {confidence * PERCENT:.0f}%")
    st.caption(f"{result.get('latency_s', 0)} s · {result.get('mode', 'ask')}")
    if result.get("invalid_citations"):
        st.error("The model cited sources that do not exist: " + ", ".join(f"[{n}]" for n in result["invalid_citations"]))
    for source in result.get("sources", []):
        if source["verified"] is True:
            badge = "✅ verified"
        elif source["verified"] is None:
            badge = "cited" if source["cited"] else "not cited"
        elif source["cited"]:
            badge = "⚠️ unverified"
        else:
            badge = "not cited"
        section = f" · {source['section']}" if source.get("section") else ""
        label = f"[{source['index']}] {source['doc_id']} · p.{source['page_number']}{section} · {badge}"
        with st.expander(label):
            st.caption(f"relevance {source['relevance']:.2f}")
            text = source["text"]
            st.write(text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "…")


def sidebar() -> None:
    """Backend status, New chat, upload (max 10 per request) and this chat's documents."""
    with st.sidebar:
        st.title("Thesora")
        health, error, _ = call_api("GET", "/health", timeout=HEALTH_TIMEOUT_S)
        if error:
            st.error(error)
        else:
            ollama = health["ollama"]
            if not ollama["reachable"]:
                st.warning("Ollama is not reachable. Run `ollama serve`.")
            elif not ollama["model_installed"]:
                st.warning(f"Model missing. Run `ollama pull {ollama['model']}`.")
            else:
                st.success(f"Ollama ready ({ollama['model']})")

        if st.button("➕ New chat", use_container_width=True):
            new_chat()
            st.rerun()
        if st.session_state.session_error:
            st.error(st.session_state.session_error)
            return

        st.subheader("Upload PDFs")
        files = st.file_uploader(
            f"Up to {MAX_FILES_PER_REQUEST} PDFs per upload",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        too_many = bool(files) and len(files) > MAX_FILES_PER_REQUEST
        if too_many:
            st.error(f"{len(files)} files selected. Remove some: at most {MAX_FILES_PER_REQUEST} per upload.")
        if st.button("Ingest", disabled=not files or too_many):
            payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
            data, err, status = call_api("POST", session_path("/ingest"), label="Reading PDFs…", files=payload)
            if not handle_expired(status):
                for row in (data or {}).get("results", []):
                    if row["status"] == "ingested":
                        st.success(f"{row['document']}: {row['chunks_added']} chunks")
                    elif row["status"] == "skipped":
                        st.info(f"{row['document']}: already in this chat")
                    else:
                        st.error(f"{row['document']}: {row['error']}")
                if err and not data:
                    st.error(err)

        docs, err, status = call_api("GET", session_path("/documents"), timeout=HEALTH_TIMEOUT_S)
        if handle_expired(status):
            return
        st.subheader("Documents in this chat")
        if docs and docs["documents"]:
            for doc in docs["documents"]:
                st.write(f"• {doc['doc_id']} ({doc['pages']} pages, {doc['chunks']} chunks)")
        else:
            st.caption("None yet. Every new chat starts empty.")


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Thesora", page_icon="📚", layout="wide")
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    if "session_id" not in st.session_state:  # first load or browser refresh: brand-new chat
        new_chat(delete_old=False)
    sidebar()

    st.header("Ask your papers")
    mode = st.radio("Mode", ["Ask a question", "Compare papers (synthesis)"], horizontal=True)
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["prompt"])
        with st.chat_message("assistant"):
            if turn.get("error"):
                st.error(turn["error"])
            else:
                render_result(turn["data"])

    placeholder = "Ask about your PDFs…" if mode.startswith("Ask") else "Topic to compare across papers…"
    prompt = st.chat_input(placeholder, disabled=not st.session_state.session_id)
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            if mode.startswith("Ask"):
                data, err, status = call_api(
                    "POST", session_path("/ask"), label="Retrieving, reranking and generating…", json={"question": prompt}
                )
            else:
                data, err, status = call_api(
                    "POST", session_path("/synthesize"), label="Comparing papers…", json={"topic": prompt}
                )
            if handle_expired(status):
                return
            turn = {"prompt": prompt, "error": err} if err else {"prompt": prompt, "data": data}
            st.session_state.history.append(turn)
            if err:
                st.error(err)
            else:
                render_result(data)


main()
