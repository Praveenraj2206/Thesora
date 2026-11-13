"""FastAPI backend for Thesora.

Every chat is a session with its own in-memory corpus, so a new chat (or a
browser refresh, which starts a new chat in the UI) never sees PDFs or
references from an earlier one.

Endpoints
  GET    /health                              status, Ollama, limits, live sessions
  POST   /sessions                            start a new, empty chat -> {session_id}
  GET    /sessions/{session_id}               documents + chunk count of that chat
  DELETE /sessions/{session_id}               end the chat and forget its PDFs
  POST   /sessions/{session_id}/ingest        upload 1..MAX_FILES_PER_REQUEST PDFs (field "files")
  GET    /sessions/{session_id}/documents     documents in that chat
  POST   /sessions/{session_id}/ask           {"question"} -> grounded, verified answer
  POST   /sessions/{session_id}/synthesize    {"topic"}    -> comparison table + research gaps

Errors follow INT-2: a JSON ``{"detail": "<friendly, actionable message>"}`` with
400 (bad input), 404 (unknown/expired chat), 413 (file too large), 422 (PDF
unreadable), 502 (LLM error reply) or 503 (Ollama not running / model missing).
"""
from __future__ import annotations

import logging
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import config
from app.core.errors import (
    EmbeddingError,
    LLMGenerationError,
    LLMUnavailableError,
    PDFLoadError,
    SessionNotFoundError,
    StoreError,
    ThesoraError,
    UploadError,
)
from app.pipelines.sessions import SessionManager

logger = logging.getLogger(__name__)

HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_LARGE = 413
HTTP_UNPROCESSABLE = 422
HTTP_INTERNAL = 500
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
MIN_TEXT_CHARS = 3
MAX_TEXT_CHARS = 2000
BYTES_PER_MB = 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]")

_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_manager() -> SessionManager:
    """Return the process-wide SessionManager, creating it on first use."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = SessionManager()
        return _manager


def set_manager(manager: Optional[SessionManager]) -> None:
    """Replace the SessionManager (tests inject one with stub models)."""
    global _manager
    with _manager_lock:
        _manager = manager


def _warm_up_background() -> None:
    """Load models in a background thread so startup is not blocked."""
    try:
        get_manager().warm_up()
    except Exception as exc:  # warm-up is best effort; requests will report real errors
        logger.warning("Warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start model warm-up in the background."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    threading.Thread(target=_warm_up_background, daemon=True).start()
    yield


app = FastAPI(
    title="Thesora",
    description="Evidence-grounded research assistant (RAG) with per-chat document sessions",
    version="2.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """Body of POST /sessions/{id}/ask."""

    question: str = Field(min_length=MIN_TEXT_CHARS, max_length=MAX_TEXT_CHARS)


class SynthesizeRequest(BaseModel):
    """Body of POST /sessions/{id}/synthesize."""

    topic: str = Field(min_length=MIN_TEXT_CHARS, max_length=MAX_TEXT_CHARS)


# ----------------------------------------------------------------------------- error mapping
def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message, **extra})


@app.exception_handler(SessionNotFoundError)
async def _session_missing(_: Request, exc: SessionNotFoundError) -> JSONResponse:
    return _error(HTTP_NOT_FOUND, str(exc))


@app.exception_handler(UploadError)
async def _upload_error(_: Request, exc: UploadError) -> JSONResponse:
    return _error(exc.status_code, str(exc))


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
    return _error(HTTP_UNAVAILABLE, str(exc))


@app.exception_handler(LLMGenerationError)
async def _llm_generation(_: Request, exc: LLMGenerationError) -> JSONResponse:
    return _error(HTTP_BAD_GATEWAY, str(exc))


@app.exception_handler(PDFLoadError)
async def _pdf_error(_: Request, exc: PDFLoadError) -> JSONResponse:
    return _error(HTTP_UNPROCESSABLE, str(exc))


@app.exception_handler(EmbeddingError)
@app.exception_handler(StoreError)
async def _internal_known(_: Request, exc: ThesoraError) -> JSONResponse:
    return _error(HTTP_INTERNAL, str(exc))


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
    return _error(HTTP_BAD_REQUEST, str(exc))


def describe_validation_errors(errors: Sequence[dict[str, Any]]) -> str:
    """Turn pydantic validation errors into one friendly, actionable sentence (INT-2)."""
    messages = []
    for error in errors:
        field = next((str(part) for part in reversed(error.get("loc", ())) if isinstance(part, str)), "body")
        kind = error.get("type", "")
        if kind == "json_invalid":
            messages.append("The request body is not valid JSON. Send e.g. {\"question\": \"...\"}.")
        elif kind == "missing":
            messages.append(f"'{field}' is required.")
        elif kind == "string_too_short":
            messages.append(f"'{field}' is too short: write at least {MIN_TEXT_CHARS} characters.")
        elif kind == "string_too_long":
            messages.append(f"'{field}' is too long: keep it under {MAX_TEXT_CHARS} characters.")
        else:
            messages.append(f"'{field}': {error.get('msg', 'invalid value')}.")
    return " ".join(dict.fromkeys(messages)) or "The request is invalid."


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(HTTP_UNPROCESSABLE, describe_validation_errors(exc.errors()))


# ----------------------------------------------------------------------------- health + sessions
@app.get("/health")
def health() -> dict[str, Any]:
    """Service status, Ollama reachability, limits and number of live chats."""
    manager = get_manager()
    return {
        "status": "ok",
        "ollama": manager.llm.health(),
        "active_sessions": manager.count(),
        "limits": {
            "max_files_per_request": config.MAX_FILES_PER_REQUEST,
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "session_ttl_min": config.SESSION_TTL_MIN,
        },
    }


@app.post("/sessions", status_code=201)
def create_session() -> dict[str, Any]:
    """Start a new chat with an empty corpus."""
    return get_manager().create().summary()


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Documents and chunk count of one chat."""
    return get_manager().get(session_id).summary()


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    """End a chat and forget its PDFs and references."""
    if not get_manager().delete(session_id):
        raise SessionNotFoundError("This chat session has already ended. Start a new chat.")
    return {"session_id": session_id, "deleted": True}


@app.get("/sessions/{session_id}/documents")
def list_documents(session_id: str) -> dict[str, Any]:
    """List the documents uploaded to this chat."""
    return {"documents": get_manager().get(session_id).assistant.documents()}


# ----------------------------------------------------------------------------- ingestion
def safe_filename(raw: Optional[str]) -> str:
    """Validate and sanitise an uploaded file name (must end in .pdf)."""
    original = Path(raw or "").name.strip()
    if not original:
        raise UploadError("A file has no name.")
    if not original.lower().endswith(".pdf"):
        raise UploadError(f"{original}: only .pdf files are accepted.")
    safe = _SAFE_NAME_RE.sub("_", original)
    if len(safe) > config.MAX_FILENAME_CHARS:
        stem = safe[: config.MAX_FILENAME_CHARS - len(".pdf")]
        safe = f"{stem}.pdf"
    return safe


def save_upload(file: UploadFile, folder: Path) -> Path:
    """Stream one upload into ``folder`` after checking type and size. Returns the saved path."""
    name = safe_filename(file.filename)
    destination = folder / name
    limit = config.MAX_UPLOAD_MB * BYTES_PER_MB
    written = 0
    with open(destination, "wb") as handle:
        first = True
        while True:
            block = file.file.read(config.UPLOAD_READ_CHUNK_BYTES)
            if not block:
                break
            if first and not block.startswith(config.PDF_MAGIC):
                raise UploadError(f"{name}: the file is not a valid PDF.")
            first = False
            written += len(block)
            if written > limit:
                raise UploadError(f"{name}: file exceeds the {config.MAX_UPLOAD_MB} MB limit.", HTTP_TOO_LARGE)
            handle.write(block)
    if written == 0:
        raise UploadError(f"{name}: the uploaded file is empty.")
    return destination


def check_file_count(count: int) -> None:
    """Enforce 1..MAX_FILES_PER_REQUEST files per upload request."""
    if count == 0:
        raise UploadError("No files were uploaded. Attach at least one PDF in the 'files' field.")
    if count > config.MAX_FILES_PER_REQUEST:
        raise UploadError(
            f"You sent {count} files, but at most {config.MAX_FILES_PER_REQUEST} PDFs are allowed per upload. "
            "Split them into smaller batches."
        )


@app.post("/sessions/{session_id}/ingest")
def ingest(session_id: str, files: list[UploadFile] = File(default=[])) -> JSONResponse:
    """Upload up to MAX_FILES_PER_REQUEST PDFs into this chat.

    Each file is reported separately (``ingested`` / ``skipped`` duplicate / ``failed`` with a reason).
    Files are parsed from a private temporary folder that is deleted afterwards, so no PDF is kept on disk.
    Returns 200 if at least one file was ingested or already present, otherwise 422.
    """
    check_file_count(len(files))
    session = get_manager().get(session_id)
    results: list[dict[str, Any]] = []
    with session.lock, tempfile.TemporaryDirectory(prefix="thesora-") as temp:
        for upload in files:
            label = Path(upload.filename or "unnamed").name
            try:
                path = save_upload(upload, Path(temp))
                results.append(session.assistant.ingest_pdf(path, doc_id=path.name))
            except (UploadError, PDFLoadError) as exc:
                results.append({"document": label, "status": "failed", "error": str(exc)})
            except ThesoraError as exc:
                logger.error("Ingest of %s failed: %s", label, exc)
                results.append({"document": label, "status": "failed", "error": str(exc)})
            finally:
                upload.file.close()
    counts = {status: sum(1 for r in results if r["status"] == status) for status in ("ingested", "skipped", "failed")}
    body = {
        "session_id": session_id,
        "results": results,
        **counts,
        "documents": session.assistant.documents(),
    }
    ok = counts["ingested"] + counts["skipped"] > 0
    if not ok:
        body["detail"] = "None of the files could be ingested. " + " ".join(r["error"] for r in results)
    return JSONResponse(status_code=200 if ok else HTTP_UNPROCESSABLE, content=body)


# ----------------------------------------------------------------------------- question answering
@app.post("/sessions/{session_id}/ask")
def ask(session_id: str, request: AskRequest) -> dict[str, Any]:
    """Answer from this chat's PDFs only, with citations, verification, confidence and abstention."""
    return get_manager().get(session_id).assistant.ask(request.question).to_dict()


@app.post("/sessions/{session_id}/synthesize")
def synthesize(session_id: str, request: SynthesizeRequest) -> dict[str, Any]:
    """Cross-paper comparison table plus research gaps over this chat's PDFs."""
    return get_manager().get(session_id).assistant.synthesize(request.topic).to_dict()
