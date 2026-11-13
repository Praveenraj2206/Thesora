"""LLM interface, the local Ollama implementation and the prompt templates.

``BaseLLM`` is the seam for swapping in a hosted API later: implement
``generate`` (and optionally ``health``) and pass the instance to
``ResearchAssistant``.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

import requests

from app import config
from app.core.errors import LLMGenerationError, LLMUnavailableError
from app.core.models import RetrievedChunk

logger = logging.getLogger(__name__)

HTTP_BAD_REQUEST = 400
_THINK_CLOSE = "</think>"
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are Thesora, a careful research assistant. Answer ONLY from the numbered context passages provided.\n"
    "Rules:\n"
    "1. Every factual sentence must end with citation markers such as [1] or [2][3] that refer to the passage numbers.\n"
    "2. Stay close to the wording of the passages. Never use outside knowledge and never invent facts.\n"
    "3. Cite only passage numbers that exist.\n"
    f"4. If the passages do not contain the answer, reply with exactly: {config.INSUFFICIENT_MARKER}\n"
    "5. Be concise: at most about 200 words."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are Thesora, a research synthesis assistant. Use ONLY the numbered context passages, which are grouped "
    "by paper. Output markdown with exactly these two parts and nothing before them:\n"
    "A comparison table that starts with the header line | Paper | Approach / Method | Key Results |, then the "
    "separator line | --- | --- | --- |, then EXACTLY ONE row per paper listed in the prompt (merge everything a "
    "paper's passages say into its single row). Put each table row on its own line. Put citation markers like [2] "
    "inside each cell. Write 'Not stated' when the passages do not say.\n"
    "Then a heading '## Research Gaps' followed by bullet points naming gaps, limitations or future work that the "
    "papers themselves state, each ending with citation markers.\n"
    "Never use outside knowledge. Cite only passage numbers that exist. "
    f"If the passages are insufficient for the topic, reply with exactly: {config.INSUFFICIENT_MARKER}"
)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render chunks as numbered passages with provenance headers."""
    blocks = []
    for number, item in enumerate(chunks, start=1):
        c = item.chunk
        section = f", section: {c.section}" if c.section else ""
        blocks.append(f"[{number}] (document: {c.doc_id}, page {c.page_number}{section})\n{c.text}")
    return "\n\n".join(blocks)


def build_ask_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """User prompt for question answering."""
    return f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer with citations:"


def build_synthesis_prompt(topic: str, chunks: Sequence[RetrievedChunk]) -> str:
    """User prompt for cross-paper synthesis.

    Passages are grouped under one header per paper (keeping their global ``[n]`` numbers, which
    match the response's sources) and the paper list is spelled out, so the small local model
    writes one table row per paper instead of one row per passage.
    """
    numbered = list(enumerate(chunks, start=1))
    papers = list(dict.fromkeys(item.chunk.doc_id for item in chunks))
    groups = []
    for paper in papers:
        passages = [
            f"[{number}] (page {item.chunk.page_number}"
            + (f", section: {item.chunk.section}" if item.chunk.section else "")
            + f")\n{item.chunk.text}"
            for number, item in numbered
            if item.chunk.doc_id == paper
        ]
        groups.append(f"=== Paper: {paper} ===\n" + "\n\n".join(passages))
    paper_list = ", ".join(papers)
    return (
        f"Context passages, grouped by paper:\n\n" + "\n\n".join(groups)
        + f"\n\nTopic: {topic}\n\nPapers (one table row each, in this order): {paper_list}"
        + "\n\nWrite the comparison table and research gaps:"
    )


def strip_think(text: str) -> str:
    """Remove chain-of-thought from a model reply.

    Handles three shapes: a full ``<think>...</think>`` block, a lone
    ``</think>`` (Qwen3-Thinking templates put the opening tag in the prompt, so
    the reply is "reasoning</think>answer"), and an unterminated ``<think>``.
    """
    closing = text.lower().rfind(_THINK_CLOSE)
    if closing != -1:
        text = text[closing + len(_THINK_CLOSE):]
    return _THINK_OPEN_RE.sub("", text).strip()


class BaseLLM(ABC):
    """Interface every LLM backend implements."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = config.ASK_MAX_TOKENS) -> str:
        """Return the model's text reply.

        Raises:
            LLMUnavailableError: backend unreachable or model missing.
            LLMGenerationError: backend replied with an error or empty text.
        """

    def health(self) -> dict[str, Any]:
        """Report backend status; hosted backends may keep the default."""
        return {"reachable": True, "model_installed": True, "model": "unknown"}


class OllamaLLM(BaseLLM):
    """Local Ollama chat backend (default model: qwen3:4b-instruct)."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout_s: int = config.OLLAMA_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self._think_supported = True

    def _unavailable(self, reason: str) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"{reason} Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {self.model}`)."
        )

    def _post_chat(self, payload: dict[str, Any]) -> requests.Response:
        """POST to ``/api/chat`` and translate network failures into clean errors."""
        try:
            return self._session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as exc:
            raise self._unavailable(f"Cannot connect to Ollama at {self.base_url}.") from exc
        except requests.exceptions.Timeout as exc:
            raise self._unavailable(f"Ollama did not answer within {self.timeout_s}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise self._unavailable(f"Request to Ollama failed: {exc}.") from exc

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = config.ASK_MAX_TOKENS) -> str:
        """Call Ollama's ``/api/chat`` and return the reply with thinking removed."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": max_tokens,
            },
        }
        if self._think_supported:
            payload["think"] = False
        response = self._post_chat(payload)
        if response.status_code == HTTP_BAD_REQUEST and "think" in response.text.lower():
            # This model rejects the 'think' option; remember that and retry once without it.
            logger.warning("Model %s rejected the 'think' option; retrying without it.", self.model)
            self._think_supported = False
            payload = {key: value for key, value in payload.items() if key != "think"}
            response = self._post_chat(payload)
        if response.status_code == 404:
            raise self._unavailable(f"Ollama does not have model '{self.model}'.")
        if not response.ok:
            raise LLMGenerationError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            body = response.json()
            content = body.get("message", {}).get("content", "")
        except ValueError as exc:
            raise LLMGenerationError("Ollama returned a non-JSON response.") from exc
        if body.get("done_reason") == "length" and _THINK_CLOSE not in content.lower():
            raise LLMGenerationError(
                f"The model hit the {max_tokens}-token limit before finishing its answer. This usually means a "
                f"'thinking' model spent the budget on reasoning; use a non-thinking model (set OLLAMA_MODEL) "
                f"or raise ASK_MAX_TOKENS."
            )
        cleaned = strip_think(content)
        if not cleaned:
            raise LLMGenerationError("Ollama returned an empty answer (try raising ASK_MAX_TOKENS).")
        return cleaned

    def health(self) -> dict[str, Any]:
        """Check that Ollama is reachable and the configured model is installed."""
        status: dict[str, Any] = {"reachable": False, "model_installed": False, "model": self.model}
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=config.OLLAMA_HEALTH_TIMEOUT_S)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.exceptions.RequestException, ValueError):
            return status
        status["reachable"] = True
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        status["model_installed"] = wanted in names
        return status
