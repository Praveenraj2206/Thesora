"""API tests for chat sessions and the upload limit.

Run with:  python -m unittest discover -s tests -v
Uses the real PDF parser, chunker, FAISS and BM25 on data/papers, with a small
hashing embedder, no cross-encoder and a stub LLM, so no model downloads or
Ollama are needed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.generation.llm import BaseLLM  # noqa: E402

PAPERS = sorted((ROOT / "data" / "papers").glob("*.pdf"))
_DEPS = all(importlib.util.find_spec(m) for m in ("fitz", "faiss", "langchain_text_splitters", "rank_bm25", "httpx"))
_WORD_RE = re.compile(r"\w+")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (stand-in for MiniLM)."""

    def warm_up(self) -> None:
        pass

    def embed(self, texts):
        out = np.zeros((len(texts), config.EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % config.EMBEDDING_DIM] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out

    def embed_query(self, text):
        return self.embed([text])


class NoReranker:
    """Cross-encoder that is 'unavailable', forcing the dense-cosine fallback."""

    def model(self):
        return None

    def logits(self, query, texts):
        raise RuntimeError("no cross-encoder in tests")


class EchoLLM(BaseLLM):
    """Replies with the first words of passage [1], cited, so verification passes."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt, user_prompt, max_tokens=0):
        self.calls += 1
        if "prompt engineer" in system_prompt:  # the optional question refiner
            return '{"refined": "Which brain regions are involved in deja vu?", "why": "Named the target."}'
        body = user_prompt.split("\n", 2)[2]  # skip "Context passages:" + first header line
        first = body.split("\n", 1)[1].split("\n\n[2]")[0]
        return " ".join(first.split()[:20]) + " [1]."


@unittest.skipUnless(_DEPS and PAPERS, "needs requirements.txt installed and PDFs in data/papers")
class SessionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.api import main
        from app.pipelines.sessions import SessionManager

        cls.llm = EchoLLM()
        cls.main = main
        main.set_manager(SessionManager(embedder=HashEmbedder(), reranker=NoReranker(), llm=cls.llm))
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.main.set_manager(None)

    def new_session(self) -> str:
        response = self.client.post("/sessions")
        self.assertEqual(response.status_code, 201)
        return response.json()["session_id"]

    def upload(self, session: str, paths):
        files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in paths]
        return self.client.post(f"/sessions/{session}/ingest", files=files)

    def test_more_than_ten_files_per_request_is_rejected(self):
        session = self.new_session()
        response = self.upload(session, [PAPERS[0]] * 11)
        self.assertEqual(response.status_code, 400)
        self.assertIn("at most 10", response.json()["detail"])
        self.assertEqual(self.client.get(f"/sessions/{session}/documents").json()["documents"], [])

    def test_exactly_ten_files_per_request_is_accepted(self):
        session = self.new_session()
        response = self.upload(session, [PAPERS[0]] * 10)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["ingested"], body["skipped"], body["failed"]), (1, 9, 0))

    def test_new_chat_does_not_see_old_pdfs_or_references(self):
        old = self.new_session()
        self.assertEqual(self.upload(old, PAPERS).json()["ingested"], len(PAPERS))
        answered = self.client.post(f"/sessions/{old}/ask", json={"question": "What is deja vu?"}).json()
        self.assertTrue(answered["sources"])

        fresh = self.new_session()
        self.assertEqual(self.client.get(f"/sessions/{fresh}/documents").json()["documents"], [])
        calls_before = self.llm.calls
        result = self.client.post(f"/sessions/{fresh}/ask", json={"question": "What is deja vu?"}).json()
        self.assertTrue(result["abstained"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["message"], config.NO_DOCUMENTS_MESSAGE)
        self.assertEqual(self.llm.calls, calls_before)

    def test_answers_only_cite_this_chats_documents(self):
        first, second = self.new_session(), self.new_session()
        self.upload(first, PAPERS[:1])
        self.upload(second, PAPERS[1:2])
        result = self.client.post(f"/sessions/{second}/ask", json={"question": "What do the authors report?"}).json()
        self.assertEqual({s["doc_id"] for s in result["sources"]}, {PAPERS[1].name})

    def test_ended_chat_is_forgotten(self):
        session = self.new_session()
        self.upload(session, PAPERS[:1])
        self.assertTrue(self.client.delete(f"/sessions/{session}").json()["deleted"])
        self.assertEqual(self.client.get(f"/sessions/{session}/documents").status_code, 404)
        self.assertEqual(self.client.post(f"/sessions/{session}/ask", json={"question": "Anything?"}).status_code, 404)

    def test_bad_files_fail_individually(self):
        session = self.new_session()
        files = [("files", ("notes.txt", b"hello", "text/plain")),
                 ("files", ("fake.pdf", b"not a pdf at all", "application/pdf")),
                 ("files", (PAPERS[0].name, PAPERS[0].read_bytes(), "application/pdf"))]
        body = self.client.post(f"/sessions/{session}/ingest", files=files).json()
        self.assertEqual([r["status"] for r in body["results"]], ["failed", "failed", "ingested"])

    def test_validation_errors_are_friendly_sentences(self):
        session = self.new_session()
        short = self.client.post(f"/sessions/{session}/ask", json={"question": "a"})
        self.assertEqual(short.status_code, 422)
        self.assertIsInstance(short.json()["detail"], str)
        self.assertIn("too short", short.json()["detail"])
        missing = self.client.post(f"/sessions/{session}/ask", json={})
        self.assertIn("'question' is required", missing.json()["detail"])
        broken = self.client.post(f"/sessions/{session}/ask", content=b"{oops",
                                  headers={"content-type": "application/json"})
        self.assertIn("not valid JSON", broken.json()["detail"])

    def test_refine_suggests_a_sharper_question(self):
        session = self.new_session()
        response = self.client.post(f"/sessions/{session}/refine", json={"text": "brain deja vu", "mode": "ask"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["original"], "brain deja vu")
        self.assertTrue(body["refined"].endswith("?"))
        self.assertTrue(body["changed"])
        bad = self.client.post(f"/sessions/{session}/refine", json={"text": "brain deja vu", "mode": "poem"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Unknown mode", bad.json()["detail"])

    def test_upload_with_no_files_is_rejected(self):
        session = self.new_session()
        self.assertEqual(self.client.post(f"/sessions/{session}/ingest").status_code, 400)


class SessionExpiryTests(unittest.TestCase):
    def test_idle_sessions_expire_and_oldest_is_evicted(self):
        from app.core.errors import SessionNotFoundError
        from app.pipelines.sessions import SessionManager

        now = [0.0]
        manager = SessionManager(embedder=HashEmbedder(), reranker=NoReranker(), llm=EchoLLM(),
                                 ttl_s=60, max_sessions=2, clock=lambda: now[0])
        manager._chunker = object()  # avoid importing the splitter in this unit test
        a = manager.create().session_id
        b = manager.create().session_id
        c = manager.create().session_id  # evicts a (least recently used)
        with self.assertRaises(SessionNotFoundError):
            manager.get(a)
        now[0] = 61.0
        with self.assertRaises(SessionNotFoundError):
            manager.get(b)
        self.assertEqual(manager.count(), 0)
        self.assertNotEqual(b, c)


if __name__ == "__main__":
    unittest.main()
