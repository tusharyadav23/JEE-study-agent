"""Offline tests: use a deterministic hashed bag-of-words embedding instead of downloading
the real Hugging Face model, so the pipeline logic can be verified anywhere (CI included)."""
import hashlib
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

import app.ingest as ingest
import app.main as main
import app.rag as rag
from app import config


class FakeEmbeddings(Embeddings):
    DIM = 512

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype="float32")
        for tok in re.split(r"[\s,.:;?!()/=]+", text.lower()):
            if tok:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n * 5).tolist() if n else v.tolist()  # scale so L2 distances are ~0-100

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    store = tmp_path_factory.mktemp("vs")
    ingest.VECTORSTORE_DIR = store
    rag.VECTORSTORE_DIR = store
    return rag.JEEAgent(embeddings=FakeEmbeddings())


def test_index_built_and_loaded(agent):
    assert agent.vectorstore.index.ntotal > 10


def test_retrieves_relevant_chunk(agent):
    docs = agent.retrieve("What is the marking scheme of JEE Main negative marking", k=4)
    assert docs and any("negative" in d.page_content.lower() for d in docs)


def test_weak_matches_are_filtered(agent):
    assert agent.retrieve_with_scores("zzz qqq xxx unrelated gibberish", threshold=0.0) == []


def test_unknown_question_gets_refusal_not_hallucination(agent, monkeypatch):
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 0.0)
    out = agent.answer("Who won the FIFA World Cup?")
    assert out["grounded"] is False and out["sources"] == []


def test_hindi_detection_and_hindi_refusal(agent, monkeypatch):
    assert rag.is_hindi("JEE Main का पैटर्न क्या है?") and not rag.is_hindi("What is JEE?")
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 0.0)
    assert agent.answer("आज मौसम कैसा है?")["answer"] == rag.NO_ANSWER_HI


def test_extractive_answer_without_llm(agent, monkeypatch):
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 1000.0)
    agent.llm = None
    out = agent.answer("What is the ideal gas equation PV = nRT?")
    assert out["grounded"] and out["sources"] and out["answer"]


def test_api_endpoints(agent, monkeypatch):
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 1000.0)
    monkeypatch.setattr(main, "JEEAgent", lambda: agent)
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        r = client.post("/ask", json={"question": "Who conducts JEE Main?"})
        assert r.status_code == 200 and "answer" in r.json()
        assert client.post("/ask", json={"question": "x"}).status_code == 422
