"""FastAPI service e2e tests.

The RAG endpoint depends on the FAISS index built by the Flask app's
RAGChat — too heavy for unit tests. We monkeypatch `_ensure_rag_chat`
and the synthesizer so the endpoint exercises the cache + RAGAS-proxy
pipeline without loading any models.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import api as api_mod
from src.cache import reset_default_cache
from src.rag.pipeline import RAGAnswer


@pytest.fixture
def client(monkeypatch, tmp_ragas_log):
    """Patch heavy dependencies and yield a fresh TestClient."""
    reset_default_cache()
    # --- stub the RAG retrieval + synthesis -----------------------------
    class _StubRAGChat:
        loaded = True

        def _load_vector_db(self):  # noqa: D401
            self.loaded = True

        def semantic_search(self, query, top_k=15, restaurant_filter=None):
            docs = [
                {"text": f"Sample review {i} mentioning food and service.",
                 "metadata": {"rating": 4 + (i % 2)}}
                for i in range(min(top_k, 5))
            ]
            scores = [0.9 - 0.1 * i for i in range(len(docs))]
            return docs, scores

    monkeypatch.setattr(api_mod, "_ensure_rag_chat", lambda: _StubRAGChat())

    class _StubSynth:
        rerank_k = 5

        def synthesize(self, query, retrieved_docs, restaurant=None, intent=None):
            return RAGAnswer(
                answer=f"Stub answer for: {query}",
                intent="quality",
                model="stub-llm",
                reranked=True,
                fallback_used=False,
                retrieved_count=len(retrieved_docs),
            )

    monkeypatch.setattr(api_mod, "_get_rag", lambda: _StubSynth())

    return TestClient(api_mod.app)


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "restoai-api"
    assert body["cache_backend"] == "memory"


def test_health_cache_endpoint(client):
    r = client.get("/health/cache")
    assert r.status_code == 200
    assert r.json()["backend"] == "memory"


def test_sentiment_endpoint_happy_path(client):
    # Force VADER backend to avoid loading NLI in tests.
    sent = api_mod._get_sent()
    sent._prefer_nli = False
    r = client.post("/sentiment", json={"text": "The food was great."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"Positive", "Neutral", "Negative"}
    assert "latency_ms" in body
    assert isinstance(body["keywords"], list)


def test_sentiment_validation_rejects_empty(client):
    r = client.post("/sentiment", json={"text": ""})
    assert r.status_code == 422  # min_length=1 violation


def test_sentiment_validation_rejects_missing_field(client):
    r = client.post("/sentiment", json={})
    assert r.status_code == 422


def test_complaints_endpoint_happy_path(client):
    r = client.post("/complaints", json={"text": "The waiter was rude and slow."})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["categories"], list)
    assert "service" in body["categories"]
    assert "latency_ms" in body


def test_complaints_validation_rejects_empty(client):
    r = client.post("/complaints", json={"text": ""})
    assert r.status_code == 422


def test_rag_endpoint_returns_schema(client):
    r = client.post("/rag", json={"query": "How is the food?"})
    assert r.status_code == 200
    body = r.json()
    for field in ("answer", "sources", "model", "reranked", "fallback_used",
                  "retrieved_count", "latency_ms", "cache_hit", "ragas_proxy"):
        assert field in body
    assert body["cache_hit"] is False  # cold path
    assert body["retrieved_count"] > 0


def test_rag_cache_hit_on_repeat(client):
    payload = {"query": "What about service?", "restaurant": "Acme",
               "top_k": 5, "rerank_k": 3}
    first = client.post("/rag", json=payload).json()
    assert first["cache_hit"] is False
    second = client.post("/rag", json=payload).json()
    assert second["cache_hit"] is True
    # Cached payload should be byte-identical (modulo cache_hit + latency).
    assert second["answer"] == first["answer"]
    assert second["model"] == first["model"]


def test_rag_use_cache_false_bypasses_cache(client):
    payload = {"query": "Is it expensive?", "use_cache": True}
    client.post("/rag", json=payload).json()
    bypassed = client.post("/rag", json={**payload, "use_cache": False}).json()
    assert bypassed["cache_hit"] is False


def test_rag_validation_rejects_oversize_top_k(client):
    r = client.post("/rag", json={"query": "q", "top_k": 999})
    assert r.status_code == 422


def test_rag_validation_rejects_zero_top_k(client):
    r = client.post("/rag", json={"query": "q", "top_k": 0})
    assert r.status_code == 422


def test_rag_logs_ragas_proxy(client, tmp_ragas_log):
    client.post("/rag", json={"query": "How is the ambience?"})
    assert tmp_ragas_log.exists()
    content = tmp_ragas_log.read_text().strip()
    assert content, "ragas log should have at least one line"


def test_metrics_ragas_summary_after_request(client):
    client.post("/rag", json={"query": "How is the service?"})
    r = client.get("/metrics/ragas")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert body["summary"] is not None
    assert "composite_mean" in body["summary"]
    assert "cache_hit_rate" in body["summary"]


def test_metrics_ragas_when_empty(client, tmp_ragas_log):
    # Don't fire any RAG requests; tmp_ragas_log is empty.
    r = client.get("/metrics/ragas")
    body = r.json()
    assert body["count"] == 0
    assert body["summary"] is None


def test_openapi_schema_includes_all_endpoints(client):
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    for p in ("/health", "/health/cache", "/sentiment", "/complaints",
              "/rag", "/metrics/ragas"):
        assert p in paths
