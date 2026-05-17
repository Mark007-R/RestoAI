"""RAG cache tests.

Tests run against the in-memory backend (no Redis required). Key
normalisation, TTL expiry, LRU eviction, and graceful fallback when
REDIS_URL points at an unreachable server are all covered.
"""

from __future__ import annotations

import os
import time

import pytest

from src.cache import get_cache, make_key, reset_default_cache
from src.cache.rag_cache import _InMemoryCache, RAGCache


def test_make_key_is_deterministic():
    a = make_key("How is the food?", "Acme", 15, 5)
    b = make_key("How is the food?", "Acme", 15, 5)
    assert a == b


def test_make_key_normalises_case_and_whitespace():
    a = make_key("Best food?", "Acme", 15, 5)
    b = make_key("  best   FOOD? ", "ACME", 15, 5)
    assert a == b


def test_make_key_distinguishes_different_inputs():
    base = make_key("q", "r", 15, 5)
    assert make_key("q2", "r", 15, 5) != base
    assert make_key("q", "r2", 15, 5) != base
    assert make_key("q", "r", 20, 5) != base
    assert make_key("q", "r", 15, 10) != base


def test_make_key_handles_no_restaurant():
    a = make_key("q", None, 15, 5)
    b = make_key("q", "", 15, 5)
    assert a == b  # empty/None normalise to the same slot


def test_make_key_starts_with_prefix():
    assert make_key("q", "r", 15, 5).startswith("rag:")


def test_get_returns_none_on_miss():
    c = get_cache()
    assert c.get("rag:nope") is None


def test_set_then_get_roundtrip():
    c = get_cache()
    k = make_key("test", None, 5, 3)
    c.set(k, {"answer": "hello", "sources": ["s1"]})
    out = c.get(k)
    assert out == {"answer": "hello", "sources": ["s1"]}


def test_stats_track_hits_and_misses():
    c = get_cache()
    k = make_key("q", None, 5, 3)
    c.set(k, {"x": 1})
    c.get(k)  # hit
    c.get(k)  # hit
    c.get("rag:other")  # miss
    s = c.stats()
    assert s["hits"] >= 2
    assert s["misses"] >= 1


def test_clear_removes_entries():
    c = get_cache()
    k = make_key("clearme", None, 1, 1)
    c.set(k, {"x": 1})
    assert c.get(k) is not None
    c.clear()
    assert c.get(k) is None


def test_in_memory_lru_evicts_oldest():
    cache = RAGCache(_InMemoryCache(max_entries=3), "memory")
    for i in range(5):
        cache.set(f"rag:{i}", {"v": i})
    # The first two inserts should be evicted; last three remain.
    assert cache.get("rag:0") is None
    assert cache.get("rag:1") is None
    assert cache.get("rag:4") == {"v": 4}
    assert cache.stats()["size"] == 3


def test_in_memory_lru_promotes_on_read():
    cache = RAGCache(_InMemoryCache(max_entries=3), "memory")
    for i in range(3):
        cache.set(f"rag:{i}", {"v": i})
    # Touch rag:0 -> it becomes most-recently-used; rag:1 is now the oldest.
    assert cache.get("rag:0") == {"v": 0}
    cache.set("rag:3", {"v": 3})
    assert cache.get("rag:0") == {"v": 0}  # survived
    assert cache.get("rag:1") is None      # evicted


def test_ttl_expiry():
    cache = RAGCache(_InMemoryCache(max_entries=10), "memory")
    cache.set("rag:short", {"v": 1}, ttl=1)
    assert cache.get("rag:short") is not None
    time.sleep(1.1)
    assert cache.get("rag:short") is None


def test_default_backend_is_memory_without_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_default_cache()
    c = get_cache()
    assert c.backend == "memory"


def test_falls_back_to_memory_when_redis_unreachable(monkeypatch):
    """When REDIS_URL points somewhere unreachable, get_cache() must
    fall back to memory rather than raising or returning None."""
    # 1: invalid port that nothing listens on
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    reset_default_cache()
    c = get_cache()
    assert c.backend == "memory"


def test_stats_returns_backend_label():
    c = get_cache()
    assert c.stats()["backend"] == "memory"
