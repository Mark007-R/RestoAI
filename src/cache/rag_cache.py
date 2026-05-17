"""Two-backend cache for RAG responses.

Why this exists
---------------
The Day-3 champion flan-t5-base + ms-marco rerank path measured at ~2.4 s
per query on CPU. Many RAG calls in the manager dashboard / demo repeat
the same `(query, restaurant, top_k, rerank_k)` shape (e.g. "what do
people say about service" is the dashboard's default opening question).
Caching the synthesized JSON response collapses the second-and-later
latency to <10 ms.

Design
------
- `RAGCache.get(key) / set(key, value, ttl)` — JSON-only payloads.
- Two implementations behind the same interface:
    1. `_RedisCache`: production. Activated when `REDIS_URL` env var is
       set AND `redis.Redis().ping()` succeeds at construction time.
    2. `_InMemoryCache`: fallback. Thread-safe dict with a bounded
       capacity (`max_entries`, default 512) and TTL eviction on read.
- `make_key(query, restaurant, top_k, rerank_k)` produces a stable,
  collision-resistant SHA-1 key. Whitespace and case are normalized in
  the query string so "Best food?" and " best  FOOD?" hit the same
  slot — that's the dashboard refresh / multiple-user pattern.

The cache is wired into `api.py` behind a single decorator point. When
the cache is unavailable for any reason, the endpoint computes fresh
and reports `cache_hit=False` in the response — the API never returns
500 because the cache is down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = int(os.environ.get("RAG_CACHE_TTL_SEC", "3600"))
DEFAULT_MAX_ENTRIES = int(os.environ.get("RAG_CACHE_MAX_ENTRIES", "512"))


_WS = re.compile(r"\s+")


def make_key(query: str, restaurant: Optional[str], top_k: int, rerank_k: int) -> str:
    """Stable cache key. Normalises whitespace + case in the query."""
    q = _WS.sub(" ", (query or "").strip().lower())
    r = (restaurant or "").strip().lower()
    payload = f"{q}|{r}|{top_k}|{rerank_k}".encode("utf-8")
    return "rag:" + hashlib.sha1(payload).hexdigest()


class RAGCache:
    """Thin wrapper exposing get/set + a `backend` label for /health."""

    def __init__(self, impl, backend: str):
        self._impl = impl
        self.backend = backend  # "redis" | "memory" | "disabled"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._impl.get(key)

    def set(self, key: str, value: Dict[str, Any], ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._impl.set(key, value, ttl)

    def clear(self) -> None:
        self._impl.clear()

    def stats(self) -> Dict[str, Any]:
        return self._impl.stats()


# ---------------------------------------------------------------------------
# In-memory LRU + TTL
# ---------------------------------------------------------------------------
class _InMemoryCache:
    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._max = max(1, int(max_entries))
        self._store: "OrderedDict[str, Tuple[Dict[str, Any], float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            slot = self._store.get(key)
            if slot is None:
                self._misses += 1
                return None
            value, expires_at = slot
            if expires_at and time.time() >= expires_at:
                self._store.pop(key, None)
                self._misses += 1
                return None
            # move to most-recently-used end
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        with self._lock:
            expires_at = time.time() + ttl if ttl > 0 else 0.0
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": "memory",
                "size": len(self._store),
                "max_entries": self._max,
                "hits": self._hits,
                "misses": self._misses,
            }


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
class _RedisCache:
    def __init__(self, url: str):
        import redis  # noqa: WPS433 — only imported when redis is enabled
        self._client = redis.Redis.from_url(url, socket_connect_timeout=2,
                                            socket_timeout=2, decode_responses=True)
        # Force a ping so init fails fast if the server isn't there — falls
        # back to in-memory at construction time.
        self._client.ping()
        self._url = url
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._client.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis get failed for %s: %s", key, exc)
            return None
        if raw is None:
            self._misses += 1
            return None
        try:
            self._hits += 1
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError):
            return
        try:
            if ttl > 0:
                self._client.setex(key, ttl, payload)
            else:
                self._client.set(key, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis set failed for %s: %s", key, exc)

    def clear(self) -> None:
        # Only clear the keys we own to avoid wiping a shared Redis.
        try:
            for k in self._client.scan_iter(match="rag:*", count=200):
                self._client.delete(k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis clear failed: %s", exc)

    def stats(self) -> Dict[str, Any]:
        try:
            n = sum(1 for _ in self._client.scan_iter(match="rag:*", count=200))
        except Exception:  # noqa: BLE001
            n = -1
        return {
            "backend": "redis",
            "url": self._url,
            "size": n,
            "hits": self._hits,
            "misses": self._misses,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_DEFAULT: Optional[RAGCache] = None


def get_cache(force: bool = False) -> RAGCache:
    """Return the singleton cache, constructing it on first call.

    Redis is used when REDIS_URL is set AND the client can ping. Otherwise
    we fall back to an in-memory LRU. `force=True` rebuilds (test hook).
    """
    global _DEFAULT
    if _DEFAULT is not None and not force:
        return _DEFAULT

    url = os.environ.get("REDIS_URL")
    if url:
        try:
            impl = _RedisCache(url)
            _DEFAULT = RAGCache(impl, "redis")
            logger.info("RAG cache backend: redis at %s", url)
            return _DEFAULT
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable (%s); using in-memory cache", exc)

    impl = _InMemoryCache(max_entries=DEFAULT_MAX_ENTRIES)
    _DEFAULT = RAGCache(impl, "memory")
    logger.info("RAG cache backend: memory (max_entries=%d)", DEFAULT_MAX_ENTRIES)
    return _DEFAULT


def reset_default_cache() -> None:
    """Test hook — drop the cached singleton so the next `get_cache()` rebuilds."""
    global _DEFAULT
    _DEFAULT = None
