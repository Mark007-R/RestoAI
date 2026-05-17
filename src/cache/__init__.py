"""Phase-6 caching layer for the RAG endpoint.

The flan-t5-base + cross-encoder rerank path runs ~2.4 s/sample on CPU
(Day-3 measurement). Repeated queries from the manager dashboard,
restaurant-comparison views, and the demo all hit the same `(query,
restaurant, top_k, rerank_k)` shape — so a small cache makes the
median latency disappear.

`get_cache()` returns a Redis-backed cache when `REDIS_URL` is set and
the connection succeeds; otherwise a thread-safe LRU in-memory cache so
local dev, CI, and the test suite work without a Redis container.
"""

from .rag_cache import RAGCache, get_cache, make_key, reset_default_cache  # noqa: F401
