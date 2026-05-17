"""RestoAI Phase-3+6 FastAPI service.

Separate from the existing Flask app (app.py, port 5000). This service
exposes the three champion components as async JSON endpoints on port 8000:

    POST /sentiment   ->  NLI zero-shot (distilbart-mnli-12-3); VADER fallback
    POST /complaints  ->  TF-IDF + LightGBM OvR (blended with keyword); keyword fallback
    POST /rag         ->  FAISS top-15 -> ms-marco rerank -> flan-t5-base; template fallback

Day-7 (Phase 6) additions:
    - Redis cache for /rag responses (in-memory LRU fallback when REDIS_URL
      is absent or the server is unreachable). Cache backend reported in
      /health/cache.
    - Per-request RAGAS-proxy logging: every /rag response computes
      faithfulness / relevancy / context_precision / context_recall (Day-3
      deterministic proxy) and appends to logs/ragas_proxy.jsonl. The
      Streamlit dashboard (`app_dashboard.py`) tails the file for live
      quality plots.
    - /metrics/ragas returns a rolling summary (last N requests) for the
      dashboard. /metrics/cache returns cache stats.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Each request is Pydantic-validated. Model weights are lazy-loaded on first
call so boot time stays under 2 seconds; first request to each endpoint
takes ~10-20 s while flan-t5-base / cross-encoder / distilbart-mnli load.

The Flask app keeps running on its own port and is untouched. The shims in
manager_system/{analyzer,rag_chat}.py delegate to the same src.* modules
this service uses, so behaviour is consistent across both surfaces.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Make `src/` importable when running `uvicorn api:app` from the repo root.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.sentiment.classifier import get_default as _get_sent
from src.complaints.classifier import get_default as _get_comp
from src.rag.pipeline import get_default as _get_rag
from src.cache import get_cache, make_key  # noqa: WPS433 — exported by package init
from src.observability import get_logger as _get_ragas_logger

logger = logging.getLogger("restoai.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


# ----- request models -------------------------------------------------------
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw review text.")


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1)
    restaurant: Optional[str] = Field(None, description="Restaurant name filter; uses consolidated index if omitted.")
    top_k: int = Field(15, ge=1, le=50,
                       description="FAISS candidates retrieved before rerank. Day-3 champion uses 15.")
    rerank_k: int = Field(5, ge=1, le=20, description="Top-K kept after cross-encoder rerank.")
    use_cache: bool = Field(True, description="If False, bypass the cache for this request (write-through still applies).")


# ----- response models ------------------------------------------------------
class SentimentResponse(BaseModel):
    label: str
    compound: float
    keywords: List[str]
    model: str
    fallback_used: bool
    latency_ms: float


class ComplaintResponse(BaseModel):
    categories: List[str]
    scores: Optional[dict] = None
    model: str
    fallback_used: bool
    latency_ms: float


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    intent: Optional[str]
    model: str
    reranked: bool
    fallback_used: bool
    retrieved_count: int
    latency_ms: float
    cache_hit: bool = False
    ragas_proxy: Optional[Dict[str, float]] = None


# ----- app ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    cache = get_cache()
    logger.info("RestoAI Phase-6 API starting (cache=%s, lazy model load)", cache.backend)
    yield
    logger.info("RestoAI Phase-6 API shutdown")


app = FastAPI(
    title="RestoAI Phase-6 API",
    description="Champion sentiment + complaint + RAG synthesis with cache + RAGAS-proxy logging.",
    version="0.6.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    cache = get_cache()
    return {
        "status": "ok",
        "service": "restoai-api",
        "version": app.version,
        "cache_backend": cache.backend,
    }


@app.get("/health/cache")
async def health_cache():
    return get_cache().stats()


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(req: TextRequest):
    t0 = time.perf_counter()
    try:
        pred = _get_sent().predict(req.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("sentiment failed")
        raise HTTPException(status_code=500, detail=f"sentiment failed: {exc}") from exc
    dt = (time.perf_counter() - t0) * 1000
    return SentimentResponse(
        label=pred.label, compound=pred.compound, keywords=pred.keywords,
        model=pred.model, fallback_used=pred.fallback_used, latency_ms=round(dt, 2),
    )


@app.post("/complaints", response_model=ComplaintResponse)
async def complaints(req: TextRequest):
    t0 = time.perf_counter()
    try:
        pred = _get_comp().predict(req.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("complaints failed")
        raise HTTPException(status_code=500, detail=f"complaints failed: {exc}") from exc
    dt = (time.perf_counter() - t0) * 1000
    return ComplaintResponse(
        categories=pred.categories, scores=pred.scores, model=pred.model,
        fallback_used=pred.fallback_used, latency_ms=round(dt, 2),
    )


@app.post("/rag", response_model=RAGResponse)
async def rag(req: RAGRequest):
    t0 = time.perf_counter()
    cache = get_cache()
    key = make_key(req.query, req.restaurant, req.top_k, req.rerank_k)

    cache_hit = False
    payload: Optional[Dict[str, Any]] = None
    if req.use_cache:
        payload = cache.get(key)
        if payload is not None:
            cache_hit = True

    if payload is None:
        # cold path — run retrieval + synthesis
        try:
            rag_chat = _ensure_rag_chat()
            if not rag_chat.loaded:
                rag_chat._load_vector_db()  # noqa: SLF001 — same path as answer_query
                rag_chat.loaded = True

            retrieved_docs, _scores = rag_chat.semantic_search(
                req.query, top_k=req.top_k, restaurant_filter=req.restaurant,
            )
            synth = _get_rag()
            synth.rerank_k = req.rerank_k
            out = synth.synthesize(
                query=req.query, retrieved_docs=retrieved_docs, restaurant=req.restaurant,
            )
            sources: List[str] = []
            seen = set()
            for d in retrieved_docs:
                t = d["text"].strip()
                if t and t not in seen:
                    seen.add(t)
                    sources.append(t)
            payload = {
                "answer": out.answer,
                "sources": sources,
                "intent": out.intent,
                "model": out.model,
                "reranked": out.reranked,
                "fallback_used": out.fallback_used,
                "retrieved_count": out.retrieved_count,
            }
            # Don't cache failures or empty retrievals — they're transient.
            if out.retrieved_count > 0 and not out.fallback_used:
                cache.set(key, payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("rag failed")
            raise HTTPException(status_code=500, detail=f"rag failed: {exc}") from exc

    dt = (time.perf_counter() - t0) * 1000

    # Per-request RAGAS-proxy logging — always runs, even on cache hits.
    try:
        ragas = _get_ragas_logger().log(
            query=req.query,
            restaurant=req.restaurant,
            answer=payload["answer"],
            retrieved_texts=payload.get("sources", []),
            model=payload.get("model", "unknown"),
            reranked=payload.get("reranked", False),
            fallback_used=payload.get("fallback_used", False),
            cache_hit=cache_hit,
            latency_ms=dt,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAGAS-proxy logging failed: %s", exc)
        ragas = None

    return RAGResponse(
        answer=payload["answer"],
        sources=payload.get("sources", []),
        intent=payload.get("intent"),
        model=payload.get("model", "unknown"),
        reranked=payload.get("reranked", False),
        fallback_used=payload.get("fallback_used", False),
        retrieved_count=payload.get("retrieved_count", 0),
        latency_ms=round(dt, 2),
        cache_hit=cache_hit,
        ragas_proxy=ragas,
    )


@app.get("/metrics/ragas")
async def metrics_ragas(limit: int = Query(200, ge=1, le=10000)):
    """Tail the RAGAS-proxy log and return rolling summary stats."""
    path = _get_ragas_logger().path
    rows: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                tail = fh.readlines()[-limit:]
            for line in tail:
                try:
                    rows.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
        except OSError as exc:
            logger.warning("metrics_ragas read failed: %s", exc)

    if not rows:
        return {"count": 0, "summary": None, "recent": []}

    def _mean(field: str) -> float:
        vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "count": len(rows),
        "faithfulness_mean": _mean("faithfulness"),
        "relevancy_mean": _mean("relevancy"),
        "context_precision_mean": _mean("context_precision"),
        "context_recall_mean": _mean("context_recall"),
        "composite_mean": _mean("composite"),
        "latency_ms_mean": _mean("latency_ms"),
        "cache_hit_rate": round(
            sum(1 for r in rows if r.get("cache_hit")) / len(rows), 4
        ),
        "fallback_rate": round(
            sum(1 for r in rows if r.get("fallback_used")) / len(rows), 4
        ),
    }
    return {"count": len(rows), "summary": summary, "recent": rows[-25:]}


_RAG_CHAT = None


def _ensure_rag_chat():
    global _RAG_CHAT
    if _RAG_CHAT is None:
        sys.path.insert(0, os.path.join(_ROOT, "manager_system"))
        from rag_chat import RAGChat  # noqa: WPS433
        _RAG_CHAT = RAGChat()
    return _RAG_CHAT


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
