"""RestoAI Phase-3 FastAPI service.

Separate from the existing Flask app (app.py, port 5000). This service exposes
the three champion components as async JSON endpoints on port 8000:

    POST /sentiment   ->  NLI zero-shot (distilbart-mnli-12-3); VADER fallback
    POST /complaints  ->  TF-IDF + LightGBM OvR (blended with keyword); keyword fallback
    POST /rag         ->  FAISS top-15 -> ms-marco rerank -> flan-t5-base; template fallback

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Each request is Pydantic-validated. Model weights are lazy-loaded on first call
so boot time stays under 2 seconds; first request to each endpoint takes
~10-20 s while flan-t5-base / cross-encoder / distilbart-mnli load.

The Flask app keeps running on its own port and is untouched. The shims in
manager_system/{analyzer,rag_chat}.py delegate to the same src.* modules this
service uses, so behaviour is consistent across both surfaces.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make `src/` importable when running `uvicorn api:app` from the repo root.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.sentiment.classifier import get_default as _get_sent
from src.complaints.classifier import get_default as _get_comp
from src.rag.pipeline import get_default as _get_rag

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


# ----- app ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("RestoAI Phase-3 API starting (model weights load lazily on first call)")
    yield
    logger.info("RestoAI Phase-3 API shutdown")


app = FastAPI(
    title="RestoAI Phase-3 API",
    description="Champion sentiment + complaint + RAG synthesis on top of the existing Flask app.",
    version="0.4.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "restoai-api", "version": app.version}


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
    # The Phase-3 RAG path is: existing FAISS retrieve top-K -> ms-marco rerank
    # to top rerank_k -> flan-t5-base synthesis. We pull retrieval from RAGChat
    # so this service uses the same vector store as the Flask app.
    try:
        sys.path.insert(0, os.path.join(_ROOT, "manager_system"))
        from rag_chat import RAGChat  # noqa: WPS433
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
            query=req.query,
            retrieved_docs=retrieved_docs,
            restaurant=req.restaurant,
        )
        sources = []
        seen = set()
        for d in retrieved_docs:
            t = d["text"].strip()
            if t and t not in seen:
                seen.add(t)
                sources.append(t)
        dt = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            answer=out.answer, sources=sources, intent=out.intent,
            model=out.model, reranked=out.reranked, fallback_used=out.fallback_used,
            retrieved_count=out.retrieved_count, latency_ms=round(dt, 2),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rag failed")
        raise HTTPException(status_code=500, detail=f"rag failed: {exc}") from exc


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
