"""Pydantic v2 request / response schemas shared by api.py and the src.* modules."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw review text.")


class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1)
    restaurant: Optional[str] = Field(None, description="Restaurant filter; uses consolidated index if omitted.")
    top_k: int = Field(15, ge=1, le=50, description="Number of FAISS candidates to retrieve before reranking.")
    rerank_k: int = Field(5, ge=1, le=20, description="Number of chunks kept after cross-encoder rerank.")


class SentimentResponse(BaseModel):
    label: str
    compound: float
    keywords: List[str]
    model: str
    fallback_used: bool = False


class ComplaintResponse(BaseModel):
    categories: List[str]
    scores: Optional[dict] = None  # category -> probability when trained model is loaded
    model: str
    fallback_used: bool = False


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    intent: Optional[str] = None
    model: str
    reranked: bool = False
    fallback_used: bool = False
