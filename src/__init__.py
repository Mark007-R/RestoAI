"""RestoAI Phase-3 production modules.

Champion components from Day-2/3 head-to-head, integrated for serving via
FastAPI (`api.py`) and consumed by the existing Flask app via lightweight
shims in `manager_system/analyzer.py` and `manager_system/rag_chat.py`.

  - src.sentiment.classifier  -> NLI zero-shot (distilbart-mnli-12-3); VADER fallback
  - src.complaints.classifier -> TF-IDF (word 1-2 + char 3-5) + LightGBM OvR; keyword fallback
  - src.rag.pipeline          -> FAISS top-15 -> cross-encoder ms-marco rerank -> flan-t5-base; template fallback
"""
