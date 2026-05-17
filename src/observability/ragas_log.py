"""Deterministic structural proxy for RAGAS — runs on every RAG request.

Background
----------
Day 3 picked the RAG champion (flan-t5-base + ms-marco rerank) using a
deterministic RAGAS proxy because no Anthropic / OpenAI key was
available in the scheduled-task subprocess. The same proxy is used here
for *runtime* quality logging: every `/rag` request gets four scores in
[0, 1] persisted to JSONL, so the manager dashboard can plot p50 / p90
faithfulness over time without an LLM judge call per request.

The four scores mirror RAGAS:
    faithfulness     — fraction of answer tokens that appear in any
                       retrieved doc (paraphrase proxy)
    relevancy        — Jaccard overlap between query tokens and answer
                       tokens (excluding stopwords)
    context_precision — fraction of retrieved docs whose tokens overlap
                       with the answer (signal-to-noise of retrieval)
    context_recall    — fraction of query tokens covered by the union
                       of retrieved-doc tokens (retrieval coverage)
The composite is the geometric mean of the four — matches Day-3.

Storage
-------
JSONL at `logs/ragas_proxy.jsonl` (configurable via `RAGAS_LOG_PATH`).
One line per request, fixed schema, append-only. The Streamlit dashboard
tails the last N lines to render the live distribution.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_FALLBACK_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "ragas_proxy.jsonl",
)


def _default_log_path() -> str:
    """Re-read the env var on every call so tests can monkeypatch RAGAS_LOG_PATH
    without having to rebuild the module."""
    return os.environ.get("RAGAS_LOG_PATH", _FALLBACK_LOG_PATH)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "with", "this",
    "that", "have", "has", "was", "were", "from", "they", "their", "them", "our",
    "very", "really", "just", "like", "also", "into", "your", "what", "when",
    "there", "here", "been", "being", "would", "could", "should", "about", "does",
    "did", "doing", "much", "many", "some", "more", "most", "than", "then", "too",
    "very", "will", "off", "out", "how", "why", "where", "which", "who", "whom",
    "its", "is", "in", "at", "of", "to", "a", "an", "or", "on", "by", "as", "be",
    "it", "i", "we", "us", "me", "my",
}


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _overlap_ratio(needle: List[str], haystack_tokens: List[str]) -> float:
    if not needle:
        return 0.0
    if not haystack_tokens:
        return 0.0
    h = set(haystack_tokens)
    hit = sum(1 for t in set(needle) if t in h)
    return hit / len(set(needle))


def score_request(
    query: str,
    answer: str,
    retrieved_texts: List[str],
) -> Dict[str, float]:
    """Compute the four RAGAS-proxy scores in [0, 1] + composite.

    Pure function, no I/O — safe to call from the request hot path.
    """
    q_tok = _tokens(query)
    a_tok = _tokens(answer)
    doc_tokens = [_tokens(t) for t in retrieved_texts]
    all_doc_tok = [t for ts in doc_tokens for t in ts]

    faithfulness = _overlap_ratio(a_tok, all_doc_tok)
    relevancy = _jaccard(q_tok, a_tok)

    if doc_tokens:
        ctx_prec_hits = 0
        for ts in doc_tokens:
            if not ts:
                continue
            # a doc counts as "precise" if at least 10% of its non-stopword tokens
            # appear in the answer — same threshold as Day-3 structural_rag_metrics.
            shared = set(ts) & set(a_tok) if a_tok else set()
            if len(shared) / max(1, len(set(ts))) >= 0.10:
                ctx_prec_hits += 1
        context_precision = ctx_prec_hits / len(doc_tokens)
    else:
        context_precision = 0.0

    context_recall = _overlap_ratio(q_tok, all_doc_tok)

    # geometric mean — matches Day-3 composite. Floor each term so a single
    # zero doesn't collapse the whole composite.
    floored = [max(1e-3, v) for v in (faithfulness, relevancy, context_precision, context_recall)]
    composite = math.exp(sum(math.log(v) for v in floored) / 4.0)

    return {
        "faithfulness": round(faithfulness, 4),
        "relevancy": round(relevancy, 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "composite": round(composite, 4),
    }


class RAGASProxyLogger:
    """Append-only JSONL logger. One line per request."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or _default_log_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        query: str,
        restaurant: Optional[str],
        answer: str,
        retrieved_texts: List[str],
        model: str,
        reranked: bool,
        fallback_used: bool,
        cache_hit: bool,
        latency_ms: float,
    ) -> Dict[str, float]:
        scores = score_request(query, answer, retrieved_texts)
        record = {
            "ts": time.time(),
            "query": query[:500],
            "restaurant": restaurant,
            "model": model,
            "reranked": bool(reranked),
            "fallback_used": bool(fallback_used),
            "cache_hit": bool(cache_hit),
            "retrieved_count": len(retrieved_texts),
            "latency_ms": round(latency_ms, 2),
            **scores,
        }
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("RAGAS-proxy log write failed: %s", exc)
        return scores


_DEFAULT: Optional[RAGASProxyLogger] = None


def get_logger(path: Optional[str] = None) -> RAGASProxyLogger:
    global _DEFAULT
    target = path or _default_log_path()
    if _DEFAULT is None or target != _DEFAULT.path:
        _DEFAULT = RAGASProxyLogger(target)
    return _DEFAULT
