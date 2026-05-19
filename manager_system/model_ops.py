"""Read-only model-ops helpers for the Flask manager dashboard.

These functions back the ``/manager/model-ops`` route. They replace the
Phase-6 Streamlit dashboard (``app_dashboard.py``, removed) by surfacing
the same three views directly in Flask:

  1. The locked Day-6 champion model card (static table the Streamlit
     dashboard rendered from markdown).
  2. Live RAGAS-proxy quality — rolling mean of the four scores plus
     cache-hit rate, fallback rate, and latency percentiles. Reads the
     same ``logs/ragas_proxy.jsonl`` the FastAPI service writes to.
  3. Live RAG cache state — backend, entry count, hit/miss/eviction
     counters, sampled keys. Reads through the in-process
     ``src.cache.rag_cache`` singleton.

Everything here is pure read-only and tolerant of missing files / missing
optional deps — the route still renders sensibly on a fresh checkout
where no ``/rag`` request has ever been served.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = BASE_DIR / "logs" / "ragas_proxy.jsonl"


CHAMPION_MODEL_CARD: list[dict[str, str]] = [
    {
        "component": "Sentiment",
        "champion": "valhalla/distilbart-mnli-12-3 (NLI zero-shot)",
        "fallback": "VADER",
        "eval_set": "200-review (Day-2)",
        "macro_f1": "0.701",
        "latency_ms": "522",
        "notes": "Lifts Neutral-class F1 0.081 -> 0.478 (6x). Hard Rule 5 signature preserved.",
    },
    {
        "component": "Sentiment (production trade-off)",
        "champion": "distilbert-base-uncased-finetuned-sst-2-english",
        "fallback": "VADER",
        "eval_set": "100 fresh held-out (Day-6)",
        "macro_f1": "0.560",
        "latency_ms": "40",
        "notes": "13x faster than NLI; ships when latency budget < 100ms.",
    },
    {
        "component": "Complaints",
        "champion": "TF-IDF + LightGBM one-vs-rest (Optuna t=0.5)",
        "fallback": "8-keyword CATEGORY_KEYWORDS scan",
        "eval_set": "100 fresh held-out (Day-6)",
        "macro_f1": "0.853",
        "latency_ms": "34",
        "notes": "Per-category gains: delivery 0.537->0.923, portion 0.808->0.927, food_quality 0.788->0.865.",
    },
    {
        "component": "RAG retriever",
        "champion": "all-MiniLM-L6-v2 (384-dim) + FAISS",
        "fallback": "n/a",
        "eval_set": "-",
        "macro_f1": "-",
        "latency_ms": "-",
        "notes": "Consolidated index across all restaurants.",
    },
    {
        "component": "RAG reranker",
        "champion": "cross-encoder/ms-marco-MiniLM-L-6-v2 (top-15 -> top-5)",
        "fallback": "None",
        "eval_set": "-",
        "macro_f1": "-",
        "latency_ms": "-",
        "notes": "Improves context recall +0.105 vs template baseline.",
    },
    {
        "component": "RAG synthesis",
        "champion": "google/flan-t5-base",
        "fallback": "Templated rule-based synthesis",
        "eval_set": "50 QA (Day-3)",
        "macro_f1": "0.663 composite",
        "latency_ms": "2400",
        "notes": "Per-restaurant prose vs templated sentences.",
    },
    {
        "component": "RAG cache",
        "champion": "Redis (when REDIS_URL is set)",
        "fallback": "In-memory LRU + TTL",
        "eval_set": "-",
        "macro_f1": "-",
        "latency_ms": "< 10",
        "notes": "240x speedup on warm hit (2.4s -> <10ms).",
    },
]


def resolve_log_path() -> Path:
    """Resolve the RAGAS proxy log location (env override wins)."""
    env_path = os.environ.get("RAGAS_LOG_PATH")
    return Path(env_path) if env_path else DEFAULT_LOG_PATH


def load_ragas_records(limit: int = 500, *, path: Path | None = None) -> list[dict[str, Any]]:
    """Read up to ``limit`` most recent records from the RAGAS proxy log.

    Returns an empty list if the file is missing or unreadable — the page
    must render cleanly on a fresh checkout.
    """
    log_path = path or resolve_log_path()
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh.readlines()[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    return rows


def summarise_ragas(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the rolling means + percentiles the dashboard surfaces."""
    if not records:
        return {
            "request_count": 0,
            "has_data": False,
            "faithfulness": None,
            "relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "composite": None,
            "cache_hit_rate": None,
            "fallback_rate": None,
            "latency_p50_ms": None,
            "latency_p90_ms": None,
            "latency_p99_ms": None,
            "recent": [],
        }

    def _mean(field: str) -> float | None:
        values = [float(r[field]) for r in records if isinstance(r.get(field), (int, float))]
        return round(sum(values) / len(values), 4) if values else None

    def _rate(field: str) -> float | None:
        flags = [1 for r in records if r.get(field)]
        return round(len(flags) / len(records), 4) if records else None

    latencies = sorted(
        [float(r["latency_ms"]) for r in records if isinstance(r.get("latency_ms"), (int, float))]
    )

    def _quantile(q: float) -> float | None:
        if not latencies:
            return None
        if len(latencies) == 1:
            return round(latencies[0], 2)
        idx = max(0, min(len(latencies) - 1, int(round(q * (len(latencies) - 1)))))
        return round(latencies[idx], 2)

    # Most recent first, capped at 25 rows for the table.
    recent = sorted(records, key=lambda r: r.get("ts", 0), reverse=True)[:25]
    for r in recent:
        ts = r.get("ts")
        if isinstance(ts, (int, float)):
            try:
                import datetime as _dt

                r["ts_iso"] = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (OSError, ValueError):
                r["ts_iso"] = str(ts)

    return {
        "request_count": len(records),
        "has_data": True,
        "faithfulness": _mean("faithfulness"),
        "relevancy": _mean("relevancy"),
        "context_precision": _mean("context_precision"),
        "context_recall": _mean("context_recall"),
        "composite": _mean("composite"),
        "cache_hit_rate": _rate("cache_hit"),
        "fallback_rate": _rate("fallback_used"),
        "latency_p50_ms": _quantile(0.50),
        "latency_p90_ms": _quantile(0.90),
        "latency_p99_ms": _quantile(0.99),
        "recent": recent,
    }


def get_cache_state() -> dict[str, Any]:
    """Snapshot the RAG cache backend + counters via the in-process singleton.

    Returns a safe stub if the cache module is not importable (e.g., the
    runtime is the slim Flask-only environment without the production
    extras installed).
    """
    try:
        from src.cache.rag_cache import get_cache  # type: ignore
    except ImportError:
        return {
            "available": False,
            "reason": "src.cache.rag_cache not importable in this environment",
        }
    try:
        cache = get_cache()
        stats = cache.stats() if hasattr(cache, "stats") else {}
        return {
            "available": True,
            "backend": stats.get("backend", "unknown"),
            "size": stats.get("size"),
            "max_entries": stats.get("max_entries"),
            "ttl_seconds": stats.get("ttl_seconds"),
            "hits": stats.get("hits"),
            "misses": stats.get("misses"),
            "evictions": stats.get("evictions"),
            "hit_rate": (
                round(stats.get("hits", 0) / max(1, stats.get("hits", 0) + stats.get("misses", 0)), 4)
                if stats.get("hits") is not None and stats.get("misses") is not None
                else None
            ),
            "stats_raw": stats,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"cache.stats() raised: {exc}"}


def build_model_ops_context() -> dict[str, Any]:
    """One-call context builder for the Jinja template."""
    log_path = resolve_log_path()
    records = load_ragas_records(limit=500, path=log_path)
    return {
        "champion_card": CHAMPION_MODEL_CARD,
        "ragas": summarise_ragas(records),
        "cache": get_cache_state(),
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
    }
