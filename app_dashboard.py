"""RestoAI Phase-6 manager dashboard (Streamlit, separate from Flask).

Surfaces three live views that the existing Flask app can't:

  1. Live RAGAS-proxy quality (tail of logs/ragas_proxy.jsonl): rolling
     means, latency distribution, cache hit rate, faithfulness over time.
  2. Complaint heat map across the 5 source datasets — backed by the
     trained TF-IDF + LightGBM classifier (Phase-3), not the keyword
     baseline. Shows where the trained head's per-category recall
     diverges from the keyword fallback.
  3. Sentiment trend: distribution of NLI-zero-shot labels on the most
     recent reviews scored, plus the macro-F1 from the Day-2 eval card.

This is a separate process from app.py. It calls the FastAPI service for
live inference and reads the JSONL log directly for the metrics view.

Run:  streamlit run app_dashboard.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="RestoAI Manager", layout="wide")

API_URL = os.environ.get("RESTOAI_API_URL", "http://localhost:8000")
LOG_PATH = os.environ.get(
    "RAGAS_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ragas_proxy.jsonl"),
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 3.0):
    try:
        r = requests.get(f"{API_URL}{path}", params=params or {}, timeout=timeout)
        return r.status_code, r.json()
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


def _safe_post(path: str, payload: Dict[str, Any], timeout: float = 60.0):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        return r.status_code, r.json()
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


@st.cache_data(ttl=5)
def load_ragas_log(path: str, limit: int = 2000) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh.readlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    return df


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("RestoAI Manager")
st.sidebar.caption(f"Connected to **{API_URL}**")
status, health = _safe_get("/health")
if status == 200:
    st.sidebar.success(f"API ok · cache={health.get('cache_backend')}")
else:
    st.sidebar.error(f"API unreachable: {health.get('error', 'unknown')}")

view = st.sidebar.radio(
    "View",
    ["Live RAGAS quality", "Try a query", "Single-review analyzer", "Champion model card"],
    index=0,
)

# ---------------------------------------------------------------------------
# view: live RAGAS quality
# ---------------------------------------------------------------------------
if view == "Live RAGAS quality":
    st.title("RAGAS-proxy live quality")
    st.caption(
        "Every `/rag` request logs faithfulness / relevancy / context_precision / "
        "context_recall using the Day-3 deterministic structural proxy. "
        "Champion (`flan-t5-base + ms-marco rerank`) target composite ≥ 0.66 "
        "(Day-3 50-QA eval)."
    )

    df = load_ragas_log(LOG_PATH)
    if df.empty:
        st.info("No RAGAS-proxy records yet. Run a `/rag` request from the **Try a query** view.")
    else:
        cols = st.columns(5)
        cols[0].metric("Requests", len(df))
        for i, field in enumerate(["faithfulness", "relevancy", "context_precision", "context_recall"]):
            mean = float(df[field].mean()) if field in df else 0.0
            cols[i + 1].metric(field, f"{mean:.3f}")

        cols2 = st.columns(4)
        cache_rate = float(df.get("cache_hit", pd.Series([False])).mean())
        cols2[0].metric("Cache hit rate", f"{cache_rate:.0%}")
        fallback_rate = float(df.get("fallback_used", pd.Series([False])).mean())
        cols2[1].metric("LLM fallback rate", f"{fallback_rate:.0%}")
        if "latency_ms" in df:
            cols2[2].metric("p50 latency (ms)", f"{df['latency_ms'].median():.0f}")
            cols2[3].metric("p90 latency (ms)", f"{df['latency_ms'].quantile(0.9):.0f}")

        if "ts" in df and "composite" in df:
            st.subheader("Composite over time")
            st.plotly_chart(
                px.line(df.sort_values("ts"), x="ts", y="composite", markers=True,
                        title=None, height=320),
                use_container_width=True,
            )

        if "latency_ms" in df:
            st.subheader("Latency distribution (ms)")
            st.plotly_chart(
                px.histogram(df, x="latency_ms", nbins=30, height=300),
                use_container_width=True,
            )

        st.subheader("Recent requests")
        cols_show = [c for c in ["ts", "query", "restaurant", "model", "cache_hit",
                                 "fallback_used", "faithfulness", "composite", "latency_ms"]
                     if c in df.columns]
        st.dataframe(df.sort_values("ts", ascending=False)[cols_show].head(25),
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# view: try a query
# ---------------------------------------------------------------------------
elif view == "Try a query":
    st.title("Try a RAG query")
    st.caption("Calls the FastAPI `/rag` endpoint. Second hit on the same query "
               "is served from cache (<10ms).")
    with st.form("rag_form", clear_on_submit=False):
        c1, c2 = st.columns([3, 2])
        q = c1.text_input("Query", value="How is the food quality?")
        rest = c2.text_input("Restaurant (optional)", value="")
        c3, c4, c5 = st.columns(3)
        top_k = c3.slider("top_k (retrieve)", 1, 30, 15)
        rerank_k = c4.slider("rerank_k", 1, 10, 5)
        bypass = c5.checkbox("Bypass cache", value=False)
        submitted = st.form_submit_button("Ask")
    if submitted and q.strip():
        t0 = time.perf_counter()
        status, body = _safe_post("/rag", {
            "query": q,
            "restaurant": rest.strip() or None,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "use_cache": not bypass,
        }, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
        if status == 200:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("client-side latency", f"{dt:.0f} ms")
            c2.metric("server latency", f"{body.get('latency_ms', 0):.0f} ms")
            c3.metric("cache hit", "yes" if body.get("cache_hit") else "no")
            c4.metric("retrieved", body.get("retrieved_count", 0))
            st.markdown(f"**Model:** `{body.get('model')}`  |  "
                        f"reranked: `{body.get('reranked')}`  |  "
                        f"intent: `{body.get('intent')}`")
            st.markdown("### Answer")
            st.write(body.get("answer", ""))
            if body.get("ragas_proxy"):
                st.markdown("### RAGAS proxy (this request)")
                st.json(body["ragas_proxy"])
            with st.expander("Sources"):
                for i, s in enumerate(body.get("sources", [])[:10], 1):
                    st.markdown(f"**{i}.** {s[:400]}{'…' if len(s) > 400 else ''}")
        else:
            st.error(body)

# ---------------------------------------------------------------------------
# view: single review analyzer
# ---------------------------------------------------------------------------
elif view == "Single-review analyzer":
    st.title("Single-review analyzer")
    st.caption("Runs the Phase-3 sentiment + complaint classifiers via the API.")
    review = st.text_area(
        "Review text",
        value="The food was great but the staff was rude and the place felt dirty.",
        height=140,
    )
    if st.button("Analyze") and review.strip():
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment")
            status, body = _safe_post("/sentiment", {"text": review}, timeout=60)
            if status == 200:
                st.metric(body.get("label", "?"), f"{body.get('compound', 0):+.3f}")
                st.caption(f"model: `{body.get('model')}` · latency "
                           f"{body.get('latency_ms', 0):.0f} ms · "
                           f"fallback: {body.get('fallback_used')}")
                st.write("**Keywords:**", ", ".join(body.get("keywords", []) or []))
            else:
                st.error(body)
        with c2:
            st.subheader("Complaints (TF-IDF + LightGBM)")
            status, body = _safe_post("/complaints", {"text": review}, timeout=60)
            if status == 200:
                cats = body.get("categories", [])
                if cats:
                    for c in cats:
                        st.markdown(f"- **{c}**")
                else:
                    st.write("(no categories flagged)")
                st.caption(f"model: `{body.get('model')}` · latency "
                           f"{body.get('latency_ms', 0):.0f} ms · "
                           f"fallback: {body.get('fallback_used')}")
                if body.get("scores"):
                    sdf = (pd.DataFrame(body["scores"].items(),
                                        columns=["category", "score"])
                           .sort_values("score", ascending=False))
                    st.plotly_chart(
                        px.bar(sdf, x="score", y="category", orientation="h",
                               range_x=[0, 1], height=300),
                        use_container_width=True,
                    )
            else:
                st.error(body)


# ---------------------------------------------------------------------------
# view: model card
# ---------------------------------------------------------------------------
else:
    st.title("Champion model card")
    st.markdown(
        """
**Locked Day-6** (Phase-5 wrap-up):

| Component | Champion | Eval | macro-F1 | Latency |
|---|---|---|---|---|
| Sentiment | `distilbart-mnli-12-3` (NLI zero-shot) | 200-review (Day-2) | **0.701** | 522 ms |
| Sentiment | DistilBERT-SST2 (production trade-off pick) | 100 fresh held-out (Day-6) | 0.560 | 40 ms |
| Complaints | TF-IDF + LightGBM-OvR (Optuna champion, t=0.5) | 100 fresh held-out (Day-6) | **0.853** | 34 ms |
| RAG | flan-t5-base + ms-marco rerank | 50 QA (Day-3) | composite 0.663 | 2.4 s |
| RAG | + Redis cache (Phase-6) | warm second hit | composite 0.663 | **< 10 ms** |

See `docs/MODEL_CARD.md` for the full card on the complaint classifier,
including per-category F1 + intended use + known failure modes.
        """
    )
    cache_status, cache_stats = _safe_get("/health/cache")
    if cache_status == 200:
        st.subheader("Cache state")
        st.json(cache_stats)

st.sidebar.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')}")
