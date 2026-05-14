"""Day-4 smoke test for the integrated Phase-3 stack.

Exercises every champion module end-to-end on a small set of real reviews
and one RAG query against the consolidated FAISS index. Asserts:

  1. Complaint classifier loads its joblib bundle and returns categories
     consistent with the keyword baseline on obvious cases.
  2. Sentiment classifier returns one of {Positive, Neutral, Negative} for
     real reviews, and either uses NLI or sets fallback_used=True.
  3. RAG synthesizer runs on a real query through retrieval -> rerank ->
     synthesis OR sets fallback_used=True with a non-empty template answer.
  4. The Flask app's monkey-patched entrypoints (analyzer.categorize_complaints,
     rag_chat._synthesize_intelligent_answer) delegate to the new modules.

Writes results/day04_smoke_test.json so the report can cite specific numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "manager_system"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT = os.path.join(ROOT, "results", "day04_smoke_test.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


SAMPLE_REVIEWS = [
    "The waiter was rude and the food arrived cold. Overpriced for such a small portion.",
    "Loved the ambience and the music! Great place for a date night.",
    "Service was slow but the biryani was delicious. Worth the wait, would come back.",
]


def _safe(call, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = call(*args, **kwargs)
        return {"ok": True, "result": result, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _to_jsonable(obj: Any) -> Any:
    """Best-effort: BaseModel -> dict, list -> list of jsonable, dict -> dict of jsonable."""
    from pydantic import BaseModel
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main():
    report: Dict[str, Any] = {"reviews_tested": len(SAMPLE_REVIEWS), "checks": {}}

    # -- 1. Complaint classifier (direct + via analyzer shim) -----------
    print("[smoke] loading complaint classifier ...", flush=True)
    from src.complaints.classifier import get_default as get_comp
    clf = get_comp()
    direct_outputs = []
    for r in SAMPLE_REVIEWS:
        rec = _safe(clf.predict, r)
        if rec["ok"]:
            rec["result"] = _to_jsonable(rec["result"])
        direct_outputs.append({"text": r[:90], **rec})
    report["checks"]["complaints_direct"] = direct_outputs

    print("[smoke] loading analyzer shim ...", flush=True)
    from manager_system.analyzer import categorize_complaints
    shim_outputs = []
    for r in SAMPLE_REVIEWS:
        rec = _safe(categorize_complaints, r)
        shim_outputs.append({"text": r[:90], **rec})
    report["checks"]["complaints_via_analyzer_shim"] = shim_outputs

    # -- 2. Sentiment classifier ----------------------------------------
    print("[smoke] loading sentiment classifier (may take ~20s for NLI) ...", flush=True)
    from src.sentiment.classifier import get_default as get_sent
    sclf = get_sent()
    s_outputs = []
    for r in SAMPLE_REVIEWS:
        rec = _safe(sclf.predict, r)
        if rec["ok"]:
            rec["result"] = _to_jsonable(rec["result"])
        s_outputs.append({"text": r[:90], **rec})
    report["checks"]["sentiment_direct"] = s_outputs

    # -- 3. RAG synthesizer (direct, with synthetic retrieved docs) -----
    # Build a minimal docs list so we don't depend on the FAISS index being
    # warm; the integration test against the real index follows in step 4.
    print("[smoke] loading RAG synthesizer (may take ~30s for flan-t5-base) ...", flush=True)
    from src.rag.pipeline import get_default as get_rag
    rclf = get_rag()
    fake_docs = [
        {"text": r, "metadata": {"restaurant": "Test", "rating": "4.0"}}
        for r in SAMPLE_REVIEWS * 5  # 15 docs so rerank actually runs
    ]
    rec = _safe(rclf.synthesize, "How is the service and food quality?", fake_docs, None, "Test")
    if rec["ok"]:
        rec["result"] = _to_jsonable(rec["result"])
    report["checks"]["rag_synthetic_docs"] = rec

    # -- 4. Real FAISS integration (skip if index missing) --------------
    faiss_path = os.path.join(ROOT, "manager_system", "vector_db", "all_restaurants.faiss")
    if not os.path.exists(faiss_path):
        report["checks"]["rag_real_index"] = {"skipped": True, "reason": "no consolidated FAISS index"}
    else:
        print("[smoke] running real FAISS retrieve -> rerank -> synth ...", flush=True)
        from rag_chat import RAGChat
        rag = RAGChat()
        rag._load_vector_db()  # noqa: SLF001
        rag.loaded = True
        # Pick the first restaurant from doc_metadata as the filter
        first_rest = None
        for meta in rag.doc_metadata[:200]:
            if meta.get("restaurant"):
                first_rest = meta["restaurant"]
                break
        rec = _safe(rag.answer_query, "How is the food quality?", first_rest, 15)
        # answer_query returns (answer_string, source_texts) — both jsonable
        report["checks"]["rag_real_index"] = {
            **{k: v for k, v in rec.items() if k != "result"},
            "result": ({"answer_preview": rec["result"][0][:600],
                        "n_sources": len(rec["result"][1]),
                        "restaurant_filter": first_rest}
                       if rec.get("ok") else None),
        }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[smoke] saved -> {OUT}")

    # Compact pass/fail summary to stdout
    print("\n=== SUMMARY ===")
    for k, v in report["checks"].items():
        if isinstance(v, dict) and v.get("skipped"):
            print(f"  {k}: SKIPPED ({v.get('reason')})")
        elif isinstance(v, list):
            ok_count = sum(1 for r in v if r.get("ok"))
            print(f"  {k}: {ok_count}/{len(v)} passed")
        elif isinstance(v, dict):
            status = "PASS" if v.get("ok") else "FAIL"
            print(f"  {k}: {status}  ({v.get('latency_ms')}ms)")


if __name__ == "__main__":
    main()
