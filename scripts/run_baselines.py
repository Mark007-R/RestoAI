"""Day-1 baseline runner for RestoAI.

Measures the current shipped behavior against the eval sets built in
data/eval/ and writes results/baseline_metrics.json plus per-component
prediction CSVs and a small samples directory for manual inspection.

Components evaluated:
  1. Sentiment    : analyze_text_and_keywords (VADER) on 200 reviews
  2. Complaints   : categorize_complaints (substring scan) on 100 reviews, multi-label
  3. RAG synth    : rag_chat._synthesize_intelligent_answer on 50 QA pairs,
                    judged by Claude Opus 4.6 (answer-only faithfulness + relevancy + groundedness in our extracted facts)

All Anthropic calls are cached on (model, prompt_hash) under results/.judge_cache.json
so re-runs are cheap. Set RESTOAI_SKIP_JUDGE=1 to skip the LLM judge entirely.
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "manager_system"))
EVAL_DIR = os.path.join(ROOT, "data", "eval")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("baselines")

# Quiet down third-party loggers
for noisy in ("manager_system", "rag_chat", "analyzer", "sentence_transformers", "transformers", "faiss", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

CACHE_PATH = os.path.join(RESULTS_DIR, ".judge_cache.json")
JUDGE_MODEL = "claude-opus-4-7"  # latest Opus per env
SKIP_JUDGE = os.getenv("RESTOAI_SKIP_JUDGE") == "1"

CATEGORIES = ["service", "food_quality", "hygiene", "price", "delivery", "portion", "ambience", "variety"]

# ---------------------------------------------------------------------------
# Sentiment baseline
# ---------------------------------------------------------------------------

def run_sentiment_baseline():
    from analyzer import analyze_text_and_keywords

    df = pd.read_csv(os.path.join(EVAL_DIR, "sentiment_eval.csv"))
    preds, labels = [], []
    rows = []
    for _, r in df.iterrows():
        label, comp, _kws = analyze_text_and_keywords(str(r["text"]))
        preds.append(label)
        labels.append(r["gold_label"])
        rows.append({
            "text_preview": str(r["text"])[:140],
            "gold": r["gold_label"],
            "vader_pred": label,
            "vader_compound": comp,
            "rating": r["rating"],
            "source": r["source"],
            "correct": label == r["gold_label"],
        })

    macro_f1 = f1_score(labels, preds, average="macro", labels=["Positive","Neutral","Negative"])
    p, rec, f, sup = precision_recall_fscore_support(labels, preds, labels=["Positive","Neutral","Negative"], zero_division=0)
    accuracy = float(np.mean([a == b for a, b in zip(labels, preds)]))
    per_class = {
        c: {"precision": float(p[i]), "recall": float(rec[i]), "f1": float(f[i]), "support": int(sup[i])}
        for i, c in enumerate(["Positive","Neutral","Negative"])
    }

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(RESULTS_DIR, "baseline_sentiment_preds.csv"), index=False)
    # save 5 wins / 5 losses as samples
    pred_df[pred_df["correct"]].head(5).to_csv(os.path.join(SAMPLES_DIR, "sentiment_baseline_wins.csv"), index=False)
    pred_df[~pred_df["correct"]].head(5).to_csv(os.path.join(SAMPLES_DIR, "sentiment_baseline_losses.csv"), index=False)

    return {
        "n": len(df),
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "per_class": per_class,
    }

# ---------------------------------------------------------------------------
# Complaint classifier baseline
# ---------------------------------------------------------------------------

def run_complaint_baseline():
    from analyzer import categorize_complaints

    df = pd.read_csv(os.path.join(EVAL_DIR, "complaint_eval.csv"))
    n = len(df)
    y_true = np.zeros((n, len(CATEGORIES)), dtype=int)
    y_pred = np.zeros((n, len(CATEGORIES)), dtype=int)

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        gold = [c.strip() for c in str(r["gold_labels"]).split(",") if c.strip()]
        pred = categorize_complaints(str(r["text"]))
        for c in gold:
            if c in CATEGORIES:
                y_true[i, CATEGORIES.index(c)] = 1
        for c in pred:
            if c in CATEGORIES:
                y_pred[i, CATEGORIES.index(c)] = 1
        rows.append({
            "text_preview": str(r["text"])[:160],
            "gold_labels": ",".join(sorted(gold)),
            "baseline_pred": ",".join(sorted(pred)),
            "exact_match": sorted(gold) == sorted(pred),
            "missed": ",".join(sorted(set(gold) - set(pred))),
            "spurious": ",".join(sorted(set(pred) - set(gold))),
        })

    # Per-class binary metrics
    per_class = {}
    for j, c in enumerate(CATEGORIES):
        p, rec, f, _ = precision_recall_fscore_support(
            y_true[:, j], y_pred[:, j], average="binary", zero_division=0
        )
        per_class[c] = {
            "precision": float(p), "recall": float(rec), "f1": float(f),
            "support": int(y_true[:, j].sum()),
        }

    macro_f1 = float(np.mean([per_class[c]["f1"] for c in CATEGORIES]))
    micro_f1 = float(f1_score(y_true.flatten(), y_pred.flatten(), average="binary", zero_division=0))
    exact_match_rate = float(np.mean([sorted(r["gold_labels"].split(",")) == sorted(r["baseline_pred"].split(","))
                                       for r in rows]))
    subset_acc = float(np.mean([(y_true[i] == y_pred[i]).all() for i in range(n)]))
    hamming = float(np.mean(y_true != y_pred))

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(RESULTS_DIR, "baseline_complaints_preds.csv"), index=False)
    pred_df[pred_df["exact_match"]].head(5).to_csv(os.path.join(SAMPLES_DIR, "complaints_baseline_wins.csv"), index=False)
    pred_df[~pred_df["exact_match"]].head(8).to_csv(os.path.join(SAMPLES_DIR, "complaints_baseline_losses.csv"), index=False)

    return {
        "n": n,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_acc,
        "exact_match_rate": exact_match_rate,
        "hamming_loss": hamming,
        "per_class": per_class,
    }

# ---------------------------------------------------------------------------
# RAG synthesis baseline
# ---------------------------------------------------------------------------

def _load_judge_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_judge_cache(c):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)

def _judge_key(prompt):
    return hashlib.sha256((JUDGE_MODEL + "::" + prompt).encode("utf-8")).hexdigest()

def llm_judge(answer, question, gold_facts, retrieved_snippets, anthropic_client, cache):
    """Score one (question, answer) pair on faithfulness + relevancy + grounding."""
    prompt = f"""You are an evaluator scoring an automated restaurant-review summary.

Question: {question}

Ground-truth facts (computed by aggregating the restaurant's reviews):
- Average rating: {gold_facts.get('avg_rating')}
- Sentiment direction: {gold_facts.get('sentiment_dir')}
- Top complaint categories: {gold_facts.get('top_categories')}

System answer (this is what we are scoring):
\"\"\"{answer}\"\"\"

Retrieved review snippets that the system was given:
{chr(10).join(f"- {s[:240]}" for s in retrieved_snippets[:5])}

Score the system answer on three axes from 1 (worst) to 5 (best):

1. faithfulness  - Does the answer avoid claims that contradict the retrieved snippets and ground-truth facts? (1=hallucinated/contradicting, 5=fully grounded)
2. relevancy     - Does the answer actually address the question? (1=off-topic boilerplate, 5=directly answers)
3. groundedness  - Does the answer reference specifics from the snippets/facts (sentiment direction, complaints, rating) rather than generic templates? (1=generic, 5=specific & cited)

Respond with ONLY one JSON object on a single line, no markdown, no prose:
{{"faithfulness": <int 1-5>, "relevancy": <int 1-5>, "groundedness": <int 1-5>, "reason": "<<=20 word justification>"}}"""

    key = _judge_key(prompt)
    if key in cache:
        return cache[key]

    try:
        resp = anthropic_client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # extract first json object
        m = re.search(r"\{.*?\}", text, re.S)
        if not m:
            raise ValueError(f"no json in: {text[:200]}")
        scores = json.loads(m.group(0))
        for k in ("faithfulness", "relevancy", "groundedness"):
            scores[k] = int(scores.get(k, 0))
        cache[key] = scores
        _save_judge_cache(cache)
        return scores
    except Exception as e:
        log.error(f"judge call failed: {e}")
        return {"faithfulness": 0, "relevancy": 0, "groundedness": 0, "reason": f"ERROR: {e}"}

def run_rag_baseline():
    """Use the existing RAGChat to synthesize answers via _synthesize_intelligent_answer.

    Note: index is built lazily as we ask questions per restaurant; the very first
    query for each restaurant triggers a load_csv_data + index_documents call.
    """
    with open(os.path.join(EVAL_DIR, "rag_qa_eval.json"), "r", encoding="utf-8") as f:
        qa = json.load(f)

    # Lazy import (heavy): SBERT model loads here
    print(f"  Initializing RAGChat (loads SBERT all-MiniLM-L6-v2)...")
    from rag_chat import RAGChat
    rag = RAGChat()

    rows = []
    print(f"  Running {len(qa)} questions...")
    for i, item in enumerate(qa, 1):
        q = item["question"]
        rest = item["restaurant"]
        try:
            answer, sources = rag.answer_query(q, restaurant_name=rest, top_k=5)
        except Exception as e:
            log.error(f"answer_query failed for {rest!r}: {e}")
            answer, sources = f"<ERROR: {e}>", []
        rows.append({
            "id": item["id"],
            "restaurant": rest,
            "intent": item["intent"],
            "question": q,
            "answer": answer,
            "sources": sources[:5],
            "gold_facts": item["gold_facts"],
        })
        if i % 10 == 0:
            print(f"    [{i}/{len(qa)}] retrieved")

    # Persist raw answers regardless of whether we judge
    raw_path = os.path.join(RESULTS_DIR, "baseline_rag_answers.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        # avoid serialising huge sources, trim
        slim = []
        for r in rows:
            slim.append({**r, "sources": [s[:300] for s in r["sources"]]})
        json.dump(slim, f, indent=2)

    summary = {"n": len(rows)}

    if SKIP_JUDGE:
        print("  RESTOAI_SKIP_JUDGE=1 set; skipping Claude judge.")
        summary.update({
            "faithfulness_mean": None,
            "relevancy_mean": None,
            "groundedness_mean": None,
            "composite": None,
            "judge_model": None,
            "note": "judge skipped",
        })
        return summary

    # LLM judge
    print(f"  Asking {JUDGE_MODEL} to judge {len(rows)} answers...")
    import anthropic
    client = anthropic.Anthropic()
    cache = _load_judge_cache()
    judged = []
    for i, r in enumerate(rows, 1):
        if isinstance(r["answer"], str) and r["answer"].startswith("<ERROR"):
            judged.append({**r, "judge": {"faithfulness": 0, "relevancy": 0, "groundedness": 0, "reason": "answer error"}})
            continue
        scores = llm_judge(
            answer=r["answer"],
            question=r["question"],
            gold_facts=r["gold_facts"],
            retrieved_snippets=r["sources"],
            anthropic_client=client,
            cache=cache,
        )
        judged.append({**r, "judge": scores})
        if i % 10 == 0:
            print(f"    judged {i}/{len(rows)}")

    judged_df = pd.DataFrame([{
        "id": j["id"], "restaurant": j["restaurant"], "intent": j["intent"],
        "faithfulness": j["judge"].get("faithfulness", 0),
        "relevancy": j["judge"].get("relevancy", 0),
        "groundedness": j["judge"].get("groundedness", 0),
        "reason": j["judge"].get("reason", ""),
        "answer_preview": str(j["answer"])[:200],
    } for j in judged])
    judged_df.to_csv(os.path.join(RESULTS_DIR, "baseline_rag_judged.csv"), index=False)

    # filter rows where judge succeeded (>0 means ran)
    valid = judged_df[(judged_df[["faithfulness","relevancy","groundedness"]] > 0).all(axis=1)]
    if len(valid) == 0:
        return {**summary, "note": "all judge calls failed"}

    summary.update({
        "faithfulness_mean": float(valid["faithfulness"].mean()),
        "relevancy_mean": float(valid["relevancy"].mean()),
        "groundedness_mean": float(valid["groundedness"].mean()),
        # geometric mean over normalised [0,1] scores
        "composite": float(np.exp(np.mean(np.log(valid[["faithfulness","relevancy","groundedness"]].values / 5.0), axis=None))),
        "judge_model": JUDGE_MODEL,
        "n_judged": int(len(valid)),
        "per_intent": {
            intent: {
                "faithfulness": float(valid[valid["intent"]==intent]["faithfulness"].mean()),
                "relevancy": float(valid[valid["intent"]==intent]["relevancy"].mean()),
                "groundedness": float(valid[valid["intent"]==intent]["groundedness"].mean()),
                "n": int((valid["intent"]==intent).sum()),
            } for intent in valid["intent"].unique()
        },
    })

    # samples
    judged_df.sort_values("faithfulness", ascending=False).head(5).to_csv(
        os.path.join(SAMPLES_DIR, "rag_baseline_top.csv"), index=False)
    judged_df.sort_values("groundedness").head(5).to_csv(
        os.path.join(SAMPLES_DIR, "rag_baseline_bottom.csv"), index=False)
    return summary

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    started = datetime.now()
    out = {
        "run_at": started.isoformat(timespec="seconds"),
        "components": {},
    }

    print("=" * 70); print("1. Sentiment baseline (VADER on 200 reviews)"); print("=" * 70)
    out["components"]["sentiment"] = {
        "engine": "vaderSentiment.SentimentIntensityAnalyzer (analyze_text_and_keywords)",
        "metrics": run_sentiment_baseline(),
    }
    print(f"  macro-F1 = {out['components']['sentiment']['metrics']['macro_f1']:.4f}")
    print(f"  accuracy = {out['components']['sentiment']['metrics']['accuracy']:.4f}")

    print("=" * 70); print("2. Complaint baseline (categorize_complaints on 100 reviews, multi-label)"); print("=" * 70)
    out["components"]["complaints"] = {
        "engine": "categorize_complaints (substring match against CATEGORY_KEYWORDS)",
        "metrics": run_complaint_baseline(),
    }
    print(f"  macro-F1     = {out['components']['complaints']['metrics']['macro_f1']:.4f}")
    print(f"  micro-F1     = {out['components']['complaints']['metrics']['micro_f1']:.4f}")
    print(f"  exact-match  = {out['components']['complaints']['metrics']['exact_match_rate']:.4f}")
    print(f"  per-class F1: {[(c, round(out['components']['complaints']['metrics']['per_class'][c]['f1'],3)) for c in CATEGORIES]}")

    print("=" * 70); print("3. RAG synthesis baseline (_synthesize_intelligent_answer on 50 QA)"); print("=" * 70)
    out["components"]["rag"] = {
        "engine": "RAGChat.answer_query -> _generate_answer -> _synthesize_intelligent_answer (template if/elif)",
        "metrics": run_rag_baseline(),
    }
    rag = out["components"]["rag"]["metrics"]
    if rag.get("composite") is not None:
        print(f"  faithfulness = {rag['faithfulness_mean']:.3f} / 5")
        print(f"  relevancy    = {rag['relevancy_mean']:.3f} / 5")
        print(f"  groundedness = {rag['groundedness_mean']:.3f} / 5")
        print(f"  composite    = {rag['composite']:.4f} (geo-mean of normalised)")

    out["elapsed_sec"] = (datetime.now() - started).total_seconds()
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'baseline_metrics.json')}")
    print(f"Elapsed: {out['elapsed_sec']:.1f}s")

if __name__ == "__main__":
    main()
