"""Structural metrics for RAG synthesis baseline.

Replaces the LLM judge for Day 1 because the autonomous run cannot reach the
Anthropic API. These metrics are entirely deterministic and directly reveal
the template pathology described in docs/COMPONENT_AUDIT.md:

  - sentiment_dir_match : does the answer's positive/negative tone match the
                          gold aggregate (computed from all reviews)?
  - top_category_hit    : does the answer reference any of the gold top-3
                          complaint categories for that restaurant?
  - rating_mention      : does the answer mention the avg rating numerically?
  - intent_addressed    : does the answer use vocabulary from the question's
                          intent (quality/service/price/hygiene/ambience/recommend)?
  - specificity_score   : 0..1, fraction of {has_rating, has_top_cat_term,
                          has_intent_term, has_pos_or_neg_token} signals present.
  - composite           : geometric mean of the four binary rates.
"""

import os, json, re
import numpy as np
import pandas as pd
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

INTENT_VOCAB = {
    "quality":   {"quality","tasty","taste","flavour","flavor","delicious","food"},
    "service":   {"service","staff","waiter","wait","server","attitude"},
    "price":     {"price","cost","expensive","cheap","value","worth","pricey","money"},
    "hygiene":   {"clean","hygiene","dirty","sanit","smell"},
    "ambience":  {"ambience","ambient","atmosphere","decor","vibe","interior","seating"},
    "recommend": {"recommend","suggest","worth","best","avoid","try"},
}

CATEGORY_TERMS = {
    "service":      {"service","staff","waiter","wait","server"},
    "food_quality": {"food","quality","taste","flavour","flavor","cooked","fresh","stale"},
    "hygiene":      {"clean","hygiene","dirty","filthy","sanit","smell"},
    "price":        {"price","cost","expensive","cheap","value","worth"},
    "delivery":     {"delivery","late","driver","packag","arriv"},
    "portion":      {"portion","quantity","size","serving","small","tiny"},
    "ambience":     {"ambience","ambient","atmosphere","decor","vibe","interior","seating"},
    "variety":      {"menu","options","variety","limited","choices","selection"},
}

POS_TOKENS = {"praised","positive","good","great","recommended","highly","appreciated","excellent","love"}
NEG_TOKENS = {"caution","concern","mixed","negative","poor","slow","needs improvement","disappointing","avoid"}

RATING_RE = re.compile(r"\b\d(?:\.\d)?\s*/\s*5\b|\baverage\s+rating[^0-9]*\d", re.I)

def detect_dir(text):
    t = text.lower()
    pos = sum(1 for w in POS_TOKENS if w in t)
    neg = sum(1 for w in NEG_TOKENS if w in t)
    if pos > neg: return "positive"
    if neg > pos: return "negative"
    return "mixed"

def score_one(answer, gold_facts, intent):
    a = answer.lower()
    gold_dir = gold_facts.get("sentiment_dir","mixed")
    pred_dir = detect_dir(a)
    sentiment_match = (pred_dir == gold_dir)

    top_cats = gold_facts.get("top_categories", []) or []
    top_cat_hit = False
    for c in top_cats[:3]:
        terms = CATEGORY_TERMS.get(c, set())
        if any(t in a for t in terms):
            top_cat_hit = True
            break

    rating_mention = bool(RATING_RE.search(answer))
    intent_terms = INTENT_VOCAB.get(intent, set())
    intent_addressed = any(t in a for t in intent_terms)

    sigs = [rating_mention, top_cat_hit, intent_addressed, (pred_dir != "mixed")]
    specificity = sum(sigs) / len(sigs)

    return {
        "sentiment_dir_match": int(sentiment_match),
        "top_category_hit":    int(top_cat_hit),
        "rating_mention":      int(rating_mention),
        "intent_addressed":    int(intent_addressed),
        "specificity_score":   round(specificity, 3),
        "pred_dir":            pred_dir,
        "gold_dir":            gold_dir,
    }

def main():
    with open(os.path.join(RESULTS, "baseline_rag_answers.json"), "r", encoding="utf-8") as f:
        answers = json.load(f)

    rows = []
    for r in answers:
        ans = r["answer"]
        if not isinstance(ans, str) or ans.startswith("<ERROR"):
            rows.append({"id": r["id"], "intent": r["intent"], "restaurant": r["restaurant"],
                         "sentiment_dir_match": 0, "top_category_hit": 0,
                         "rating_mention": 0, "intent_addressed": 0,
                         "specificity_score": 0.0, "answer_failed": True})
            continue
        s = score_one(ans, r["gold_facts"], r["intent"])
        rows.append({"id": r["id"], "intent": r["intent"], "restaurant": r["restaurant"], **s,
                     "answer_failed": False, "answer_preview": ans[:160]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "baseline_rag_structural.csv"), index=False)

    valid = df[~df["answer_failed"]]
    n = len(valid)
    sd = float(valid["sentiment_dir_match"].mean())
    tc = float(valid["top_category_hit"].mean())
    rm = float(valid["rating_mention"].mean())
    ia = float(valid["intent_addressed"].mean())
    sp = float(valid["specificity_score"].mean())
    # geo mean of the four binary rates (small epsilon to avoid log(0))
    eps = 1e-3
    rates = np.array([max(sd, eps), max(tc, eps), max(rm, eps), max(ia, eps)])
    composite = float(np.exp(np.mean(np.log(rates))))

    per_intent = {}
    for intent in valid["intent"].unique():
        sub = valid[valid["intent"] == intent]
        per_intent[intent] = {
            "n": int(len(sub)),
            "sentiment_dir_match": float(sub["sentiment_dir_match"].mean()),
            "top_category_hit":    float(sub["top_category_hit"].mean()),
            "rating_mention":      float(sub["rating_mention"].mean()),
            "intent_addressed":    float(sub["intent_addressed"].mean()),
            "specificity_score":   float(sub["specificity_score"].mean()),
        }

    out = {
        "n": int(len(df)),
        "n_scored": n,
        "n_failed": int((df["answer_failed"]).sum()),
        "sentiment_dir_match": sd,
        "top_category_hit": tc,
        "rating_mention": rm,
        "intent_addressed": ia,
        "specificity_score_mean": sp,
        "composite": composite,
        "per_intent": per_intent,
        "metric_definition": {
            "sentiment_dir_match": "answer's pos/neg/mixed direction matches gold aggregate",
            "top_category_hit":    "answer references any of restaurant's top-3 complaint categories",
            "rating_mention":      "answer cites a numeric rating like '4.2/5'",
            "intent_addressed":    "answer uses vocabulary from question's intent group",
            "specificity_score":   "fraction of {rating, top_cat, intent, non-mixed_dir} signals present",
            "composite":           "geometric mean of the four binary rates",
        },
        "note": "Day-1 LLM judge unavailable in autonomous subprocess (ANTHROPIC_API_KEY not exposed). Structural metrics substituted; they are deterministic, reproducible, and directly probe the template-only pathology documented in docs/COMPONENT_AUDIT.md. Day 3 LLM-judged RAGAS will replace this once an interactive run is available.",
    }

    print(f"Structural RAG metrics over {n} answers:")
    for k in ("sentiment_dir_match","top_category_hit","rating_mention","intent_addressed","specificity_score_mean","composite"):
        print(f"  {k:24s} = {out[k]:.4f}")

    # update results/baseline_metrics.json
    metrics_path = os.path.join(RESULTS, "baseline_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        full = json.load(f)
    full["components"]["rag"]["metrics"] = out
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    print(f"\nUpdated {metrics_path}")

if __name__ == "__main__":
    main()
