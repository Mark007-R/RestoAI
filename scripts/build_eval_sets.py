"""Build the three evaluation sets for RestoAI Day 1 baseline.

Outputs (under data/eval/):
  - sentiment_eval.csv         : 200 rows, columns [text, restaurant, source, gold_label]
  - complaint_eval.csv         : 100 rows, columns [text, restaurant, source, gold_labels]
  - rag_qa_eval.json           : 50 QA items {question, restaurant, intent, gold_facts}
  - eval_corpus_meta.json      : provenance & label-construction notes

Labels are constructed programmatically in a fully reproducible way (Day-1 is autonomous).
The construction is documented inline so Day-5 error analysis can re-validate.
"""

import os
import re
import json
import ast
import random
import sys
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets")
OUT_DIR = os.path.join(ROOT, "data", "eval")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Pull a uniform pool of reviews from all 5 datasets.
# ---------------------------------------------------------------------------

def safe_read(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    raise IOError(f"Could not read {path}")

def normalize_rating(val):
    """Return float in [0,5] or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        f = float(m.group(0))
    except Exception:
        return None
    # zomato format e.g. "4.1/5"
    if "/5" in s:
        return f
    if f > 5:  # already 0-100 scale or noise
        return None
    return f

def load_pool():
    pool = []

    # mumbaires
    p = os.path.join(DATA_DIR, "mumbaires.csv")
    if os.path.exists(p):
        df = safe_read(p)
        for _, row in df.iterrows():
            txt = str(row.get("Review Text", "") or "").strip()
            if 30 <= len(txt) <= 1500:
                pool.append({
                    "text": txt,
                    "restaurant": str(row.get("Restaurant Name", "") or "").strip(),
                    "rating": normalize_rating(row.get("Reviewer Rating")),
                    "source": "mumbaires.csv",
                })

    # Resreviews
    p = os.path.join(DATA_DIR, "Resreviews.csv")
    if os.path.exists(p):
        df = safe_read(p)
        for _, row in df.iterrows():
            txt = str(row.get("Review", "") or "").strip()
            if 30 <= len(txt) <= 1500:
                pool.append({
                    "text": txt,
                    "restaurant": str(row.get("Restaurant", "") or "").strip(),
                    "rating": normalize_rating(row.get("Rating")),
                    "source": "Resreviews.csv",
                })

    # reviews.csv
    p = os.path.join(DATA_DIR, "reviews.csv")
    if os.path.exists(p):
        df = safe_read(p)
        for _, row in df.iterrows():
            txt = str(row.get("text", "") or "").strip()
            if 30 <= len(txt) <= 1500:
                pool.append({
                    "text": txt,
                    "restaurant": str(row.get("business_name", "") or "").strip(),
                    "rating": normalize_rating(row.get("rating")),
                    "source": "reviews.csv",
                })

    # zomato.csv -- reviews are inside a list literal in reviews_list
    p = os.path.join(DATA_DIR, "zomato.csv")
    if os.path.exists(p):
        df = safe_read(p)
        # use a sample for speed (51K restaurants)
        df_s = df.sample(n=min(2000, len(df)), random_state=42)
        for _, row in df_s.iterrows():
            raw = row.get("reviews_list", "")
            if not isinstance(raw, str) or not raw.strip().startswith("["):
                continue
            try:
                lst = ast.literal_eval(raw)
            except Exception:
                continue
            if not isinstance(lst, list):
                continue
            rest = str(row.get("name", "") or "").strip()
            for item in lst[:5]:
                if not (isinstance(item, tuple) and len(item) == 2):
                    continue
                rated, body = item
                txt = str(body or "").replace("RATED\n", "").strip()
                if 30 <= len(txt) <= 1500:
                    rating = None
                    m = re.search(r"Rated\s+([\d.]+)", str(rated))
                    if m:
                        try: rating = float(m.group(1))
                        except: pass
                    pool.append({
                        "text": txt,
                        "restaurant": rest,
                        "rating": rating,
                        "source": "zomato.csv",
                    })

    # zomato2.csv -- not really reviews (item names) but we sample a few
    p = os.path.join(DATA_DIR, "zomato2.csv")
    if os.path.exists(p):
        df = safe_read(p)
        df_s = df.sample(n=min(1500, len(df)), random_state=42)
        for _, row in df_s.iterrows():
            item = str(row.get("Item_Name", "") or "").strip()
            if not item or item.lower() == "nan":
                continue
            best = bool(row.get("Best_Seller", 0))
            votes = row.get("Votes", 0)
            try: votes_i = int(votes) if not pd.isna(votes) else 0
            except: votes_i = 0
            txt = f"Tried {item}"
            if best: txt += " - this is a best seller"
            if votes_i > 0: txt += f" with {votes_i} votes"
            # zomato2 records are item-mentions, NOT review prose.
            # We tag source so they can be excluded from prose-style evals.
            if 30 <= len(txt) <= 400:
                pool.append({
                    "text": txt,
                    "restaurant": str(row.get("Restaurant_Name", "") or "").strip(),
                    "rating": normalize_rating(row.get("Average_Rating")),
                    "source": "zomato2.csv",
                })

    return pool

# ---------------------------------------------------------------------------
# 2. Build the SENTIMENT eval set (200 rows, label from star rating).
# ---------------------------------------------------------------------------

def build_sentiment_eval(pool, n=200):
    """Star-rating-derived labels. Stratified across {Pos, Neg, Neu} and across sources.

    rating >= 4 -> Positive ; rating == 3 -> Neutral ; rating <= 2 -> Negative.
    Reviews without a rating are skipped (we need ground truth).
    Excludes zomato2.csv item-mentions (not prose).
    """
    prose = [r for r in pool if r["source"] != "zomato2.csv" and r["rating"] is not None]
    buckets = {"Positive": [], "Negative": [], "Neutral": []}
    for r in prose:
        rt = r["rating"]
        if rt >= 4: buckets["Positive"].append(r)
        elif rt <= 2: buckets["Negative"].append(r)
        elif rt == 3 or (rt > 2 and rt < 4): buckets["Neutral"].append(r)

    per_class = n // 3  # 66 each, +2 leftover -> Positive
    counts = {"Positive": per_class + (n - per_class * 3), "Negative": per_class, "Neutral": per_class}

    rows = []
    for label, k in counts.items():
        bucket = buckets[label]
        random.shuffle(bucket)
        # de-dup by (text first 80 chars) within the class
        seen = set()
        picked = []
        for r in bucket:
            key = r["text"][:80].lower()
            if key in seen: continue
            seen.add(key)
            picked.append(r)
            if len(picked) >= k:
                break
        for r in picked:
            rows.append({
                "text": r["text"],
                "restaurant": r["restaurant"],
                "source": r["source"],
                "rating": r["rating"],
                "gold_label": label,
            })
    random.shuffle(rows)
    return rows[:n]

# ---------------------------------------------------------------------------
# 3. Build the COMPLAINT eval set (100 rows, multi-label across 8 categories).
# ---------------------------------------------------------------------------
# We build a *rich* labeller — broader than the baseline's substring scan — and
# treat its output as gold. This intentionally measures how much the baseline
# *misses*. Day-5 error analysis re-validates a sample.

RICH_PATTERNS = {
    "service": [
        r"\bservice(?:s)?\b", r"\bwait(?:ed|ing|er|ress|staff)?\b", r"\bstaff\b",
        r"\bserver\b", r"\battitude\b", r"\brude\b", r"\bunfriend\w*\b",
        r"\bimpolite\b", r"\bignor(?:ed|ing)\b", r"\battention\b",
        r"\bslow(?:er)?\b.*\b(?:service|wait|staff|food|order)\b",
        r"\b(?:bad|poor|terrible|horrible) service\b", r"\bunprofessional\b",
        r"\bmanager\b.*\b(?:rude|attitude|bad)\b",
    ],
    "food_quality": [
        r"\b(?:cold|burnt|undercooked|bland|overcooked|raw|soggy|stale|spoil\w*|rotten)\b",
        r"\btast(?:e|y|eless|ed)\b", r"\bflavou?r(?:less|ed)?\b", r"\bdry\b",
        r"\bgreasy\b", r"\boily\b", r"\bhard\b.*\b(?:to eat|like rock|stale)\b",
        r"\b(?:not|no)\s+fresh\b", r"\bdisgust\w*\b", r"\bawful\b.*\b(?:food|dish|meal)\b",
        r"\b(?:bad|poor|worst|terrible|horrible)\b.*\b(?:food|dish|meal|biryani|curry|pizza|burger|sandwich)\b",
        r"\bquality\b.*\b(?:food|ingredient)\b", r"\bnot\s+cooked\b",
    ],
    "hygiene": [
        r"\bdirty\b", r"\bhygien\w*\b", r"\bunclean\b", r"\bunhygien\w*\b",
        r"\bfilth\w*\b", r"\bsmell\w*\b", r"\bstink\w*\b", r"\bsanit\w*\b",
        r"\bcockroach\w*\b", r"\binsect\w*\b", r"\bflies\b",
        r"\bnot\s+clean\b", r"\b(?:hair|fly|insect)\s+in\s+(?:my|the)\b",
    ],
    "price": [
        r"\bexpens\w*\b", r"\boverprice\w*\b", r"\bpric(?:e|ey|ing|y)\b",
        r"\bcost\w*\b", r"\bvalue\b.*\bmoney\b", r"\bnot\s+worth\b",
        r"\b(?:too|very|highly?)\s+(?:expensive|costly|pricey)\b",
        r"\bpaisa\s+vasool\b", r"\baffordable\b", r"\bcheap\b",
    ],
    "delivery": [
        r"\bdeliver\w*\b", r"\b(?:late|delay\w*)\b.*\b(?:order|deliver|food|arriv)\b",
        r"\bpackag\w*\b", r"\bmissing\b.*\b(?:item|order|food)\b",
        r"\bdriver\b", r"\b(?:cold|cool)\s+(?:on|when|by)\s+(?:arrival|delivery)\b",
        r"\bdamaged\b.*\b(?:packag|order|food)\b", r"\bswiggy\b", r"\bzomato\s+order\b",
        r"\barrived\b.*\b(?:late|cold|damaged|spilled)\b",
    ],
    "portion": [
        r"\b(?:small|tiny|inadequate|less|insufficient)\b.*\b(?:portion|quantity|size|serving)\b",
        r"\bportion\b.*\b(?:small|tiny|less|inadequate)\b",
        r"\bquantity\b", r"\bserving\s+size\b", r"\bnot\s+(?:enough|filling)\b",
        r"\bvery\s+(?:little|small)\b.*\b(?:food|amount)\b",
    ],
    "ambience": [
        r"\bambien(?:ce|t)\b", r"\batmosphere\b", r"\bdecor\b", r"\bvibe\b",
        r"\b(?:loud|noisy|crowded)\b", r"\blighting\b", r"\bmusic\b",
        r"\bseating\b", r"\binterior\b", r"\bcomfort\w*\b.*\b(?:seat|chair|space)\b",
        r"\b(?:beautiful|lovely|cosy|cozy)\b.*\b(?:place|interior|ambien|decor)\b",
    ],
    "variety": [
        r"\bmenu\b.*\b(?:limited|small|few|variety|options)\b",
        r"\b(?:limited|few|fewer)\s+options\b", r"\bvariety\b",
        r"\bchoice(?:s)?\b.*\b(?:limited|few|small)\b",
        r"\bselection\b.*\b(?:limited|small|few|good|great)\b",
    ],
}

def rich_label(text):
    t = text.lower()
    hits = []
    for cat, patterns in RICH_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t):
                hits.append(cat)
                break
    return list(dict.fromkeys(hits))

def build_complaint_eval(pool, n=100):
    """Stratified so each of 8 categories is represented in >=8 reviews.
    Uses the RICH_PATTERNS labeller as gold (broader than baseline keyword scan).
    """
    prose = [r for r in pool if r["source"] not in ("zomato2.csv",)]
    # First pass: assign rich labels
    labeled = []
    for r in prose:
        cats = rich_label(r["text"])
        if cats:
            labeled.append({**r, "gold_labels": cats})

    # Stratified sampling: target ~12 per category, with multi-label allowed
    by_cat = defaultdict(list)
    for r in labeled:
        for c in r["gold_labels"]:
            by_cat[c].append(r)
    for c in by_cat:
        random.shuffle(by_cat[c])

    target_per_cat = max(8, n // 8)  # 12
    chosen = {}  # text-key -> row
    for cat in RICH_PATTERNS.keys():
        added = 0
        for r in by_cat.get(cat, []):
            key = r["text"][:80].lower()
            if key in chosen: continue
            chosen[key] = r
            added += 1
            if added >= target_per_cat: break

    # Fill to n with multi-label rich rows
    multi = [r for r in labeled if len(r["gold_labels"]) >= 2]
    random.shuffle(multi)
    for r in multi:
        if len(chosen) >= n: break
        key = r["text"][:80].lower()
        if key in chosen: continue
        chosen[key] = r

    # If still under n, fill with single-label rows
    if len(chosen) < n:
        random.shuffle(labeled)
        for r in labeled:
            if len(chosen) >= n: break
            key = r["text"][:80].lower()
            if key in chosen: continue
            chosen[key] = r

    rows = list(chosen.values())[:n]
    random.shuffle(rows)
    out = []
    for r in rows:
        out.append({
            "text": r["text"],
            "restaurant": r["restaurant"],
            "source": r["source"],
            "rating": r["rating"],
            "gold_labels": ",".join(r["gold_labels"]),
        })
    return out

# ---------------------------------------------------------------------------
# 4. Build the RAG QA eval set (50 questions across restaurants & intents).
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = {
    "quality": "How is the food quality at {r}?",
    "service": "What do customers say about the service at {r}?",
    "price": "Is {r} good value for money?",
    "hygiene": "Are there any hygiene concerns at {r}?",
    "ambience": "What is the ambience like at {r}?",
    "recommend": "Would customers recommend {r}?",
}

POSITIVE_W = {"good","great","excellent","amazing","delicious","perfect","wonderful","fantastic","love","best","awesome","outstanding","tasty"}
NEGATIVE_W = {"bad","poor","terrible","horrible","awful","worst","disappointing","waste","avoid","never","disgusting","pathetic","slow","cold","rude"}

def aggregate_facts_for_restaurant(reviews):
    """Compute structured ground-truth facts for a restaurant."""
    texts = [r["text"].lower() for r in reviews]
    all_text = " ".join(texts)
    pos = sum(all_text.count(w) for w in POSITIVE_W)
    neg = sum(all_text.count(w) for w in NEGATIVE_W)
    ratings = [r["rating"] for r in reviews if r["rating"] is not None]
    avg_r = round(sum(ratings)/len(ratings), 2) if ratings else None
    sentiment_dir = "positive" if pos > neg * 1.2 else ("negative" if neg > pos * 1.2 else "mixed")
    # complaints distribution
    cat_counts = Counter()
    for t in texts:
        for cat in rich_label(t):
            cat_counts[cat] += 1
    top_complaints = [c for c,_ in cat_counts.most_common(3)]
    return {
        "n_reviews": len(reviews),
        "avg_rating": avg_r,
        "sentiment_dir": sentiment_dir,
        "pos_term_count": pos,
        "neg_term_count": neg,
        "top_categories": top_complaints,
    }

def build_rag_qa(pool, n=50):
    by_rest = defaultdict(list)
    for r in pool:
        if r["source"] == "zomato2.csv": continue
        if not r["restaurant"]: continue
        by_rest[r["restaurant"]].append(r)
    # restaurants with at least 8 reviews
    eligible = [(rest, revs) for rest, revs in by_rest.items() if len(revs) >= 8]
    eligible.sort(key=lambda x: -len(x[1]))
    # sample diverse restaurants -- avoid one mega restaurant dominating
    pick = eligible[:120]
    random.shuffle(pick)

    items = []
    intents = list(QUESTION_TEMPLATES.keys())
    used_keys = set()
    rest_idx = 0
    while len(items) < n and rest_idx < len(pick):
        rest, revs = pick[rest_idx]
        rest_idx += 1
        intent = intents[len(items) % len(intents)]
        key = (rest.lower(), intent)
        if key in used_keys: continue
        used_keys.add(key)
        facts = aggregate_facts_for_restaurant(revs)
        items.append({
            "id": f"q{len(items)+1:03d}",
            "restaurant": rest,
            "intent": intent,
            "question": QUESTION_TEMPLATES[intent].format(r=rest),
            "gold_facts": facts,
            "n_source_reviews": len(revs),
        })
    return items[:n]

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading review pool from 5 datasets...")
    pool = load_pool()
    print(f"  Total pooled review snippets: {len(pool)}")
    by_src = Counter(r["source"] for r in pool)
    for s,c in by_src.items():
        print(f"    {s}: {c}")

    print("\nBuilding sentiment eval (200 rows, label from rating)...")
    sent = build_sentiment_eval(pool, n=200)
    pd.DataFrame(sent).to_csv(os.path.join(OUT_DIR, "sentiment_eval.csv"), index=False)
    sent_dist = Counter(r["gold_label"] for r in sent)
    print(f"  Distribution: {dict(sent_dist)} | n={len(sent)}")
    by_src_s = Counter(r["source"] for r in sent)
    print(f"  By source: {dict(by_src_s)}")

    print("\nBuilding complaint eval (100 rows, multi-label, rich-pattern gold)...")
    comp = build_complaint_eval(pool, n=100)
    pd.DataFrame(comp).to_csv(os.path.join(OUT_DIR, "complaint_eval.csv"), index=False)
    cat_dist = Counter()
    for r in comp:
        for c in r["gold_labels"].split(","):
            cat_dist[c] += 1
    print(f"  Per-category support: {dict(cat_dist)} | n={len(comp)}")

    print("\nBuilding RAG QA eval (50 questions across restaurants & intents)...")
    rag = build_rag_qa(pool, n=50)
    with open(os.path.join(OUT_DIR, "rag_qa_eval.json"), "w", encoding="utf-8") as f:
        json.dump(rag, f, indent=2)
    intent_dist = Counter(q["intent"] for q in rag)
    print(f"  Per-intent: {dict(intent_dist)} | n={len(rag)}")
    print(f"  Distinct restaurants: {len(set(q['restaurant'] for q in rag))}")

    meta = {
        "constructed_on": "2026-05-11",
        "label_construction": {
            "sentiment": "rating>=4 Positive | rating==3 Neutral | rating<=2 Negative; ratings without numeric value skipped; zomato2.csv item-mentions excluded.",
            "complaints": "Multi-label assigned by RICH_PATTERNS regex (broader than baseline substring scan). Stratified to ensure each of 8 categories has >=8 examples.",
            "rag": "Question = template per intent; ground truth = aggregate facts (avg_rating, sentiment_direction, top categories) computed across all reviews of the restaurant. Answer-only judging in Day 1.",
        },
        "pool_size": len(pool),
        "by_source_pool": {s: int(c) for s, c in by_src.items()},
        "eval_sizes": {"sentiment": len(sent), "complaint": len(comp), "rag": len(rag)},
    }
    with open(os.path.join(OUT_DIR, "eval_corpus_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote eval files to {OUT_DIR}")

if __name__ == "__main__":
    main()
