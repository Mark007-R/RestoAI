"""Day-1 baseline visualizations.

Outputs PNGs under results/charts/:
  - sentiment_confusion.png        (3x3 confusion matrix)
  - sentiment_by_source.png        (grouped per-class F1 by source dataset)
  - complaints_per_class_f1.png    (per-class F1 with support overlay)
  - complaints_failure_breakdown.png (missed vs spurious by category)
  - rag_per_intent_metrics.png     (4 structural metrics across 6 intents)
  - rag_specificity_distribution.png (specificity score histogram)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CHART_DIR = os.path.join(RESULTS, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "figure.autolayout": True})

CATEGORIES = ["service","food_quality","hygiene","price","delivery","portion","ambience","variety"]
SENT_LABELS = ["Positive","Neutral","Negative"]

def chart_sentiment_confusion():
    df = pd.read_csv(os.path.join(RESULTS, "baseline_sentiment_preds.csv"))
    cm = confusion_matrix(df["gold"], df["vader_pred"], labels=SENT_LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(SENT_LABELS); ax.set_yticklabels(SENT_LABELS)
    ax.set_xlabel("VADER prediction"); ax.set_ylabel("Gold (rating-derived)")
    ax.set_title("Sentiment baseline confusion matrix (VADER, n=200)")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = os.path.join(CHART_DIR, "sentiment_confusion.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

def chart_sentiment_by_source():
    with open(os.path.join(RESULTS, "baseline_slices.json")) as f:
        slices = json.load(f)["sentiment_by_source"]
    sources = list(slices.keys())
    pos = [slices[s]["per_class_f1"]["Positive"] for s in sources]
    neu = [slices[s]["per_class_f1"]["Neutral"]  for s in sources]
    neg = [slices[s]["per_class_f1"]["Negative"] for s in sources]
    x = np.arange(len(sources)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w, pos, w, label="Positive", color="#16a34a")
    ax.bar(x,     neu, w, label="Neutral",  color="#94a3b8")
    ax.bar(x + w, neg, w, label="Negative", color="#dc2626")
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("VADER per-class F1 by source dataset (Neutral fails uniformly)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(neu):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    out = os.path.join(CHART_DIR, "sentiment_by_source.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

def chart_complaints_per_class():
    with open(os.path.join(RESULTS, "baseline_metrics.json")) as f:
        m = json.load(f)["components"]["complaints"]["metrics"]["per_class"]
    cats = sorted(CATEGORIES, key=lambda c: -m[c]["f1"])
    f1s = [m[c]["f1"] for c in cats]
    sup = [m[c]["support"] for c in cats]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(cats, f1s, color=["#16a34a" if v > 0.85 else "#f59e0b" if v > 0.7 else "#dc2626" for v in f1s])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Complaint baseline (substring keyword scan) per-class F1")
    for b, v, s in zip(bars, f1s, sup):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}\nn={s}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    out = os.path.join(CHART_DIR, "complaints_per_class_f1.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

def chart_complaints_failure_breakdown():
    df = pd.read_csv(os.path.join(RESULTS, "baseline_complaints_preds.csv"))
    missed_counts = {c: 0 for c in CATEGORIES}
    spurious_counts = {c: 0 for c in CATEGORIES}
    for _, r in df.iterrows():
        for c in str(r.get("missed","")).split(","):
            c = c.strip()
            if c in CATEGORIES: missed_counts[c] += 1
        for c in str(r.get("spurious","")).split(","):
            c = c.strip()
            if c in CATEGORIES: spurious_counts[c] += 1
    cats = CATEGORIES
    miss = [missed_counts[c] for c in cats]
    spur = [spurious_counts[c] for c in cats]
    x = np.arange(len(cats)); w = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w/2, miss, w, label="Missed (false negative)", color="#dc2626")
    ax.bar(x + w/2, spur, w, label="Spurious (false positive)", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Count of failures (out of 100)")
    ax.set_title("Complaint baseline: missed vs spurious labels by category")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = os.path.join(CHART_DIR, "complaints_failure_breakdown.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

def chart_rag_per_intent():
    with open(os.path.join(RESULTS, "baseline_slices.json")) as f:
        rag = json.load(f)["rag_by_intent"]
    intents = list(rag.keys())
    metrics = ["sentiment_dir_match","top_category_hit","rating_mention","intent_addressed"]
    nice = ["sent dir","top cat","rating cite","intent addr"]
    x = np.arange(len(intents)); w = 0.20
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#3b82f6","#16a34a","#f59e0b","#a855f7"]
    for i, m in enumerate(metrics):
        vals = [rag[it][m] for it in intents]
        ax.bar(x + (i-1.5)*w, vals, w, label=nice[i], color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(intents)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (0..1)")
    ax.set_title("RAG template baseline: 4 structural metrics across 6 intents")
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.22)
    out = os.path.join(CHART_DIR, "rag_per_intent_metrics.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

def chart_rag_specificity_dist():
    df = pd.read_csv(os.path.join(RESULTS, "baseline_rag_structural.csv"))
    df = df[~df["answer_failed"].astype(bool)] if "answer_failed" in df else df
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["specificity_score"], bins=[-0.05, 0.2, 0.45, 0.7, 0.95, 1.05], color="#3b82f6", edgecolor="white")
    ax.set_xlabel("specificity_score (fraction of {rating, top_cat, intent, dir} signals)")
    ax.set_ylabel("count of answers (n=50)")
    ax.set_title(f"RAG template baseline: distribution of specificity (mean = {df['specificity_score'].mean():.2f})")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(CHART_DIR, "rag_specificity_distribution.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

def main():
    chart_sentiment_confusion()
    chart_sentiment_by_source()
    chart_complaints_per_class()
    chart_complaints_failure_breakdown()
    chart_rag_per_intent()
    chart_rag_specificity_dist()
    print(f"\n{len(os.listdir(CHART_DIR))} charts in {CHART_DIR}")

if __name__ == "__main__":
    main()
