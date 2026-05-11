"""Per-source-dataset slice analysis for the Day-1 baseline.

Answers the question a hiring manager asks first: "is your model failing
uniformly or on a specific data slice?"

Slices computed:
  - Sentiment baseline (VADER) sliced by source CSV
  - Complaint baseline sliced by source CSV
  - RAG baseline already has per_intent in baseline_metrics.json

Outputs:
  - results/baseline_slices.json     -- machine-readable slice metrics
  - prints a markdown table to stdout (paste-ready for the report)
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

CATEGORIES = ["service","food_quality","hygiene","price","delivery","portion","ambience","variety"]
SENT_LABELS = ["Positive","Neutral","Negative"]

def slice_sentiment():
    df = pd.read_csv(os.path.join(RESULTS, "baseline_sentiment_preds.csv"))
    out = {}
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        if len(sub) < 5:
            continue
        macro = f1_score(sub["gold"], sub["vader_pred"], labels=SENT_LABELS, average="macro", zero_division=0)
        acc = float((sub["gold"] == sub["vader_pred"]).mean())
        p, r, f, sup = precision_recall_fscore_support(sub["gold"], sub["vader_pred"], labels=SENT_LABELS, zero_division=0)
        out[src] = {
            "n": int(len(sub)),
            "macro_f1": float(macro),
            "accuracy": acc,
            "per_class_f1": {c: float(f[i]) for i, c in enumerate(SENT_LABELS)},
            "support": {c: int(sup[i]) for i, c in enumerate(SENT_LABELS)},
        }
    return out

def slice_complaints():
    df = pd.read_csv(os.path.join(RESULTS, "baseline_complaints_preds.csv"))
    # rebuild gold/pred matrices, then need to know source of each row
    eval_df = pd.read_csv(os.path.join(ROOT, "data", "eval", "complaint_eval.csv"))
    # join on text_preview prefix (text_preview is first 160 chars)
    eval_df["text_key"] = eval_df["text"].str[:160]
    df["text_key"] = df["text_preview"].str[:160]
    merged = df.merge(eval_df[["text_key","source"]], on="text_key", how="left")
    out = {}
    for src in sorted(merged["source"].dropna().unique()):
        sub = merged[merged["source"] == src]
        if len(sub) < 5: continue
        n = len(sub)
        y_true = np.zeros((n, len(CATEGORIES)), dtype=int)
        y_pred = np.zeros((n, len(CATEGORIES)), dtype=int)
        for i, (_, r) in enumerate(sub.reset_index().iterrows()):
            for c in str(r["gold_labels"]).split(","):
                c = c.strip()
                if c in CATEGORIES: y_true[i, CATEGORIES.index(c)] = 1
            for c in str(r["baseline_pred"]).split(","):
                c = c.strip()
                if c in CATEGORIES: y_pred[i, CATEGORIES.index(c)] = 1
        per_class_f1 = {}
        for j, c in enumerate(CATEGORIES):
            if y_true[:, j].sum() == 0:
                per_class_f1[c] = None
                continue
            _, _, f, _ = precision_recall_fscore_support(y_true[:, j], y_pred[:, j], average="binary", zero_division=0)
            per_class_f1[c] = float(f)
        valid_f = [v for v in per_class_f1.values() if v is not None]
        macro = float(np.mean(valid_f)) if valid_f else None
        subset_acc = float(np.mean([(y_true[i] == y_pred[i]).all() for i in range(n)]))
        out[src] = {
            "n": int(n),
            "macro_f1_excl_absent_classes": macro,
            "subset_accuracy": subset_acc,
            "per_class_f1": per_class_f1,
            "n_classes_present": int(sum(1 for v in per_class_f1.values() if v is not None)),
        }
    return out

def slice_rag():
    df = pd.read_csv(os.path.join(RESULTS, "baseline_rag_structural.csv"))
    df = df[~df["answer_failed"].astype(bool)] if "answer_failed" in df else df
    out = {}
    for intent in sorted(df["intent"].unique()):
        sub = df[df["intent"] == intent]
        if len(sub) < 3: continue
        out[intent] = {
            "n": int(len(sub)),
            "sentiment_dir_match": float(sub["sentiment_dir_match"].mean()),
            "top_category_hit": float(sub["top_category_hit"].mean()),
            "rating_mention": float(sub["rating_mention"].mean()),
            "intent_addressed": float(sub["intent_addressed"].mean()),
            "specificity_score": float(sub["specificity_score"].mean()),
        }
    return out

def md_table(title, headers, rows):
    print(f"\n### {title}")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")

def main():
    sent = slice_sentiment()
    comp = slice_complaints()
    rag  = slice_rag()

    out = {"sentiment_by_source": sent, "complaints_by_source": comp, "rag_by_intent": rag}
    with open(os.path.join(RESULTS, "baseline_slices.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {os.path.join(RESULTS, 'baseline_slices.json')}")

    # paste-ready markdown
    md_table(
        "Sentiment baseline (VADER) by source dataset",
        ["source","n","macro-F1","acc","Pos F1","Neu F1","Neg F1"],
        [[s, d["n"], f"{d['macro_f1']:.3f}", f"{d['accuracy']:.3f}",
          f"{d['per_class_f1']['Positive']:.3f}",
          f"{d['per_class_f1']['Neutral']:.3f}",
          f"{d['per_class_f1']['Negative']:.3f}"] for s, d in sent.items()],
    )
    md_table(
        "Complaint baseline by source dataset",
        ["source","n","macro-F1*","subset_acc","classes_present"],
        [[s, d["n"], f"{d['macro_f1_excl_absent_classes']:.3f}" if d["macro_f1_excl_absent_classes"] is not None else "-",
          f"{d['subset_accuracy']:.3f}", d["n_classes_present"]] for s, d in comp.items()],
    )
    md_table(
        "RAG baseline by intent",
        ["intent","n","sent_dir","top_cat","rating_cite","intent_addr","specificity"],
        [[i, d["n"], f"{d['sentiment_dir_match']:.2f}", f"{d['top_category_hit']:.2f}",
          f"{d['rating_mention']:.2f}", f"{d['intent_addressed']:.2f}",
          f"{d['specificity_score']:.2f}"] for i, d in rag.items()],
    )

if __name__ == "__main__":
    main()
