"""
Day-1 polish addendum #2 — bootstrap 95% confidence intervals for all baseline metrics.

Phase-2 strategies (Day 2+) compare alternative models against the Day-1 baseline.
A claim like "TF-IDF+LightGBM beat the keyword scan by +0.05 macro-F1" is only
meaningful if +0.05 exceeds the noise floor of the baseline measurement itself.
This script computes 1000-sample non-parametric bootstrap CIs for:

  Sentiment  (n=200)  — macro-F1, accuracy, per-class F1 (Pos/Neu/Neg)
  Complaints (n=100)  — macro-F1, micro-F1, subset accuracy, hamming loss,
                        per-class F1 (8 categories)
  RAG        (n=50)   — composite (geo-mean of 4 binary rates), each rate,
                        and the specificity mean.

Reads the per-row prediction CSVs already produced by `run_baselines.py` and
`structural_rag_metrics.py`. Writes results/baseline_ci.json and a comparison
chart at results/charts/baseline_ci.png.
"""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CHARTS = os.path.join(RESULTS, "charts")
os.makedirs(CHARTS, exist_ok=True)

N_BOOT = 1000
SEED = 20260511
CI_LO, CI_HI = 2.5, 97.5

SENT_CLASSES = ["Positive", "Neutral", "Negative"]
COMPLAINT_CLASSES = [
    "service",
    "food_quality",
    "hygiene",
    "price",
    "delivery",
    "portion",
    "ambience",
    "variety",
]


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def sentiment_metrics(gold, pred):
    """Per-class F1 (one-vs-rest) and macro / accuracy."""
    per = {}
    for cls in SENT_CLASSES:
        tp = int(((pred == cls) & (gold == cls)).sum())
        fp = int(((pred == cls) & (gold != cls)).sum())
        fn = int(((pred != cls) & (gold == cls)).sum())
        per[cls] = _f1(tp, fp, fn)
    return {
        "macro_f1": float(np.mean(list(per.values()))),
        "accuracy": float((gold == pred).mean()),
        **{f"f1_{c}": per[c] for c in SENT_CLASSES},
    }


def _parse_labels(cell):
    if not isinstance(cell, str) or not cell.strip():
        return set()
    return {x.strip() for x in cell.split(",") if x.strip()}


def complaint_metrics(gold_lists, pred_lists):
    per = {}
    tp_total = fp_total = fn_total = 0
    hamming_errs = 0
    cells = 0
    exact = 0
    for cls in COMPLAINT_CLASSES:
        tp = fp = fn = 0
        for g, p in zip(gold_lists, pred_lists):
            in_g = cls in g
            in_p = cls in p
            if in_g and in_p:
                tp += 1
            elif in_p and not in_g:
                fp += 1
            elif in_g and not in_p:
                fn += 1
        per[cls] = _f1(tp, fp, fn)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    for g, p in zip(gold_lists, pred_lists):
        if g == p:
            exact += 1
        for cls in COMPLAINT_CLASSES:
            cells += 1
            if (cls in g) != (cls in p):
                hamming_errs += 1
    return {
        "macro_f1": float(np.mean(list(per.values()))),
        "micro_f1": _f1(tp_total, fp_total, fn_total),
        "subset_accuracy": exact / len(gold_lists),
        "hamming_loss": hamming_errs / cells,
        **{f"f1_{c}": per[c] for c in COMPLAINT_CLASSES},
    }


def rag_metrics(df):
    bin_rates = {
        c: float(df[c].mean())
        for c in [
            "sentiment_dir_match",
            "top_category_hit",
            "rating_mention",
            "intent_addressed",
        ]
    }
    spec_mean = float(df["specificity_score"].mean())
    # geometric mean of the 4 binary rates (matches baseline_metrics.json)
    vals = list(bin_rates.values())
    composite = (
        float(np.exp(np.mean(np.log(np.clip(vals, 1e-9, None))))) if all(v > 0 for v in vals) else 0.0
    )
    return {"composite": composite, "specificity_mean": spec_mean, **bin_rates}


def bootstrap(values_fn, n, n_boot=N_BOOT, seed=SEED):
    """Run n_boot resamples of indices into [0, n) and call values_fn(idx) → dict of scalars."""
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = values_fn(idx)
        for k, v in m.items():
            samples[k].append(v)
    out = {}
    for k, vs in samples.items():
        arr = np.asarray(vs)
        out[k] = {
            "lo": float(np.percentile(arr, CI_LO)),
            "hi": float(np.percentile(arr, CI_HI)),
            "mean_boot": float(arr.mean()),
            "std_boot": float(arr.std(ddof=1)),
        }
    return out


def main():
    # ---------------- Sentiment ----------------
    s = pd.read_csv(os.path.join(RESULTS, "baseline_sentiment_preds.csv"))
    gold = s["gold"].to_numpy()
    pred = s["vader_pred"].to_numpy()
    point_sent = sentiment_metrics(gold, pred)

    def sent_fn(idx):
        return sentiment_metrics(gold[idx], pred[idx])

    ci_sent = bootstrap(sent_fn, len(s))

    # ---------------- Complaints ----------------
    c = pd.read_csv(os.path.join(RESULTS, "baseline_complaints_preds.csv"))
    gold_lists = [_parse_labels(x) for x in c["gold_labels"].tolist()]
    pred_lists = [_parse_labels(x) for x in c["baseline_pred"].tolist()]
    point_cmp = complaint_metrics(gold_lists, pred_lists)

    def cmp_fn(idx):
        g = [gold_lists[i] for i in idx]
        p = [pred_lists[i] for i in idx]
        return complaint_metrics(g, p)

    ci_cmp = bootstrap(cmp_fn, len(c))

    # ---------------- RAG ----------------
    r = pd.read_csv(os.path.join(RESULTS, "baseline_rag_structural.csv"))
    point_rag = rag_metrics(r)

    def rag_fn(idx):
        return rag_metrics(r.iloc[idx].reset_index(drop=True))

    ci_rag = bootstrap(rag_fn, len(r))

    out = {
        "method": {
            "n_bootstrap": N_BOOT,
            "ci_pct": [CI_LO, CI_HI],
            "seed": SEED,
            "resample": "non-parametric, with replacement, row-level",
        },
        "sentiment": {
            "n": int(len(s)),
            "point": point_sent,
            "ci": ci_sent,
        },
        "complaints": {
            "n": int(len(c)),
            "point": point_cmp,
            "ci": ci_cmp,
        },
        "rag": {
            "n": int(len(r)),
            "point": point_rag,
            "ci": ci_rag,
        },
    }
    out_path = os.path.join(RESULTS, "baseline_ci.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")

    # ---------------- Chart ----------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline = [
        ("Sentiment\nmacro-F1", point_sent["macro_f1"], ci_sent["macro_f1"]),
        ("Sentiment\nNeutral F1", point_sent["f1_Neutral"], ci_sent["f1_Neutral"]),
        ("Complaints\nmacro-F1", point_cmp["macro_f1"], ci_cmp["macro_f1"]),
        ("Complaints\nsubset acc", point_cmp["subset_accuracy"], ci_cmp["subset_accuracy"]),
        ("Complaints\ndelivery F1", point_cmp["f1_delivery"], ci_cmp["f1_delivery"]),
        ("RAG\ncomposite", point_rag["composite"], ci_rag["composite"]),
        ("RAG\nrating_mention", point_rag["rating_mention"], ci_rag["rating_mention"]),
        ("RAG\nsentiment_dir", point_rag["sentiment_dir_match"], ci_rag["sentiment_dir_match"]),
    ]
    labels = [x[0] for x in headline]
    points = [x[1] for x in headline]
    los = [x[1] - x[2]["lo"] for x in headline]
    his = [x[2]["hi"] - x[1] for x in headline]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(headline))
    bars = ax.bar(xs, points, color="#4878d0", alpha=0.85)
    ax.errorbar(xs, points, yerr=[los, his], fmt="none", ecolor="black", capsize=4, lw=1.2)
    for x, p in zip(xs, points):
        ax.text(x, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Day-1 baselines with 95% bootstrap CI (n_boot={N_BOOT}, seed={SEED})\n"
        "Phase-2 wins must clear these intervals to be credible."
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    chart_path = os.path.join(CHARTS, "baseline_ci.png")
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)
    print(f"wrote {chart_path}")

    # Compact summary table to stdout for the report writer
    print("\n=== Headline metrics with 95% CI ===")
    for lab, pt, ci in headline:
        print(
            f"{lab.replace(chr(10),' '):28s}  point={pt:.3f}  "
            f"CI=[{ci['lo']:.3f}, {ci['hi']:.3f}]  width={ci['hi']-ci['lo']:.3f}"
        )


if __name__ == "__main__":
    main()
