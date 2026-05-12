"""Day-2 Phase 2a runner for RestoAI.

Compares sentiment and complaint-classification strategies head-to-head on the
Day-1 eval sets (200 sentiment / 100 complaint multi-label).

Sentiment strategies
  1. VADER (baseline, reused from results/baseline_metrics.json)
  2. DistilBERT-base-uncased-finetuned-sst-2-english  (HF, binary→3-class via P-thresholds)
  3. NLI zero-shot — valhalla/distilbart-mnli-12-3 with 3 hypothesis classes
  4. Claude Opus 4.6 zero-shot (skipped if ANTHROPIC_API_KEY is unset)

Complaint strategies (8-way multi-label)
  1. categorize_complaints substring scan (baseline, reused)
  2. TF-IDF + LightGBM (one-vs-rest), 5-fold stratified CV on 100 gold reviews
  3. SBERT all-MiniLM-L6-v2 embeddings + LightGBM (one-vs-rest), 5-fold CV
  4. NLI zero-shot multi-label
  5. Claude Opus 4.6 zero-shot (skipped if no key)

Outputs
  results/phase2a_metrics.json      — full metrics for every strategy
  results/phase2a_results.csv       — flat leaderboard for the report
  results/phase2a_sentiment_preds.csv
  results/phase2a_complaints_preds.csv
  results/samples/sentiment_*_wins.csv / _losses.csv
  results/samples/complaints_*_wins.csv / _losses.csv
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import hashlib
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "manager_system"))
EVAL_DIR = os.path.join(ROOT, "data", "eval")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("day02")

# Quiet third-party noise
for noisy in ("transformers", "sentence_transformers", "lightgbm", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

CATEGORIES = ["service", "food_quality", "hygiene", "price", "delivery", "portion", "ambience", "variety"]
SENTI_LABELS = ["Positive", "Neutral", "Negative"]

NLI_MODEL_ID = "valhalla/distilbart-mnli-12-3"
DISTILBERT_SENTI_ID = "distilbert-base-uncased-finetuned-sst-2-english"
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"

POS_THRESHOLD = 0.70
NEG_THRESHOLD = 0.30
MULTILABEL_PROB_THRESHOLD = 0.50

ANTHROPIC_OK = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def safe_text(t) -> str:
    t = "" if t is None else str(t)
    t = t.replace("\r\n", " ").replace("\n", " ").strip()
    # collapse mojibake artifacts that survived encoding round-trips
    t = re.sub(r"[\x80-\x9f]", " ", t)
    return t


def per_class_metrics(y_true, y_pred, labels):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
        for i in range(len(labels))
    }


def multilabel_metrics(y_true_mat, y_pred_mat, labels):
    """y_true_mat / y_pred_mat: (n_samples, n_classes) binary arrays."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support, hamming_loss
    p, r, f, sup = precision_recall_fscore_support(y_true_mat, y_pred_mat, average=None, zero_division=0)
    macro_f1 = f1_score(y_true_mat, y_pred_mat, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_mat, y_pred_mat, average="micro", zero_division=0)
    subset_acc = float(np.all(y_pred_mat == y_true_mat, axis=1).mean())
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "subset_accuracy": subset_acc,
        "exact_match_rate": subset_acc,
        "hamming_loss": float(hamming_loss(y_true_mat, y_pred_mat)),
        "per_class": {
            labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
            for i in range(len(labels))
        },
    }


# ----------------------------------------------------------------------------
# Sentiment runners
# ----------------------------------------------------------------------------

def load_sentiment_eval() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(EVAL_DIR, "sentiment_eval.csv"))
    df["text"] = df["text"].map(safe_text)
    return df


def run_sentiment_vader(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    from analyzer import analyze_text_and_keywords
    preds = []
    t0 = time.time()
    for txt in df["text"]:
        label, _comp, _kws = analyze_text_and_keywords(txt)
        preds.append(label)
    dt = time.time() - t0
    return preds, {
        "engine": "VADER (analyze_text_and_keywords)",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


def run_sentiment_distilbert(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tok = AutoTokenizer.from_pretrained(DISTILBERT_SENTI_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_SENTI_ID)
    mdl.eval()
    id2label = mdl.config.id2label  # {0:'NEGATIVE',1:'POSITIVE'}
    pos_idx = next(i for i, n in id2label.items() if n.upper() == "POSITIVE")

    preds, probs = [], []
    t0 = time.time()
    bs = 8
    with torch.no_grad():
        for start in range(0, len(df), bs):
            batch = df["text"].iloc[start:start + bs].tolist()
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            logits = mdl(**enc).logits
            p_pos = torch.softmax(logits, dim=-1)[:, pos_idx].tolist()
            probs.extend(p_pos)
    for p in probs:
        if p >= POS_THRESHOLD:
            preds.append("Positive")
        elif p <= NEG_THRESHOLD:
            preds.append("Negative")
        else:
            preds.append("Neutral")
    dt = time.time() - t0
    return preds, {
        "engine": f"{DISTILBERT_SENTI_ID} (binary → 3-class via P_pos thresholds {POS_THRESHOLD}/{NEG_THRESHOLD})",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


def run_sentiment_nli(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=NLI_MODEL_ID, device=-1)
    labels = ["positive", "neutral", "negative"]
    label_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    preds = []
    t0 = time.time()
    for txt in df["text"]:
        out = clf(txt[:1500], candidate_labels=labels, multi_label=False,
                  hypothesis_template="The sentiment of this restaurant review is {}.")
        top = out["labels"][0]
        preds.append(label_map[top])
    dt = time.time() - t0
    return preds, {
        "engine": f"NLI zero-shot ({NLI_MODEL_ID})",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


def evaluate_sentiment(df: pd.DataFrame, preds: List[str]) -> Dict:
    from sklearn.metrics import f1_score
    labels = df["gold_label"].tolist()
    macro_f1 = f1_score(labels, preds, average="macro", labels=SENTI_LABELS, zero_division=0)
    acc = float(np.mean([a == b for a, b in zip(labels, preds)]))
    return {
        "n": int(len(df)),
        "macro_f1": float(macro_f1),
        "accuracy": acc,
        "per_class": per_class_metrics(labels, preds, SENTI_LABELS),
    }


# ----------------------------------------------------------------------------
# Complaint runners
# ----------------------------------------------------------------------------

def load_complaint_eval() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(EVAL_DIR, "complaint_eval.csv"))
    df["text"] = df["text"].map(safe_text)
    df["gold_set"] = df["gold_labels"].map(lambda s: [c for c in str(s).split(",") if c])
    return df


def gold_matrix(df: pd.DataFrame) -> np.ndarray:
    mat = np.zeros((len(df), len(CATEGORIES)), dtype=int)
    for i, labs in enumerate(df["gold_set"]):
        for c in labs:
            if c in CATEGORIES:
                mat[i, CATEGORIES.index(c)] = 1
    return mat


def run_complaints_keyword(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    from analyzer import categorize_complaints
    preds = np.zeros((len(df), len(CATEGORIES)), dtype=int)
    t0 = time.time()
    for i, txt in enumerate(df["text"]):
        out = categorize_complaints(txt)  # list[str] of matched categories
        cats = out.keys() if isinstance(out, dict) else out
        for c in cats:
            if c in CATEGORIES:
                preds[i, CATEGORIES.index(c)] = 1
    dt = time.time() - t0
    return preds, {
        "engine": "categorize_complaints (CATEGORY_KEYWORDS substring scan)",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


def _kfold_cv_train_predict(texts: List[str], y_mat: np.ndarray, featurize_fn, *, n_splits: int = 5, seed: int = 0):
    """Stratified-by-primary-label K-fold CV for one-vs-rest LightGBM heads.

    Returns: out-of-fold prediction matrix of same shape as y_mat.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    n = len(texts)
    y_oof = np.zeros_like(y_mat)
    primary = np.array([row.argmax() if row.any() else 0 for row in y_mat])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), primary)):
        X_train_text = [texts[i] for i in train_idx]
        X_test_text = [texts[i] for i in test_idx]
        X_train, X_test = featurize_fn(X_train_text, X_test_text)
        for k in range(len(CATEGORIES)):
            y_train_k = y_mat[train_idx, k]
            if y_train_k.sum() < 2 or y_train_k.sum() > len(y_train_k) - 2:
                # degenerate fold: predict majority
                y_oof[test_idx, k] = int(y_train_k.mean() >= 0.5)
                continue
            params = dict(
                objective="binary", metric="binary_logloss",
                num_leaves=15, learning_rate=0.07, feature_fraction=0.85,
                bagging_fraction=0.85, bagging_freq=4,
                verbose=-1, min_data_in_leaf=3,
                scale_pos_weight=max(1.0, (len(y_train_k) - y_train_k.sum()) / max(y_train_k.sum(), 1)),
                seed=seed,
            )
            dtrain = lgb.Dataset(X_train, label=y_train_k)
            booster = lgb.train(params, dtrain, num_boost_round=120, callbacks=[lgb.log_evaluation(0)])
            p = booster.predict(X_test)
            y_oof[test_idx, k] = (p >= MULTILABEL_PROB_THRESHOLD).astype(int)
    return y_oof


def run_complaints_tfidf_lgbm(df: pd.DataFrame, y_mat: np.ndarray) -> Tuple[np.ndarray, Dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack

    def featurize(train_text, test_text):
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True, max_features=20000)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=0.95, sublinear_tf=True, max_features=20000)
        Xtr = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
        Xte = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
        return Xtr, Xte

    t0 = time.time()
    y_oof = _kfold_cv_train_predict(df["text"].tolist(), y_mat, featurize_fn=featurize, n_splits=5, seed=0)
    dt = time.time() - t0
    return y_oof, {
        "engine": "TF-IDF (word 1-2gram + char 3-5gram) + LightGBM OvR, 5-fold CV (out-of-fold predictions)",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


def run_complaints_sbert_lgbm(df: pd.DataFrame, y_mat: np.ndarray) -> Tuple[np.ndarray, Dict]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(SBERT_ID)
    all_emb = encoder.encode(df["text"].tolist(), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    all_emb = np.asarray(all_emb, dtype=np.float32)

    def featurize(train_text, test_text):
        # map back from text → row index in df
        text_to_idx = {t: i for i, t in enumerate(df["text"].tolist())}
        Xtr = all_emb[[text_to_idx[t] for t in train_text]]
        Xte = all_emb[[text_to_idx[t] for t in test_text]]
        return Xtr, Xte

    t0 = time.time()
    y_oof = _kfold_cv_train_predict(df["text"].tolist(), y_mat, featurize_fn=featurize, n_splits=5, seed=0)
    dt = time.time() - t0
    return y_oof, {
        "engine": f"SBERT ({SBERT_ID}) 384d + LightGBM OvR, 5-fold CV",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


def run_complaints_nli(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=NLI_MODEL_ID, device=-1)
    # Friendly natural-language category labels for the NLI hypothesis
    label_names = {
        "service": "rude or inattentive service",
        "food_quality": "bad food taste or quality",
        "hygiene": "dirty, unhygienic conditions",
        "price": "overpriced or expensive",
        "delivery": "slow service or late delivery",
        "portion": "small portion size or quantity",
        "ambience": "unpleasant ambience or atmosphere",
        "variety": "limited menu variety or options",
    }
    candidates = [label_names[c] for c in CATEGORIES]
    preds = np.zeros((len(df), len(CATEGORIES)), dtype=int)
    t0 = time.time()
    for i, txt in enumerate(df["text"]):
        out = clf(txt[:1500], candidate_labels=candidates, multi_label=True,
                  hypothesis_template="This review mentions {}.")
        # `out["labels"]` and `out["scores"]` are aligned and sorted by score desc
        score_for_label = dict(zip(out["labels"], out["scores"]))
        for k, c in enumerate(CATEGORIES):
            if score_for_label.get(label_names[c], 0.0) >= MULTILABEL_PROB_THRESHOLD:
                preds[i, k] = 1
    dt = time.time() - t0
    return preds, {
        "engine": f"NLI zero-shot multi-label ({NLI_MODEL_ID}); threshold={MULTILABEL_PROB_THRESHOLD}",
        "latency_ms_per_review": 1000 * dt / max(len(df), 1),
        "wall_seconds": dt,
    }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def write_samples(name: str, df: pd.DataFrame, preds, gold, kind: str):
    """Save 5 wins / 5 losses per strategy."""
    if kind == "sentiment":
        rows = []
        for i, (p, g) in enumerate(zip(preds, gold)):
            rows.append({
                "text_preview": df["text"].iloc[i][:160],
                "gold": g,
                "pred": p,
                "correct": p == g,
                "rating": df["rating"].iloc[i],
                "source": df["source"].iloc[i],
            })
        out = pd.DataFrame(rows)
        out[out["correct"]].head(5).to_csv(os.path.join(SAMPLES_DIR, f"day02_sentiment_{name}_wins.csv"), index=False)
        out[~out["correct"]].head(5).to_csv(os.path.join(SAMPLES_DIR, f"day02_sentiment_{name}_losses.csv"), index=False)
    elif kind == "complaints":
        rows = []
        for i in range(len(df)):
            p_set = sorted([CATEGORIES[k] for k in range(len(CATEGORIES)) if preds[i, k] == 1])
            g_set = sorted([CATEGORIES[k] for k in range(len(CATEGORIES)) if gold[i, k] == 1])
            rows.append({
                "text_preview": df["text"].iloc[i][:160],
                "gold": ",".join(g_set),
                "pred": ",".join(p_set),
                "exact_match": p_set == g_set,
                "source": df["source"].iloc[i],
            })
        out = pd.DataFrame(rows)
        out[out["exact_match"]].head(5).to_csv(os.path.join(SAMPLES_DIR, f"day02_complaints_{name}_wins.csv"), index=False)
        out[~out["exact_match"]].head(5).to_csv(os.path.join(SAMPLES_DIR, f"day02_complaints_{name}_losses.csv"), index=False)


def main():
    started = time.time()
    print(f"== Day 02 Phase 2a — RestoAI ==  (Anthropic available: {ANTHROPIC_OK})", flush=True)

    # SENTIMENT --------------------------------------------------------------
    senti_df = load_sentiment_eval()
    print(f"[sentiment] eval n={len(senti_df)}", flush=True)
    senti_results = {}
    all_senti_preds = {}

    print("[sentiment] VADER ...", flush=True)
    p_vader, meta_v = run_sentiment_vader(senti_df)
    senti_results["vader"] = {**meta_v, **evaluate_sentiment(senti_df, p_vader)}
    all_senti_preds["vader"] = p_vader
    write_samples("vader", senti_df, p_vader, senti_df["gold_label"].tolist(), "sentiment")

    print("[sentiment] DistilBERT SST-2 ...", flush=True)
    p_db, meta_db = run_sentiment_distilbert(senti_df)
    senti_results["distilbert_sst2"] = {**meta_db, **evaluate_sentiment(senti_df, p_db)}
    all_senti_preds["distilbert_sst2"] = p_db
    write_samples("distilbert_sst2", senti_df, p_db, senti_df["gold_label"].tolist(), "sentiment")

    print("[sentiment] NLI zero-shot ...", flush=True)
    p_nli, meta_nli = run_sentiment_nli(senti_df)
    senti_results["nli_zeroshot"] = {**meta_nli, **evaluate_sentiment(senti_df, p_nli)}
    all_senti_preds["nli_zeroshot"] = p_nli
    write_samples("nli_zeroshot", senti_df, p_nli, senti_df["gold_label"].tolist(), "sentiment")

    if ANTHROPIC_OK:
        try:
            from scripts.day02_claude_zeroshot import run_sentiment_claude  # type: ignore
            p_cl, meta_cl = run_sentiment_claude(senti_df)
            senti_results["claude_zeroshot"] = {**meta_cl, **evaluate_sentiment(senti_df, p_cl)}
            all_senti_preds["claude_zeroshot"] = p_cl
            write_samples("claude_zeroshot", senti_df, p_cl, senti_df["gold_label"].tolist(), "sentiment")
        except Exception as e:
            log.warning("Claude sentiment skipped: %s", e)
            senti_results["claude_zeroshot"] = {"skipped": True, "reason": str(e)}
    else:
        senti_results["claude_zeroshot"] = {"skipped": True, "reason": "ANTHROPIC_API_KEY not set in autonomous run"}

    # COMPLAINTS -------------------------------------------------------------
    comp_df = load_complaint_eval()
    y_gold = gold_matrix(comp_df)
    print(f"[complaints] eval n={len(comp_df)}  positives per class={dict(zip(CATEGORIES, y_gold.sum(0).tolist()))}", flush=True)
    comp_results = {}
    all_comp_preds = {}

    print("[complaints] keyword baseline ...", flush=True)
    p_kw, meta_kw = run_complaints_keyword(comp_df)
    comp_results["keyword"] = {**meta_kw, **multilabel_metrics(y_gold, p_kw, CATEGORIES)}
    all_comp_preds["keyword"] = p_kw
    write_samples("keyword", comp_df, p_kw, y_gold, "complaints")

    print("[complaints] TF-IDF + LightGBM (CV) ...", flush=True)
    p_tfidf, meta_tfidf = run_complaints_tfidf_lgbm(comp_df, y_gold)
    comp_results["tfidf_lgbm"] = {**meta_tfidf, **multilabel_metrics(y_gold, p_tfidf, CATEGORIES)}
    all_comp_preds["tfidf_lgbm"] = p_tfidf
    write_samples("tfidf_lgbm", comp_df, p_tfidf, y_gold, "complaints")

    print("[complaints] SBERT + LightGBM (CV) ...", flush=True)
    p_sbert, meta_sbert = run_complaints_sbert_lgbm(comp_df, y_gold)
    comp_results["sbert_lgbm"] = {**meta_sbert, **multilabel_metrics(y_gold, p_sbert, CATEGORIES)}
    all_comp_preds["sbert_lgbm"] = p_sbert
    write_samples("sbert_lgbm", comp_df, p_sbert, y_gold, "complaints")

    print("[complaints] NLI zero-shot ...", flush=True)
    p_nli_c, meta_nli_c = run_complaints_nli(comp_df)
    comp_results["nli_zeroshot"] = {**meta_nli_c, **multilabel_metrics(y_gold, p_nli_c, CATEGORIES)}
    all_comp_preds["nli_zeroshot"] = p_nli_c
    write_samples("nli_zeroshot", comp_df, p_nli_c, y_gold, "complaints")

    if ANTHROPIC_OK:
        try:
            from scripts.day02_claude_zeroshot import run_complaints_claude  # type: ignore
            p_clc, meta_clc = run_complaints_claude(comp_df)
            comp_results["claude_zeroshot"] = {**meta_clc, **multilabel_metrics(y_gold, p_clc, CATEGORIES)}
            all_comp_preds["claude_zeroshot"] = p_clc
            write_samples("claude_zeroshot", comp_df, p_clc, y_gold, "complaints")
        except Exception as e:
            log.warning("Claude complaints skipped: %s", e)
            comp_results["claude_zeroshot"] = {"skipped": True, "reason": str(e)}
    else:
        comp_results["claude_zeroshot"] = {"skipped": True, "reason": "ANTHROPIC_API_KEY not set in autonomous run"}

    # Per-prediction CSVs
    senti_rows = []
    for i in range(len(senti_df)):
        row = {
            "idx": i,
            "text_preview": senti_df["text"].iloc[i][:160],
            "rating": senti_df["rating"].iloc[i],
            "source": senti_df["source"].iloc[i],
            "gold": senti_df["gold_label"].iloc[i],
        }
        for name, preds in all_senti_preds.items():
            row[f"pred_{name}"] = preds[i]
        senti_rows.append(row)
    pd.DataFrame(senti_rows).to_csv(os.path.join(RESULTS_DIR, "phase2a_sentiment_preds.csv"), index=False)

    comp_rows = []
    for i in range(len(comp_df)):
        gold_set = sorted([CATEGORIES[k] for k in range(len(CATEGORIES)) if y_gold[i, k] == 1])
        row = {
            "idx": i,
            "text_preview": comp_df["text"].iloc[i][:160],
            "source": comp_df["source"].iloc[i],
            "gold": ",".join(gold_set),
        }
        for name, preds in all_comp_preds.items():
            ps = sorted([CATEGORIES[k] for k in range(len(CATEGORIES)) if preds[i, k] == 1])
            row[f"pred_{name}"] = ",".join(ps)
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(os.path.join(RESULTS_DIR, "phase2a_complaints_preds.csv"), index=False)

    # Leaderboard CSV
    lb_rows = []
    for k, v in senti_results.items():
        if v.get("skipped"):
            lb_rows.append({"component": "sentiment", "strategy": k, "macro_f1": None, "accuracy_or_micro": None,
                            "subset_acc": None, "latency_ms": None, "wall_s": None, "skipped": True, "notes": v.get("reason", "")})
        else:
            lb_rows.append({"component": "sentiment", "strategy": k,
                            "macro_f1": round(v["macro_f1"], 4),
                            "accuracy_or_micro": round(v["accuracy"], 4),
                            "subset_acc": None,
                            "latency_ms": round(v["latency_ms_per_review"], 1),
                            "wall_s": round(v["wall_seconds"], 1),
                            "skipped": False,
                            "notes": v["engine"]})
    for k, v in comp_results.items():
        if v.get("skipped"):
            lb_rows.append({"component": "complaints", "strategy": k, "macro_f1": None, "accuracy_or_micro": None,
                            "subset_acc": None, "latency_ms": None, "wall_s": None, "skipped": True, "notes": v.get("reason", "")})
        else:
            lb_rows.append({"component": "complaints", "strategy": k,
                            "macro_f1": round(v["macro_f1"], 4),
                            "accuracy_or_micro": round(v["micro_f1"], 4),
                            "subset_acc": round(v["subset_accuracy"], 4),
                            "latency_ms": round(v["latency_ms_per_review"], 1),
                            "wall_s": round(v["wall_seconds"], 1),
                            "skipped": False,
                            "notes": v["engine"]})
    pd.DataFrame(lb_rows).to_csv(os.path.join(RESULTS_DIR, "phase2a_results.csv"), index=False)

    # Full metrics dump
    out = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "anthropic_available": ANTHROPIC_OK,
        "wall_seconds_total": time.time() - started,
        "sentiment": senti_results,
        "complaints": comp_results,
    }
    with open(os.path.join(RESULTS_DIR, "phase2a_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n== DONE in {out['wall_seconds_total']:.1f}s ==")
    print("Sentiment macro-F1:")
    for k, v in senti_results.items():
        if v.get("skipped"):
            print(f"  - {k:20s}  SKIPPED ({v['reason']})")
        else:
            print(f"  - {k:20s}  {v['macro_f1']:.4f}  (acc {v['accuracy']:.3f}, {v['latency_ms_per_review']:.1f} ms/r)")
    print("Complaint macro-F1:")
    for k, v in comp_results.items():
        if v.get("skipped"):
            print(f"  - {k:20s}  SKIPPED ({v['reason']})")
        else:
            print(f"  - {k:20s}  {v['macro_f1']:.4f}  (micro {v['micro_f1']:.3f}, subset {v['subset_accuracy']:.3f}, {v['latency_ms_per_review']:.1f} ms/r)")


if __name__ == "__main__":
    main()
