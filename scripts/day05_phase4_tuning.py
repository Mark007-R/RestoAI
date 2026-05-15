"""Day-5 Phase-4 tuning + error analysis for the complaint classifier.

Pipeline:
  1. Optuna sweep (>=30 trials) over LightGBM hparams, evaluated via 5-fold
     stratified CV macro-F1 on the Day-1 100-review gold set. Same featurization
     as Day-2 (TF-IDF word 1-2 + char_wb 3-5).
  2. Re-train the best trial with `return_probs=True` and persist OOF
     probabilities (Day-2 binarized too early - see Day-4 report next-day note).
  3. Optimize per-class probability thresholds on the OOF probs (grid over
     [0.10, 0.90] in 0.05 steps) to maximize macro-F1. This is the "BCE with
     per-class threshold" lift that does NOT require swapping loss.
  4. Error analysis: pick 30 worst rows by per-row F1, tag each row as
        - multi_category_overlap  : gold has >=3 labels and model got >=1
        - label_noise             : gold has a label that the keyword
                                    pattern doesn't fire AND the trained
                                    head's prob is very low (suggests gold may
                                    disagree with the text)
        - model_failure           : a keyword pattern fires for a missed gold
                                    label but the trained head still scores
                                    low (model couldn't learn the obvious
                                    signal)
  5. Targeted fix: if multi_category_overlap dominates (>=40% of the 30
     failures), fit a multi-output BCE head (logistic-regression OvR on the
     same TF-IDF features but with class-balanced sample weights and
     per-class threshold optimization) and re-evaluate. Otherwise log the
     dominant failure type and skip the BCE rerun.

Outputs (all under results/):
  day05_optuna_trials.csv        - per-trial params + macro-F1 + per-class F1
  day05_oof_probs.csv            - 100 x 8 OOF probabilities (champion trial)
  day05_thresholds.json          - per-class threshold + per-class F1 at it
  day05_error_analysis.csv       - top-30 failures with tags + per-row diags
  day05_failure_breakdown.json   - counts by failure type
  day05_bce_comparison.csv       - (only if multi-cat-overlap dominates) BCE
                                   vs LGBM-tuned head-to-head
  day05_metrics.json             - master summary

This script wraps all expensive work in the `if __name__ == '__main__'` block
so the report can re-import helpers without re-running.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(ROOT, "data", "eval", "complaint_eval.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORIES = ["service", "food_quality", "hygiene", "price", "delivery",
              "portion", "ambience", "variety"]
SEED = 0
N_SPLITS = 5
N_TRIALS = 30

# Day-2 baseline (carried in Day-4 bundle.meta.macro_f1_5fold_cv_day2)
DAY2_BASELINE_MACRO_F1 = 0.682


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_gold() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(EVAL_PATH).dropna(subset=["text"]).reset_index(drop=True)
    y = np.zeros((len(df), len(CATEGORIES)), dtype=int)
    for i, raw in enumerate(df["gold_labels"].fillna("").astype(str)):
        for c in [c.strip() for c in raw.split(",") if c.strip()]:
            if c in CATEGORIES:
                y[i, CATEGORIES.index(c)] = 1
    return df, y


def keyword_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns 0/1 (n, 8) — what CATEGORY_KEYWORDS substring scan would fire."""
    sys.path.insert(0, os.path.join(ROOT, "manager_system"))
    from analyzer import CATEGORY_KEYWORDS
    mat = np.zeros((len(df), len(CATEGORIES)), dtype=int)
    for i, txt in enumerate(df["text"].astype(str)):
        tl = txt.lower()
        for j, cat in enumerate(CATEGORIES):
            for kw in CATEGORY_KEYWORDS[cat]:
                if kw in tl:
                    mat[i, j] = 1
                    break
    return mat


# -----------------------------------------------------------------------------
# CV runner used both for Optuna trials AND for the champion OOF probability dump
# -----------------------------------------------------------------------------

def _featurize(train_text: List[str], test_text: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95,
                           sublinear_tf=True, max_features=20000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                           max_df=0.95, sublinear_tf=True, max_features=20000)
    Xtr = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    Xte = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return Xtr, Xte


def kfold_oof_probs(texts: List[str], y_mat: np.ndarray, params: Dict,
                   num_boost_round: int, scale_pos_weight_mult: float,
                   n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    """Stratified-by-primary-label K-fold CV → (n, k) OOF probability matrix."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    n, k = y_mat.shape
    probs = np.zeros((n, k), dtype=np.float32)
    primary = np.array([row.argmax() if row.any() else 0 for row in y_mat])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(np.zeros(n), primary):
        X_train_text = [texts[i] for i in train_idx]
        X_test_text = [texts[i] for i in test_idx]
        X_train, X_test = _featurize(X_train_text, X_test_text)
        for j in range(k):
            y_train_j = y_mat[train_idx, j]
            n_pos = int(y_train_j.sum())
            n_neg = len(y_train_j) - n_pos
            if n_pos < 2 or n_neg < 2:
                probs[test_idx, j] = float(y_train_j.mean())
                continue
            sw = max(1.0, n_neg / max(n_pos, 1)) * scale_pos_weight_mult
            p = dict(params)
            p["objective"] = "binary"
            p["metric"] = "binary_logloss"
            p["verbose"] = -1
            p["scale_pos_weight"] = sw
            p["seed"] = seed
            dtrain = lgb.Dataset(X_train, label=y_train_j)
            booster = lgb.train(p, dtrain, num_boost_round=num_boost_round,
                                callbacks=[lgb.log_evaluation(0)])
            probs[test_idx, j] = booster.predict(X_test)
    return probs


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for j in range(y_true.shape[1]):
        tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
        fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
        fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append((2 * p * r / (p + r)) if (p + r) else 0.0)
    return float(np.mean(f1s))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for j, cat in enumerate(CATEGORIES):
        tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
        fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
        fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out[cat] = (2 * p * r / (p + r)) if (p + r) else 0.0
    return out


# -----------------------------------------------------------------------------
# Optuna sweep
# -----------------------------------------------------------------------------

def run_optuna(texts: List[str], y_mat: np.ndarray, n_trials: int = N_TRIALS) -> Tuple[Dict, pd.DataFrame]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trial_rows = []

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 8),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 5.0, log=True),
        }
        nbr = trial.suggest_int("num_boost_round", 50, 300)
        spw_mult = trial.suggest_float("scale_pos_weight_mult", 0.5, 2.0)

        probs = kfold_oof_probs(texts, y_mat, params,
                                num_boost_round=nbr,
                                scale_pos_weight_mult=spw_mult)
        pred = (probs >= 0.5).astype(int)
        m = macro_f1(y_mat, pred)
        pcf = per_class_f1(y_mat, pred)
        row = {"trial": trial.number, "macro_f1_t05": m, **{f"f1_{c}": pcf[c] for c in CATEGORIES},
               "num_boost_round": nbr, "scale_pos_weight_mult": spw_mult, **params}
        trial_rows.append(row)
        return m

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    dt = time.time() - t0

    best = study.best_trial
    print(f"[day05-optuna] {n_trials} trials in {dt:.1f}s | best macro-F1 (thr=0.5) = {best.value:.3f}")

    df_trials = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
    best_params = dict(best.params)
    best_params["macro_f1_t05"] = float(best.value)
    best_params["wall_seconds"] = dt
    return best_params, df_trials


# -----------------------------------------------------------------------------
# Per-class threshold optimization
# -----------------------------------------------------------------------------

def optimize_per_class_thresholds(y_true: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Grid-search threshold per class in [0.10, 0.90] step 0.05; maximize per-class F1."""
    grid = np.arange(0.10, 0.91, 0.05)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=float)
    per_class_f1_at_thr = {}
    for j, cat in enumerate(CATEGORIES):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred_j = (probs[:, j] >= t).astype(int)
            tp = int(((pred_j == 1) & (y_true[:, j] == 1)).sum())
            fp = int(((pred_j == 1) & (y_true[:, j] == 0)).sum())
            fn = int(((pred_j == 0) & (y_true[:, j] == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[j] = best_t
        per_class_f1_at_thr[cat] = best_f1
    return thresholds, per_class_f1_at_thr


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(probs, dtype=int)
    for j in range(probs.shape[1]):
        pred[:, j] = (probs[:, j] >= thresholds[j]).astype(int)
    return pred


# -----------------------------------------------------------------------------
# Error analysis
# -----------------------------------------------------------------------------

def row_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def tag_failure(yt_row: np.ndarray, yp_row: np.ndarray, probs_row: np.ndarray,
                kw_row: np.ndarray) -> str:
    """Assign one of: multi_category_overlap, label_noise, model_failure.

    Operational definitions:
      - multi_category_overlap : gold has >=3 labels AND model got >=1 correct AND missed at least one
      - label_noise            : there exists a missed gold label j where keyword pattern
                                 does NOT fire AND the model's prob_j < 0.20 (textual signal
                                 absent for that gold label - likely annotation disagreement)
      - model_failure          : there exists a missed gold label j where keyword pattern
                                 DOES fire but model prob_j < 0.40 (obvious signal, model missed)
    Precedence: multi_category_overlap > model_failure > label_noise.
    """
    gold = set(np.where(yt_row == 1)[0].tolist())
    pred = set(np.where(yp_row == 1)[0].tolist())
    missed = gold - pred

    if len(gold) >= 3 and len(gold & pred) >= 1 and missed:
        return "multi_category_overlap"

    for j in missed:
        if kw_row[j] == 1 and probs_row[j] < 0.40:
            return "model_failure"

    for j in missed:
        if kw_row[j] == 0 and probs_row[j] < 0.20:
            return "label_noise"

    # Fallback: false positives only, or borderline misses
    if pred - gold and not missed:
        return "model_failure"  # false-positive-only -> model failed
    return "model_failure"


def error_analysis(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                  probs: np.ndarray, kw: np.ndarray, top_n: int = 30) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows = []
    for i in range(len(df)):
        f1_i = row_f1(y_true[i], y_pred[i])
        gold_set = [CATEGORIES[j] for j in np.where(y_true[i] == 1)[0]]
        pred_set = [CATEGORIES[j] for j in np.where(y_pred[i] == 1)[0]]
        missed = sorted(set(gold_set) - set(pred_set))
        extra = sorted(set(pred_set) - set(gold_set))
        rows.append({
            "idx": i,
            "text": df["text"].iloc[i][:300],
            "rating": df["rating"].iloc[i],
            "gold": ",".join(gold_set),
            "pred": ",".join(pred_set),
            "missed": ",".join(missed),
            "extra": ",".join(extra),
            "row_f1": f1_i,
            "n_gold": int(y_true[i].sum()),
            "n_pred": int(y_pred[i].sum()),
            "tag": tag_failure(y_true[i], y_pred[i], probs[i], kw[i]),
        })
    err_df = pd.DataFrame(rows)
    # Only consider rows that have any error (row_f1 < 1.0)
    err_df = err_df[err_df["row_f1"] < 1.0].sort_values(
        ["row_f1", "n_gold"], ascending=[True, False]
    ).reset_index(drop=True)
    top = err_df.head(top_n).copy()
    breakdown = top["tag"].value_counts().to_dict()
    return top, breakdown


# -----------------------------------------------------------------------------
# BCE multi-label rerun (conditional)
# -----------------------------------------------------------------------------

def run_bce_multilabel(texts: List[str], y_mat: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """sklearn LogisticRegression in OvR mode IS per-class binary cross-entropy
    on shared TF-IDF features. The differentiator vs the LGBM tuned head:
      - linear decision boundary (no tree partitioning); shrinkage via L2
      - calibrated probabilities from sigmoid output
      - per-class threshold optimized after the fact
    This is the "BCE with per-class threshold" multi-label fix.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    n, k = y_mat.shape
    probs = np.zeros((n, k), dtype=np.float32)
    primary = np.array([row.argmax() if row.any() else 0 for row in y_mat])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    t0 = time.time()
    for train_idx, test_idx in skf.split(np.zeros(n), primary):
        X_train_text = [texts[i] for i in train_idx]
        X_test_text = [texts[i] for i in test_idx]
        X_train, X_test = _featurize(X_train_text, X_test_text)
        for j in range(k):
            y_train_j = y_mat[train_idx, j]
            n_pos = int(y_train_j.sum())
            n_neg = len(y_train_j) - n_pos
            if n_pos < 2 or n_neg < 2:
                probs[test_idx, j] = float(y_train_j.mean())
                continue
            clf = LogisticRegression(
                C=0.5, max_iter=2000, solver="liblinear",
                class_weight="balanced", random_state=SEED,
            )
            clf.fit(X_train, y_train_j)
            probs[test_idx, j] = clf.predict_proba(X_test)[:, 1]
    dt = time.time() - t0
    return probs, {"engine": "LogReg OvR (BCE) + per-class threshold",
                   "wall_seconds": dt}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print(f"[day05] loading gold set from {EVAL_PATH}")
    df, y = load_gold()
    texts = df["text"].astype(str).tolist()
    n, k = y.shape
    print(f"[day05] gold: n={n}, k={k}")
    kw = keyword_matrix(df)

    # ----- 1. Optuna sweep -----
    print(f"[day05] running Optuna with {N_TRIALS} trials...")
    best, df_trials = run_optuna(texts, y, n_trials=N_TRIALS)
    df_trials.to_csv(os.path.join(RESULTS_DIR, "day05_optuna_trials.csv"), index=False)
    print(f"[day05] best params: {best}")

    # ----- 2. Re-run best trial, persist OOF probs -----
    best_params = {kk: best[kk] for kk in ("num_leaves", "learning_rate", "min_data_in_leaf",
                                            "feature_fraction", "bagging_fraction",
                                            "bagging_freq", "lambda_l2")}
    print("[day05] re-running champion trial to persist OOF probabilities...")
    champion_probs = kfold_oof_probs(texts, y, best_params,
                                     num_boost_round=int(best["num_boost_round"]),
                                     scale_pos_weight_mult=float(best["scale_pos_weight_mult"]))
    df_probs = pd.DataFrame(champion_probs, columns=[f"p_{c}" for c in CATEGORIES])
    df_probs.insert(0, "idx", range(n))
    df_probs.insert(1, "gold", df["gold_labels"].fillna("").astype(str))
    df_probs.to_csv(os.path.join(RESULTS_DIR, "day05_oof_probs.csv"), index=False)
    pred_t05 = (champion_probs >= 0.5).astype(int)
    macro_t05 = macro_f1(y, pred_t05)
    pcf_t05 = per_class_f1(y, pred_t05)
    print(f"[day05] champion macro-F1 (thr=0.5) = {macro_t05:.3f}  (Day-2 baseline: {DAY2_BASELINE_MACRO_F1:.3f})")

    # ----- 3. Per-class thresholds -----
    thresholds, pcf_thr = optimize_per_class_thresholds(y, champion_probs)
    pred_thr = apply_thresholds(champion_probs, thresholds)
    macro_thr = macro_f1(y, pred_thr)
    print(f"[day05] champion macro-F1 (per-class thr) = {macro_thr:.3f}")
    with open(os.path.join(RESULTS_DIR, "day05_thresholds.json"), "w") as f:
        json.dump({"per_class_threshold": {c: float(thresholds[j]) for j, c in enumerate(CATEGORIES)},
                   "per_class_f1_at_threshold": {c: float(pcf_thr[c]) for c in CATEGORIES},
                   "macro_f1_global_t05": macro_t05,
                   "macro_f1_per_class_thr": macro_thr}, f, indent=2)

    # ----- 4. Error analysis -----
    print("[day05] running error analysis on 30 worst rows...")
    top_err, breakdown = error_analysis(df, y, pred_thr, champion_probs, kw, top_n=30)
    top_err.to_csv(os.path.join(RESULTS_DIR, "day05_error_analysis.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "day05_failure_breakdown.json"), "w") as f:
        json.dump({"total_failures_top30": int(len(top_err)),
                   "breakdown": {kk: int(vv) for kk, vv in breakdown.items()}}, f, indent=2)
    print(f"[day05] failure breakdown (top-30): {breakdown}")

    # ----- 5. Targeted fix -----
    moc = breakdown.get("multi_category_overlap", 0)
    bce_macro_t05 = None
    bce_macro_thr = None
    bce_thresholds = None
    bce_pcf_thr = None
    do_bce = (moc / max(len(top_err), 1)) >= 0.40
    print(f"[day05] multi-cat overlap = {moc}/{len(top_err)} ({moc/max(len(top_err),1):.0%}) — "
          f"BCE rerun {'TRIGGERED' if do_bce else 'SKIPPED'}")
    if do_bce:
        print("[day05] running BCE multi-label baseline (LogReg OvR + per-class thr)...")
        bce_probs, bce_meta = run_bce_multilabel(texts, y)
        bce_pred_t05 = (bce_probs >= 0.5).astype(int)
        bce_macro_t05 = macro_f1(y, bce_pred_t05)
        bce_thresholds, bce_pcf_thr = optimize_per_class_thresholds(y, bce_probs)
        bce_pred_thr = apply_thresholds(bce_probs, bce_thresholds)
        bce_macro_thr = macro_f1(y, bce_pred_thr)
        print(f"[day05] BCE macro-F1 (thr=0.5) = {bce_macro_t05:.3f}  |  (per-class thr) = {bce_macro_thr:.3f}")

        comp = pd.DataFrame([
            {"strategy": "LGBM tuned (thr=0.5)", "macro_f1": macro_t05,
             **{f"f1_{c}": pcf_t05[c] for c in CATEGORIES}},
            {"strategy": "LGBM tuned + per-class thr", "macro_f1": macro_thr,
             **{f"f1_{c}": pcf_thr[c] for c in CATEGORIES}},
            {"strategy": "BCE (LogReg OvR, thr=0.5)", "macro_f1": bce_macro_t05,
             **{f"f1_{c}": per_class_f1(y, bce_pred_t05)[c] for c in CATEGORIES}},
            {"strategy": "BCE (LogReg OvR) + per-class thr", "macro_f1": bce_macro_thr,
             **{f"f1_{c}": bce_pcf_thr[c] for c in CATEGORIES}},
        ])
        comp.to_csv(os.path.join(RESULTS_DIR, "day05_bce_comparison.csv"), index=False)

    # ----- 6. Master summary -----
    summary = {
        "day2_baseline_macro_f1_5fold_cv": DAY2_BASELINE_MACRO_F1,
        "optuna_trials": N_TRIALS,
        "best_params": best,
        "champion_macro_f1_t05": macro_t05,
        "champion_macro_f1_per_class_thr": macro_thr,
        "per_class_f1_t05": pcf_t05,
        "per_class_f1_thr": pcf_thr,
        "per_class_thresholds": {c: float(thresholds[j]) for j, c in enumerate(CATEGORIES)},
        "failure_breakdown_top30": {kk: int(vv) for kk, vv in breakdown.items()},
        "bce_triggered": do_bce,
        "bce_macro_f1_t05": bce_macro_t05,
        "bce_macro_f1_per_class_thr": bce_macro_thr,
        "bce_thresholds": ({c: float(bce_thresholds[j]) for j, c in enumerate(CATEGORIES)}
                           if bce_thresholds is not None else None),
        "bce_per_class_f1_thr": ({c: float(bce_pcf_thr[c]) for c in CATEGORIES}
                                 if bce_pcf_thr is not None else None),
    }
    with open(os.path.join(RESULTS_DIR, "day05_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[day05] wrote {RESULTS_DIR}/day05_metrics.json")
    return summary


if __name__ == "__main__":
    sys.exit(main() and 0)
