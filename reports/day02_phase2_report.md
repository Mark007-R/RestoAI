# Day 02 — Phase 2a: Sentiment + Complaint Classifier Comparison — RestoAI Production Upgrade
**Date:** 2026-05-12
**Day:** 02 of 7

## Resume gap progress
**Gap:** Multi-component NLP eval — the keyword "complaint classifier" and the VADER-based sentiment path were Day-1's two real, *replaceable* models. Day 2 measures whether modern alternatives win on the exact same eval sets.
**Today's contribution:** Ran 3 sentiment strategies (VADER reused + DistilBERT + NLI zero-shot) and 4 complaint strategies (keyword reused + TF-IDF+LGBM + SBERT+LGBM + NLI zero-shot) head-to-head on the locked Day-1 eval. Result: NLI zero-shot is the highest-scoring sentiment strategy measured (macro-F1 0.701 vs VADER 0.466). On complaints the keyword baseline holds the highest measured macro-F1 (0.820 vs trained best 0.682); TF-IDF+LGBM beats keyword F1 on 2 of 8 classes and on delivery precision. The Claude Opus 4.6 zero-shot legs prescribed by the SKILL were not run (no API key in autonomous-run env).

## Files touched
- `scripts/day02_phase2a.py` (created, 574 lines) — orchestrator: runs all strategies, 5-fold stratified CV for trained complaint heads, writes metrics and sample CSVs
- `results/phase2a_metrics.json` (created) — full per-class metrics for every strategy
- `results/phase2a_results.csv` (created) — flat leaderboard
- `results/phase2a_sentiment_preds.csv` (created, 200 rows × {idx,text_preview,rating,source,gold,3 pred cols})
- `results/phase2a_complaints_preds.csv` (created, 100 rows × {idx,text_preview,source,gold,4 pred cols})
- `results/phase2a_lexical_overlap.json` (created) — per-category gold-positive literal-stem hit rates
- `results/samples/day02_*_wins.csv` / `_losses.csv` (14 files) — up to 5 wins + up to 5 losses per strategy per component; `day02_complaints_nli_zeroshot_wins.csv` contains only 1 row because that strategy's subset_accuracy on the 100-review eval was 0.010
- `.gitignore` — adds `results/*.joblib`, `logs_day*.txt`

## Setup
- **Compute:** CPU only. Total wall time **685 s (11.4 min)**.
- **Sentiment eval:** Day-1 locked set of 200 reviews stratified across 4 source CSVs (zomato 58 / Resreviews 79 / mumbaires 54 / reviews.csv 9; zomato2 item-mentions excluded), 3 gold classes (68/66/66).
- **Complaint eval:** Day-1 locked set of 100 reviews, 8 multi-label categories, gold from `RICH_PATTERNS` regex. Class supports: service 53, food_quality 60, hygiene 13, price 27, delivery 19, portion 26, ambience 34, variety 14.
- **Models used:** `distilbert-base-uncased-finetuned-sst-2-english` (binary → 3-class with P_pos thresholds 0.70 / 0.30), `valhalla/distilbart-mnli-12-3` (NLI zero-shot), `sentence-transformers/all-MiniLM-L6-v2` (SBERT), `lightgbm 4.6` (one-vs-rest binary heads, 120 boost rounds, scale_pos_weight set per-class). All CPU.
- **CV protocol for trained complaint classifiers:** 5-fold stratified-by-primary-label CV on the 100-review gold set, seed=0. Out-of-fold predictions only. Threshold = 0.5 for every class. Raw probabilities were not persisted — only the binarized OvR output (see "What's not in this session's output").
- **Claude Opus 4.6 zero-shot:** **not run.** `ANTHROPIC_API_KEY` is not exposed inside the scheduled-task subprocess (same constraint as Day-1's RAG judge). No measurement available.

## Experiments

### Experiment 2.1 — Sentiment 3-way (VADER vs DistilBERT vs NLI zero-shot)
**Hypothesis:** Day 1 showed VADER's macro-F1 = 0.466 with Neutral F1 = 0.081 (recall = 4.5%). DistilBERT SST-2 is binary so a P-threshold map to 3-class should improve the firehose problem on Positive but not necessarily recover Neutral. NLI zero-shot (with explicit "neutral" hypothesis) should be the first model that actually gets Neutral right, because it directly tests "is this neutral?" rather than inferring it from a corridor.

**Method:** 200-review eval. DistilBERT: tokenize → softmax → if `P_pos ≥ 0.70` → Positive, `≤ 0.30` → Negative, else → Neutral. NLI: `pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")` with `candidate_labels=["positive","neutral","negative"]` and hypothesis template `"The sentiment of this restaurant review is {}."` (top-1 multi-class).

**Result:**

| Strategy | macro-F1 | accuracy | Pos F1 | Neu F1 | Neg F1 | ms/review |
|---|---|---|---|---|---|---|
| VADER (Day-1 baseline) | 0.466 | 0.550 | 0.650 | **0.081** | 0.667 | 0.4 |
| DistilBERT SST-2 (binary → 3-class) | 0.536 | 0.635 | 0.731 | 0.113 | 0.765 | 49.8 |
| **NLI zero-shot (distilbart-mnli-12-3)** | **0.701** | **0.735** | **0.805** | **0.478** | **0.819** | 589.3 |
| Claude Opus 4.6 zero-shot | — | — | — | — | — | not run |

**Per-class for the Day-1 headline failure mode (Neutral):**

| Strategy | Neutral P | Neutral R | Neutral F1 |
|---|---|---|---|
| VADER | 0.375 | 0.045 | 0.081 |
| DistilBERT SST-2 | 0.800 | 0.061 | 0.113 |
| **NLI zero-shot** | **0.846** | **0.333** | **0.478** |

**Measured outcomes:**

1. NLI zero-shot macro-F1 = 0.701. Day-1 reported a bootstrap upper 95% CI of 0.520 for VADER, so 0.701 falls outside that CI.
2. Neutral F1 across the three strategies: 0.081 → 0.113 → 0.478. Decomposed: Neutral precision 0.375 → 0.800 → 0.846; Neutral recall 0.045 → 0.061 → 0.333. The recall lift from DistilBERT-threshold to NLI is the largest single contributor to the F1 lift.
3. DistilBERT SST-2 with thresholds 0.70 / 0.30: Neutral precision = 0.800, Neutral recall = 0.061. The threshold band (0.30, 0.70) is narrow; how many of the 200 reviews fell into it is not directly captured in the saved metrics — only the binarized output is in `results/phase2a_sentiment_preds.csv` (saving raw P_pos would let us verify the "narrow band" hypothesis; that's a Day-5 sweep candidate).
4. Latency ratio NLI / VADER = 589.3 / 0.4 ≈ 1473×. Whether 0.6 s/review is "acceptable" depends on RestoAI's ingestion pattern; the Day-4 module design is a plan, not a Day-2 measurement.

### Experiment 2.2 — Complaint classifier 4-way (keyword vs TF-IDF+LGBM vs SBERT+LGBM vs NLI zero-shot)
**Hypothesis:** TF-IDF + LightGBM and SBERT + LightGBM should beat the keyword baseline on rare classes by generalizing past the literal substring matches in `CATEGORY_KEYWORDS` (the Day-1 prediction). NLI zero-shot will be in the middle — better than keyword on paraphrase, worse on terse phrasing where keyword is mechanical.

**Method:** 100-review multi-label eval, 8 categories. TF-IDF: word 1-2gram + char 3-5gram features → LightGBM OvR (8 binary heads). SBERT: 384-d `all-MiniLM-L6-v2` embeddings → LightGBM OvR. NLI multi-label: distilbart-mnli with friendly category labels ("rude or inattentive service", "bad food taste or quality", ...) and hypothesis `"This review mentions {}."` with `multi_label=True`, threshold = 0.5. Trained classifiers evaluated with **5-fold stratified CV** — out-of-fold predictions only.

**Result:**

| Strategy | macro-F1 | micro-F1 | subset_acc | hamming | ms/review |
|---|---|---|---|---|---|
| **Keyword baseline (Day-1)** | **0.820** | **0.847** | **0.430** | 0.099 | 0.1 |
| TF-IDF + LightGBM (5-fold CV) | 0.682 | 0.806 | 0.380 | 0.110 | 245.7 |
| SBERT + LightGBM (5-fold CV) | 0.344 | 0.554 | 0.130 | 0.249 | 43.7 |
| NLI zero-shot multi-label | 0.407 | 0.434 | 0.010 | 0.328 | 1757.1 |
| Claude Opus 4.6 zero-shot | — | — | — | — | not run |

**Per-class F1 — common-support categories (n_pos ≥ 26), TF-IDF+LGBM vs keyword:**

| Category | n_pos | keyword F1 | TF-IDF+LGBM F1 | Δ (TF-IDF − keyword) |
|---|---|---|---|---|
| service | 53 | 0.991 | 0.990 | −0.000 |
| food_quality | 60 | 0.830 | 0.874 | **+0.044** |
| portion | 26 | 0.852 | 0.920 | **+0.068** |
| ambience | 34 | 0.970 | 0.833 | **−0.137** |

(Values rounded from `results/phase2a_metrics.json`. Service exact delta = 0.990476 − 0.990654 = −0.000178.)

**Per-class precision / recall / F1 — low-support categories (n_pos ≤ 27):**

| Category | n_pos | keyword P/R/F1 | TF-IDF+LGBM P/R/F1 | NLI multi-label P/R/F1 |
|---|---|---|---|---|
| hygiene | 13 | 0.769 / 0.769 / 0.769 | 1.000 / 0.231 / 0.375 | 0.267 / 0.615 / 0.372 |
| variety | 14 | 0.583 / 1.000 / 0.737 | 1.000 / 0.571 / 0.727 | 0.125 / 0.286 / 0.174 |
| delivery | 19 | 0.432 / 0.842 / 0.571 | 0.714 / 0.263 / 0.385 | 0.222 / 0.316 / 0.261 |
| price | 27 | 0.821 / 0.852 / 0.836 | 0.857 / 0.222 / 0.353 | 0.484 / 0.556 / 0.517 |

(Values from `results/phase2a_metrics.json`. On this gold set at threshold 0.5, the keyword baseline produces the highest F1 in each of the four rows; the trained TF-IDF head reaches 1.000 precision on hygiene and variety but at recall costs that drop F1 below keyword. Whether a lower per-class threshold or larger training set would change the F1 ordering is untested.)

**Measured outcomes:**

1. **Macro-F1 leaderboard:** keyword 0.820, TF-IDF+LGBM 0.682, NLI multi-label 0.407, SBERT+LGBM 0.344. **The Day-1 bootstrap upper 95% CI for the keyword baseline was 0.859 (`results/baseline_ci.json`); no Day-2 strategy crossed it.** Day 1 noted that the gold labeller (`RICH_PATTERNS` regex) and the baseline (`CATEGORY_KEYWORDS`) share a substring vocabulary by construction. Whether trained ML would beat the baseline on a non-circular gold set (e.g., human-relabeled, or RICH_PATTERNS-labeled but with a deliberately disjoint lexicon for training labels) is untested.

2. **Delivery precision: keyword 0.432, TF-IDF+LGBM 0.714.** Absolute delta +0.282; ratio 1.65×. Delivery recall: keyword 0.842, TF-IDF+LGBM 0.263 (delta −0.579). The Day-1 audit identified delivery as the precision failure mode of the keyword baseline; the Day-2 trained classifier moves the precision–recall trade-off but the F1 score for delivery still favors keyword (0.571 vs 0.385). The threshold = 0.5 OvR setting and the small number of delivery positives per CV fold (≈3) are two factors that could be changed in Day 5; whether changing them flips the F1 outcome is untested.

3. **Where the trained classifier actually beats the baseline:** `food_quality` (+0.044) and `portion` (+0.068). These are also the two categories with the **lowest literal-name match rate in the gold positives** (food_quality: 3.3%, portion: 3.8% — measured, `results/phase2a_lexical_overlap.json`). For comparison, the categories where keyword ties or wins big — `service` (71.7% literal hit), `ambience` (70.6%), `variety` (78.6%) — all show the literal stem in the majority of gold-positive reviews. So the win/loss pattern for the two ML-winners is consistent with "paraphrastic categories favor ML"; for the ML-losers (`hygiene` −0.39, `price` −0.48, `delivery` −0.19) the picture is mixed: those have moderate literal-hit rates (23–42%) but **also small support** (n=13, 27, 19 respectively), so the gap is plausibly a mix of vocabulary structure and CV data starvation rather than vocabulary alone. The measured per-category overlap is in `results/phase2a_lexical_overlap.json`.

**SBERT + LightGBM scored macro-F1 = 0.344.** Setup: 80 training rows per CV fold, 384-d embeddings, 8 binary OvR heads at threshold 0.5. Per-class recall (`results/phase2a_metrics.json`): variety 0.000, delivery 0.053, hygiene 0.077, price 0.111, portion 0.154. Per-class F1 for the same: 0.000 / 0.091 / 0.143 / 0.167 / 0.229. Service (F1 = 0.741), food_quality (0.740), ambience (0.644) scored higher. Whether more training data (e.g., regex-weak-labeled bulk) or a lower per-class threshold would change this is the **Day-5 weak-supervision hypothesis**, untested in this session.

**NLI zero-shot on complaints:** macro-F1 = 0.407, subset_accuracy = 0.010 (1 of 100 reviews matched the full gold category set exactly). Per-class F1 ranged 0.174 (variety) to 0.517 (price); see `results/phase2a_metrics.json`. Whether different friendly-label phrasing or a different threshold would change this is untested.

## Head-to-Head Comparison (canonical Day-2 leaderboard)

### Sentiment
| Rank | Strategy | macro-F1 | acc | Neu F1 | Latency | Cost | Notes |
|------|---|---|---|---|---|---|---|
| 1 | **NLI zero-shot (distilbart-mnli-12-3)** | **0.701** | **0.735** | **0.478** | 589.3 ms | free (local) | Highest macro-F1 measured. Outside Day-1 VADER upper 95% CI (0.520). |
| 2 | DistilBERT SST-2 + thresholds 0.70/0.30 | 0.536 | 0.635 | 0.113 | 49.8 ms | free (local) | Neutral precision 0.800 / recall 0.061 measured. |
| 3 | VADER (Day-1 baseline) | 0.466 | 0.550 | 0.081 | 0.4 ms | free | Baseline. |
| n/a | Claude Opus 4.6 zero-shot | — | — | — | — | — | Not run: ANTHROPIC_API_KEY not present in scheduled-task subprocess env. |

### Complaints (8-way multi-label)
| Rank | Strategy | macro-F1 | micro-F1 | subset_acc | Latency | Notes |
|------|---|---|---|---|---|---|
| 1 | **Keyword baseline (Day-1)** | **0.820** | **0.847** | **0.430** | 0.1 ms | Highest macro-F1 measured. Gold labeller shares substring vocab with this baseline (Day-1 audit). |
| 2 | TF-IDF + LightGBM (5-fold CV) | 0.682 | 0.806 | 0.380 | 245.7 ms | Beats keyword on food_quality F1 (+0.044) and portion F1 (+0.068); delivery precision 0.43 → 0.71. |
| 3 | NLI zero-shot multi-label | 0.407 | 0.434 | 0.010 | 1757.1 ms | Per-class F1 0.174–0.517. |
| 4 | SBERT + LightGBM (5-fold CV) | 0.344 | 0.554 | 0.130 | 43.7 ms | Variety F1 = 0.000; hygiene F1 = 0.143. 80 training rows × 384-d features at threshold 0.5. |
| n/a | Claude Opus 4.6 zero-shot | — | — | — | — | Not run: ANTHROPIC_API_KEY not present in scheduled-task subprocess env. |

## Key Findings

1. **Sentiment: NLI zero-shot macro-F1 = 0.701, VADER macro-F1 = 0.466, absolute delta +0.235.** Neutral F1: VADER 0.081 → NLI 0.478 (ratio 5.9×). 0.701 is above the Day-1 bootstrap upper 95% CI for VADER (0.520) by 0.181. Day-4 integration is a *plan*, not done in this session.
2. **Complaints macro-F1 leaderboard:** keyword 0.820, TF-IDF+LGBM 0.682, NLI multi-label 0.407, SBERT+LGBM 0.344. None crossed the Day-1 keyword baseline upper 95% CI of 0.859. Measured per-class deltas where TF-IDF+LGBM beat keyword: food_quality F1 +0.044, portion F1 +0.068. Measured precision changes on delivery: 0.432 → 0.714 (delta +0.282); paired with a recall change of 0.842 → 0.263, so delivery **F1** went 0.571 → 0.385 (still favoring keyword on F1 at the default threshold).
3. **TF-IDF rare-class per-class P/R measured at threshold 0.5:** hygiene P=1.000 R=0.231 F1=0.375; variety P=1.000 R=0.571 F1=0.727; price P=0.857 R=0.222 F1=0.353; delivery P=0.714 R=0.263 F1=0.385. Reaching the Day-1 win threshold of macro-F1 > 0.86 on this gold set, and whether per-class threshold sweep + weak supervision would do so, is untested.
4. **Worst per-class measurements:** SBERT+LGBM macro-F1 = 0.344, variety F1 = 0.000, hygiene F1 = 0.143 at threshold 0.5. NLI multi-label subset_accuracy = 0.010 — 1 of 100 reviews matched the gold category set exactly.
5. **Measured latency per review:** VADER 0.4 ms, DistilBERT-SST2 49.8 ms, NLI 589.3 ms, TF-IDF+LGBM 245.7 ms, SBERT+LGBM 43.7 ms, NLI multi-label 1757.1 ms. (TF-IDF+LGBM latency includes feature-vectorization fit + transform inside each fold; a fit-once production deployment would be lower, untested.)

## Sample Outputs Saved
- `results/samples/day02_sentiment_{vader,distilbert_sst2,nli_zeroshot}_{wins,losses}.csv` — first 5 correct + first 5 incorrect per strategy on the 200-review eval (6 files)
- `results/samples/day02_complaints_{keyword,tfidf_lgbm,sbert_lgbm,nli_zeroshot}_{wins,losses}.csv` — first 5 exact-match + first 5 non-exact-match per strategy on the 100-review eval (8 files)
- `results/phase2a_sentiment_preds.csv` — per-row gold + pred for VADER / DistilBERT / NLI (200 rows × {idx,text_preview,rating,source,gold,3 pred columns})
- `results/phase2a_complaints_preds.csv` — per-row gold + pred for keyword / TF-IDF / SBERT / NLI (100 rows × {idx,text_preview,source,gold,4 pred columns})
- `results/phase2a_lexical_overlap.json` — per-category measurement of literal-name-hit rate and any-stem-hit rate across gold-positive reviews (basis for Interpretation point 3 in the complaints block).

## Day-1 win thresholds check

| Metric | Day-1 baseline | Day-1 win threshold (upper 95% CI) | Day-2 best | Met? |
|---|---|---|---|---|
| Sentiment macro-F1 | 0.466 | > 0.52 | **0.701** (NLI) | ✅ |
| Sentiment Neutral F1 | 0.081 | > 0.18 | **0.478** (NLI) | ✅ |
| Complaints macro-F1 | 0.820 | > 0.86 | 0.682 (TF-IDF) | ❌ |
| Complaints subset_acc | 0.430 | > 0.54 | 0.380 (TF-IDF) | ❌ |

Sentiment cleared both thresholds (NLI macro-F1 0.701 > 0.52; Neutral F1 0.478 > 0.18). Complaints cleared neither.

## Next Day
Day 3 — Phase 2b RAG comparison, per SKILL:
- Replace `_synthesize_intelligent_answer` template logic with LLM-backed synthesis.
- Compare 4 configs: (1) current template baseline (Day-1 composite = 0.686), (2) LLM on existing per-review chunks, (3) LLM with recursive-character chunking, (4) LLM + cross-encoder rerank (ms-marco-MiniLM-L-6-v2).
- Run RAGAS on the 50-QA eval. Day-1 win threshold: composite > 0.76.
- Requires `ANTHROPIC_API_KEY` (or compatible) for synthesis + judge. If the key is not surfaced in autonomous mode, Day 3 will need an interactive run or a structural-metric fallback (Day-1 pattern).

## Code Changes
- `scripts/day02_phase2a.py` — full orchestrator
- `.gitignore` — `results/*.joblib`, `logs_day*.txt` added to exclude rules

## What's not in this session's output (and why)

- **Claude Opus 4.6 zero-shot row (both legs).** Not run because `ANTHROPIC_API_KEY` is not exposed inside the scheduled-task subprocess (same constraint that hit Day-1's RAG judge). No measurement available.
- **Raw OOF probabilities from the trained complaint heads.** The CV runner binarized predictions at threshold 0.5 inside `_kfold_cv_train_predict` (`y_oof = (p >= 0.5).astype(int)`) and only the binarized output was persisted. Day-5 per-class threshold sweep will therefore need to re-run the CV with probability-saving turned on. This is a correction to an earlier draft that incorrectly stated the probabilities were saved.
- **Bootstrap 95% CIs on Day-2 strategies.** Day 1 produced bootstrap CIs for the baselines; Day 2 only compares point estimates against Day-1's CI upper bounds. A symmetric CI on Day-2 strategies has not been computed.
- **Per-source slice analysis on Day-2 strategies.** Day-1 polish addendum #1 produced per-source breakdowns for the baselines (zomato / Resreviews / mumbaires / reviews.csv). The same cuts on the Day-2 strategies have not been computed.
- **Day-2 visualization PNGs.** Day 1 produced 6 chart files; Day 2 produced metric tables only. No PNGs were generated this session.
- **Production wiring of the NLI sentiment classifier into `manager_system/analyzer.py`.** Day 4 by design.
- **Per-class threshold tuning, weak-supervision pass for SBERT, alternative NLI prompts.** Day 5 by design.

## Claims that *are* backed by saved measurements

- Every macro-F1, micro-F1, per-class P/R/F1, subset_accuracy, hamming_loss, and ms/review value in this report comes from `results/phase2a_metrics.json`.
- The per-category literal-stem hit rates in the complaints Interpretation block come from `results/phase2a_lexical_overlap.json` (method documented in the JSON's `method` field).
- The Day-1 baseline CI numbers come from `results/baseline_ci.json`.
- Per-row predictions and the wins/losses sample files live under `results/`.
- Mechanism sentences (e.g., "the NLI hypothesis template forces 'neutral' as a first-class option") are explicitly framed as hypotheses, not measurements. Verifying them would require additional experiments listed under "What's not in this session's output".
