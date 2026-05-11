# Day 01 — Audit + Eval Set + Baseline — RestoAI Production Upgrade
**Date:** 2026-05-11
**Day:** 01 of 7

## Resume gap progress
**Gap:** Multi-component NLP eval — the project ships three "AI" components but only one is a real model.
**Today's contribution:** Pinned down what's real vs what's templated, built reproducible eval sets for all three components (200/100/50), and measured the honest baseline that every Phase-2 strategy must beat.

## Files touched
- `docs/COMPONENT_AUDIT.md` (created) — definitive description of the current pipeline; calls out the two fakes
- `data/eval/sentiment_eval.csv` (created, 200 rows)
- `data/eval/complaint_eval.csv` (created, 100 rows)
- `data/eval/rag_qa_eval.json` (created, 50 questions across 50 restaurants × 6 intents)
- `data/eval/eval_corpus_meta.json` (created)
- `scripts/build_eval_sets.py` (created)
- `scripts/run_baselines.py` (created)
- `scripts/structural_rag_metrics.py` (created)
- `results/baseline_metrics.json` (created — canonical Day-1 baseline)
- `results/baseline_sentiment_preds.csv`, `results/baseline_complaints_preds.csv`, `results/baseline_rag_answers.json`, `results/baseline_rag_structural.csv` (created)
- `results/samples/` (5 wins / 5 losses for sentiment & complaints, top/bottom 5 for RAG)
- `.gitignore` (allow rules added for `data/`, `results/`, `docs/`, `reports/`, `scripts/`, `tests/`)

## Setup
- **Compute:** CPU only. SBERT `all-MiniLM-L6-v2` loaded once for RAG retrieval. ~13 minutes wall time end-to-end (810s).
- **Dataset slice:** all 5 CSVs in `datasets/` (zomato.csv 51K rows, zomato2.csv 123K, mumbaires 7.9K, Resreviews 10K, reviews.csv 1.1K) → 23,802 candidate review snippets after length filtering (30 ≤ chars ≤ 1500).
- **Components touched (read-only):** `manager_system/analyzer.py`, `manager_system/rag_chat.py`. No production files were modified today.

## Experiments

### Experiment 1.1 — Sentiment baseline (VADER)
**Hypothesis:** VADER is a real model so it should be a defensible baseline; rating-derived gold labels (≥4 Positive, ≤2 Negative, =3 Neutral) will reveal whether the 0.05 compound threshold is well-calibrated to restaurant prose.
**Method:** `analyze_text_and_keywords(text)` → label compared against rating-derived gold on 200 reviews stratified by class (68/66/66) and across 4 prose-style datasets (zomato.csv 58, Resreviews 79, mumbaires 54, reviews.csv 9; zomato2 item-mentions excluded because they aren't review prose).
**Result:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.486 | **0.985** | 0.650 | 68 |
| Neutral  | 0.375 | **0.045** | 0.081 | 66 |
| Negative | 0.741 | 0.606 | 0.667 | 66 |

- **Macro-F1 = 0.466**
- **Accuracy = 0.55**

**Interpretation:** VADER is a Positive-firehose. Recall of 98.5% on Positive (it labels almost everything ≥0.05 compound) but precision of only 49% (it eats half the Negatives and almost all the Neutrals). The Neutral class is essentially un-recovered (recall 4.5%, F1 0.08) because the (-0.05, +0.05) corridor is too narrow for restaurant prose where reviewers always sprinkle a few sentiment-bearing words. This is the headline gap for Day 2: **a domain-tuned classifier — even a simple TF-IDF + LR — can lift Neutral F1 by an order of magnitude.**

### Experiment 1.2 — Complaint classifier baseline (`categorize_complaints`)
**Hypothesis:** Substring keyword matching will look surprisingly OK on a dataset where keywords are often present, but **subset accuracy** (multi-label exact match) will collapse and rare classes will under-recall.
**Method:** `categorize_complaints(text)` → 8-way multi-label predictions compared against `RICH_PATTERNS` regex gold (deliberately broader than the keyword scan, with word-boundary regex + multi-word phrases) on 100 stratified reviews. Per-class binary P/R/F1 + multi-label aggregate.
**Result:**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| service | 0.981 | 1.000 | **0.991** | 53 |
| ambience | 1.000 | 0.941 | **0.970** | 34 |
| portion | 0.743 | 1.000 | 0.852 | 26 |
| price | 0.821 | 0.852 | 0.836 | 27 |
| food_quality | 0.957 | 0.733 | 0.830 | 60 |
| hygiene | 0.769 | 0.769 | 0.769 | 13 |
| variety | 0.583 | 1.000 | 0.737 | 14 |
| delivery | **0.432** | 0.842 | **0.571** | 19 |

- **Macro-F1 = 0.820**
- **Micro-F1 = 0.847**
- **Subset accuracy (all labels exact) = 0.43**
- **Hamming loss = 0.099**

**Interpretation:** The headline 0.82 macro-F1 is methodologically fragile — the gold labeller (RICH_PATTERNS) shares a substring core with the baseline's `CATEGORY_KEYWORDS`, so the comparison is partly self-referential. The honest numbers are the ones that diverge:

- **Subset accuracy 0.43** — only 43% of reviews get all categories right. On a multi-label task this is the metric a manager dashboard cares about. **This is the headline number to beat.**
- **delivery precision 0.43** — substring matching against `["delivery","late","order","arrived","driver"]` fires constantly on words like "ordered" and "arrived" in non-delivery contexts, producing massive false-positive noise.
- **food_quality recall 0.73** — the keyword list misses paraphrases ("the chef phoned it in", "chewy and grey") because no listed token appears.
- **variety precision 0.58 / hygiene precision 0.77** — small dictionaries → over-firing on adjacent topics.

Day 2 will compare TF-IDF+LightGBM and SBERT+LightGBM against this; the real win is expected on **subset accuracy** and on **delivery precision**.

### Experiment 1.3 — RAG synthesis baseline (`_synthesize_intelligent_answer`)
**Hypothesis:** The function is template strings, not an LLM, so it will mechanically score well on intent-vocabulary recall (it's literally intent-keyed) but fail on rating citation and sentiment-direction faithfulness.
**Method:** Run RAGChat.answer_query on 50 questions (6 intents × 50 distinct restaurants, each restaurant has ≥8 reviews). Compute structural metrics against ground-truth aggregate facts (avg rating, sentiment direction, top-3 categories). The intended LLM-judge step (Claude Opus 4.6 scoring faithfulness/relevancy/groundedness) failed because `ANTHROPIC_API_KEY` is not exposed inside the scheduled-task subprocess; recovered with deterministic structural metrics instead — these probe the same template pathology more directly.
**Result:**

| Metric | Value | What it measures |
|---|---|---|
| sentiment_dir_match | 0.740 | answer's pos/neg/mixed direction matches gold aggregate |
| top_category_hit | **0.980** | answer references any of restaurant's top-3 complaint categories |
| rating_mention | **0.340** | answer cites a numeric rating like "4.2/5" |
| intent_addressed | 0.900 | answer uses vocabulary from question's intent group |
| specificity_score | 0.775 | mean fraction of {rating, top_cat, intent, non-mixed_dir} signals present |
| **composite** (geo-mean of 4 binary rates) | **0.686** | overall structural faithfulness |

**Interpretation:** The templates score well exactly where they should — `intent_addressed = 0.90` and `top_category_hit = 0.98` are mechanical: the prefix injects the retrieved snippets verbatim, and the templates are intent-keyed. The two metrics that actually require *generation* fail:

- **rating_mention = 0.34**: only 34% of answers cite the numeric rating, even though `_synthesize_intelligent_answer` does include it for some intent branches. The rating is *available* in `doc['metadata']['rating']` but the templates only print it inside two of the six intent paths.
- **sentiment_dir_match = 0.74**: 26% of the time the template's hardcoded ratio rule (`pos_count > neg_count * 1.5` etc.) picks the wrong direction relative to ground-truth sentiment computed across all reviews.

The composite 0.686 is the Day-1 ceiling for the keyword-template path. Day 3 (real LLM synthesis on retrieved chunks + cross-encoder rerank) targets > 0.85.

## Head-to-Head Comparison (Day-1 baseline leaderboard)

| Component | Engine | Primary Metric | Secondary | Notes |
|---|---|---|---|---|
| Sentiment | VADER | macro-F1 = **0.466** | acc 0.55; Neutral F1 = 0.08 | Real model. Threshold-bound. |
| Complaints | substring keyword scan | macro-F1 = 0.820 / **subset_acc = 0.43** | hamming = 0.099 | Macro-F1 inflated by gold's keyword overlap; subset_acc is the honest number. |
| RAG synthesis | template if/elif | composite = **0.686** | rating cite 0.34, sentiment match 0.74 | Templates only — no LLM call. |

## Key Findings
1. **Two of three components are not models.** The audit confirmed `categorize_complaints` is a substring scan and `_synthesize_intelligent_answer` is hardcoded `if/elif` template strings. The README claims of "AI categorization" and "AI summary" are not currently backed by anything trainable.
2. **The headline metrics to beat are not the macro averages.** They are: (a) Sentiment **Neutral F1 = 0.081** — VADER essentially can't recover it; (b) Complaint **subset accuracy = 0.43** — exact-match on multi-label is where a real classifier earns its keep; (c) RAG **rating_mention = 0.34** and **sentiment_dir_match = 0.74** — the only two structural signals that require actual generation rather than passing snippets through.
3. **Delivery precision = 0.43** is the most concrete failure mode. The keyword list eats every "ordered" / "arrived" mention regardless of whether delivery is actually being discussed; this single keyword choice hurts the macro by ~0.05.
4. **What didn't work:** the Anthropic LLM-as-judge for RAG. Scheduled-task subprocesses don't see `ANTHROPIC_API_KEY` (env shows it set but value resolves to empty inside Python). Structural metrics replaced it; they are more honest for Day 1 since the templates have no model to "judge" anyway. Day 3 will run the LLM judge interactively when measuring real LLM-backed synthesis.

## Sample Outputs Saved
- `results/samples/sentiment_baseline_wins.csv` — 5 correct VADER predictions
- `results/samples/sentiment_baseline_losses.csv` — 5 misclassifications (most are Neutral→Positive)
- `results/samples/complaints_baseline_wins.csv` — 5 exact-match cases
- `results/samples/complaints_baseline_losses.csv` — 8 cases with `missed`/`spurious` columns showing where keyword scan diverged
- `results/baseline_rag_answers.json` — full 50 answers with retrieved sources
- `results/baseline_rag_structural.csv` — per-question structural scores

## Next Day
Day 2 — Phase 2a comparisons:
- Sentiment: VADER vs DistilBERT-base-uncased-finetuned-sst-2-english vs Claude Opus 4.6 zero-shot on the 200-review eval (will need interactive run for the Claude leg).
- Complaints: substring keyword vs TF-IDF + LightGBM (one-vs-rest, 8 binary heads) vs SBERT + LightGBM head vs Claude Opus 4.6 zero-shot on the 100-review multi-label eval. Pay particular attention to **subset accuracy** (current 0.43), **Neutral sentiment F1** (current 0.08), and **delivery precision** (current 0.43).

## Code Changes
- `docs/COMPONENT_AUDIT.md` (NEW) — full audit
- `scripts/build_eval_sets.py` (NEW) — reproducible eval builder (seeded RNG)
- `scripts/run_baselines.py` (NEW) — sentiment + complaints baseline + RAG answer collection (LLM judge attempted)
- `scripts/structural_rag_metrics.py` (NEW) — deterministic RAG structural metrics that replaced the failed LLM judge
- `.gitignore` — allow `data/`, `results/`, `docs/`, `reports/`, `scripts/`, `tests/`
- No changes to `manager_system/analyzer.py` or `manager_system/rag_chat.py` today (Day 4 is the modification day; Day 1 is read-only audit per the SKILL).

---

## Day-1 polish addendum — slice analysis + visualizations
**Added:** 2026-05-11 (same day as the baseline)
**New files:** `scripts/baseline_slice_analysis.py`, `scripts/baseline_visualizations.py`, `results/baseline_slices.json`, `results/charts/*.png` (6 PNGs)

The first thing a hiring manager asks of a baseline is "is the model failing uniformly or on a specific data slice?" The polish pass cuts the three baselines by source dataset (sentiment, complaints) and by intent (RAG) and produces the chart artifacts referenced in this report.

### Polish Experiment 1.4 — Sentiment by source dataset

| source | n | macro-F1 | acc | Pos F1 | Neu F1 | Neg F1 |
|---|---|---|---|---|---|---|
| Resreviews.csv | 79 | 0.467 | 0.582 | 0.706 | **0.000** | 0.696 |
| mumbaires.csv | 54 | 0.487 | 0.593 | 0.667 | 0.143 | 0.650 |
| reviews.csv | 9 | 0.400 | 0.556 | 0.800 | **0.000** | 0.400 |
| zomato.csv | 58 | 0.450 | 0.466 | 0.526 | 0.133 | 0.690 |

**Finding:** Neutral F1 is ≤ 0.15 across **every** source (and exactly 0.00 on two of them). VADER's failure on Neutral is not a dataset-distribution problem — it is a fundamental threshold problem with the (-0.05, +0.05) compound corridor on restaurant prose. zomato.csv is also the worst overall (acc 0.466) because its review snippets (extracted from list literals) are noisier and tend toward borderline language. Charts: `results/charts/sentiment_confusion.png`, `results/charts/sentiment_by_source.png`.

### Polish Experiment 1.5 — Complaint baseline by source dataset

| source | n | macro-F1* | subset_acc | classes_present |
|---|---|---|---|---|
| Resreviews.csv | 38 | 0.829 | **0.526** | 8 |
| mumbaires.csv | 32 | 0.834 | 0.406 | 8 |
| zomato.csv | 28 | 0.736 | **0.321** | 8 |

*Macro excludes classes absent in that slice's gold.

**Finding:** subset accuracy varies almost 2× across sources (0.32 zomato → 0.53 Resreviews). The keyword scan does noticeably worse on zomato — the same noisier-text problem that hurts VADER. Charts: `results/charts/complaints_per_class_f1.png`, `results/charts/complaints_failure_breakdown.png` (the failure breakdown shows `delivery` is the dominant spurious-label source, confirming the precision = 0.43 finding from the main baseline).

### Polish Experiment 1.6 — RAG template baseline by intent

| intent | n | sent dir | top cat | rating cite | intent addr | specificity |
|---|---|---|---|---|---|---|
| quality   | 9 | 0.89 | 1.00 | **1.00** | 1.00 | **1.00** |
| recommend | 8 | 0.75 | 1.00 | 0.88 | 1.00 | 0.94 |
| price     | 8 | 0.88 | 1.00 | **0.12** | **0.38** | 0.59 |
| service   | 9 | 0.67 | 1.00 | **0.00** | 1.00 | 0.72 |
| ambience  | 8 | 0.62 | 1.00 | **0.00** | 1.00 | 0.69 |
| hygiene   | 8 | 0.62 | 0.88 | **0.00** | 1.00 | 0.69 |

**Finding (the most concrete one of Day 1):** The RAG composite of 0.686 is not uniform across intents — it is a **per-branch template defect**. The `quality` and `recommend` branches in `_synthesize_intelligent_answer` happen to print the avg rating; the four other branches **never** do (rating_cite ≤ 0.12). Likewise, the `price` branch (line 561+) uses vocabulary that doesn't overlap our `price` intent set (`expensive/cheap/value/worth`), giving intent_addressed only 0.38. **A 5-line fix to the existing template — append the rating sentence to every branch, broaden the price vocabulary — would lift the composite from 0.686 toward ~0.85 without any LLM.** That would normally be tempting low-hanging fruit, but it's exactly the wrong direction for this sprint: Day 3 replaces the templates wholesale with real LLM synthesis, so the right move is to leave the template baseline as a clean comparison floor rather than patch it up.

Charts: `results/charts/rag_per_intent_metrics.png`, `results/charts/rag_specificity_distribution.png`.

### Polish key takeaway
The slice-by-source breakdown turns three macro numbers into a **structural map** of where the baseline fails: VADER fails uniformly on Neutral (threshold problem, not data problem), the keyword-scan complaints fail worst on noisy zomato extracted text (data problem, helped by a real model), and the RAG template fails per-intent-branch (4 of 6 branches never cite ratings). All three failure modes are addressable in Days 2–3 by exactly the strategies the SKILL prescribes — the polish makes the prescription specific.

---

## Day-1 polish addendum #2 — bootstrap 95% confidence intervals
**Added:** 2026-05-11 (same day as the baseline)
**New files:** `scripts/bootstrap_baseline_ci.py`, `results/baseline_ci.json`, `results/charts/baseline_ci.png`

Every Phase-2 claim from Day 2 onward has the form "model X beat the keyword baseline by ΔF1". A +0.05 ΔF1 only means something if it exceeds the noise floor of the baseline measurement itself. With evaluation sets of 200 / 100 / 50 reviews respectively, that floor is non-trivial and was previously unmeasured. This addendum runs a 1000-sample non-parametric bootstrap (row-level resampling with replacement, seed = 20260511) over each baseline's per-row prediction file and reports the 2.5%/97.5% percentile CI for every headline metric.

### Headline metrics with 95% CI

| Metric | Point | 95% CI | CI width | What this means for Phase 2 |
|---|---|---|---|---|
| Sentiment macro-F1 | 0.466 | [0.411, 0.520] | 0.109 | DistilBERT / Claude must clear **0.52** to credibly beat VADER macro. |
| Sentiment **Neutral F1** | 0.081 | [0.000, 0.177] | 0.177 | Floor is essentially zero. Even modest Neutral recovery is statistically detectable. |
| Sentiment Positive F1 | 0.650 | [0.573, 0.725] | 0.152 | Already strong (Pos = firehose problem); hard to beat by enough. |
| Sentiment Negative F1 | 0.667 | [0.581, 0.745] | 0.164 | Similar story; sensible Negative model needs ≥ 0.75 for a defensible claim. |
| Complaints macro-F1 | 0.820 | [0.770, 0.859] | 0.089 | Tighter than expected — keyword baseline is more stable on this gold set than the report's "methodologically fragile" caveat suggested. Phase-2 needs ≥ 0.86 to clearly beat. |
| Complaints **subset accuracy** | 0.430 | [0.330, 0.540] | **0.210** | Honest multi-label metric. Phase-2 must clear **0.54** to make a real claim of multi-label fidelity. |
| Complaints micro-F1 | 0.847 | [0.804, 0.882] | 0.078 | Tightest interval overall. |
| Complaints **delivery F1** | 0.571 | [0.400, 0.714] | **0.314** | Widest CI in the whole report. The delivery failure mode (precision 0.43) is real but the precise size is very noisy at n=100 — anchoring Phase-2 claims on this single class is risky. |
| Complaints hygiene F1 | 0.769 | [0.609, 0.900] | 0.291 | Wide because support = 13. Treat as illustrative, not load-bearing. |
| Complaints variety F1 | 0.737 | [0.560, 0.870] | 0.310 | Wide because support = 14. Same caveat. |
| RAG composite | 0.686 | [0.601, 0.764] | 0.162 | LLM-synthesis (Day 3) must reach ≥ **0.76** to count as a real win, not just a noise excursion. |
| RAG rating_mention | 0.340 | [0.220, 0.480] | 0.260 | The "rating not cited" gap is real but the size is noisy. |
| RAG sentiment_dir_match | 0.740 | [0.600, 0.860] | 0.260 | Wide. |
| RAG top_category_hit | 0.980 | [0.940, 1.000] | 0.060 | Ceiling effect — already mechanically very high. |
| RAG intent_addressed | 0.900 | [0.820, 0.980] | 0.160 | High; LLM gain here is bounded. |

Chart: `results/charts/baseline_ci.png` shows the eight headline metrics with error bars.

### What changes in interpretation

1. **The "complaint baseline is statistically fragile" caveat needs to be softened on macro-F1.** Bootstrap shows macro-F1 has a CI half-width of just ±0.04 — narrow. The methodological concern (RICH_PATTERNS shares substring core with CATEGORY_KEYWORDS) still holds for the *level* of the score, but the *measurement* itself is stable. The honest framing is: "the keyword baseline scores 0.82 ± 0.04 macro-F1 on this auto-generated gold set; if it scored 0.40 on human-labelled gold, that would still be the apples-to-apples improvement target for Phase 2."
2. **The honest Day-2/3 win thresholds are now explicit, not aspirational.** Day 2 sentiment must hit ≥ 0.52 macro-F1. Day 2 complaint must hit ≥ 0.86 macro-F1 (or ≥ 0.54 subset accuracy — these will likely move together). Day 3 RAG must hit ≥ 0.76 composite. Below those, the result is a noise excursion, not a model improvement.
3. **Delivery F1's CI is the widest in the report (±0.16).** The Day-1 narrative that "delivery precision 0.43 is the headline failure mode" survives — but the precise size is uncertain at n=100. If a Phase-2 strategy claims to fix this specifically, the supporting numbers should come from a held-out larger sample, not the same 100-row eval re-scored.
4. **The CI on `top_category_hit` (0.94, 1.00) is essentially the ceiling.** Even a perfect Day-3 LLM cannot meaningfully improve this metric — the templates already do it mechanically. The composite gain Day 3 can credibly target lives in `rating_mention` and `sentiment_dir_match` (their CIs straddle 0.5–0.8, leaving real headroom).

### Phase-2 win thresholds (locked in)

| Component | Metric | Day-1 point | Day-1 CI hi | **Phase-2 must exceed** |
|---|---|---|---|---|
| Sentiment | macro-F1 | 0.466 | 0.520 | **> 0.52** |
| Sentiment | Neutral F1 | 0.081 | 0.177 | **> 0.18** |
| Complaints | macro-F1 | 0.820 | 0.859 | **> 0.86** |
| Complaints | subset accuracy | 0.430 | 0.540 | **> 0.54** |
| RAG | composite | 0.686 | 0.764 | **> 0.76** |

Any Phase-2 win claim in the Day 2+ reports MUST cite these thresholds and confirm the new score exceeds the upper CI of the Day-1 baseline. This is the discipline that turns "we beat the baseline by ΔF1" from a hopeful narrative into a defensible result.
