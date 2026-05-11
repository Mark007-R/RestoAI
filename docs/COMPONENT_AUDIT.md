# RestoAI Component Audit — Day 1, Phase 1

**Date:** 2026-05-11
**Auditor:** Sprint Day 1 (RestoAI Production Upgrade)
**Scope:** `manager_system/analyzer.py`, `manager_system/rag_chat.py`
**Purpose:** Pin down what is currently shipped vs what is *claimed*. Baselines built later in this report are measured against the code below — not against a re-implementation.

---

## TL;DR — three honest findings

1. **Sentiment IS a real model.** `analyze_text_and_keywords` calls VADER (rule-based, lexicon + valence shifters). Defensible baseline.
2. **Complaint classifier IS NOT a model.** `categorize_complaints` is a substring scan against a hand-edited dictionary of 8 categories × ~6–15 keywords. No training, no learning, no probabilities. It only fires when reviewers use the *exact* listed words.
3. **RAG synthesis IS NOT an LLM call.** `_synthesize_intelligent_answer` routes via a keyword-detected "intent" into one of ~6 hardcoded `if/elif` template strings. The retrieval (FAISS + SBERT) is genuine; the *answer generation* is a deterministic templating function with zero generative model behind it.

These three are exactly the gaps the 7-day sprint will close (Phase 2 → 6).

---

## 1. Sentiment — `analyzer.py`

### `analyze_text_and_keywords(text)` — lines 51–63
- **Engine:** `vaderSentiment.SentimentIntensityAnalyzer().polarity_scores(text)`
- **Mapping:** compound ≥ 0.05 → Positive · compound ≤ -0.05 → Negative · else Neutral
- **Side output:** `extract_keywords(text, top_k=8)` — lowercase, alpha-only ≥4 chars, NLTK stopwords removed, top-8 by frequency.
- **Status:** Real model. CPU, deterministic, fast. Acceptable baseline for the 200-review eval.
- **Known limitations:**
  - Lexicon-based; misses sarcasm, code-mixed Hindi-English ("kharaab", "tasty hai"), and domain idioms ("paisa vasool" is positive but VADER-neutral).
  - Threshold 0.05 is the VADER author's recommendation — not tuned to the restaurant-review domain.

### `extract_keywords(text, top_k=8)` — lines 44–49
- Frequency-based, no TF-IDF, no n-grams. Used only for display.

---

## 2. Complaint classifier — `analyzer.py`

### `categorize_complaints(text)` — lines 65–73
```python
def categorize_complaints(text):
    text_l = text.lower()
    cats = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text_l:
                cats.append(cat)
                break
    return list(dict.fromkeys(cats))
```
- **Engine:** plain `in` substring match. NOT tokenised, NOT stemmed, NOT a model.
- **Categories (8):** service, food_quality, hygiene, price, delivery, portion, ambience, variety.
- **Dictionary:** `CATEGORY_KEYWORDS` at lines 25–39 (~80 keywords total, hand-curated).
- **Failure modes** (will show in baseline):
  - **No semantic recall.** Review "the chef clearly didn't bother today, everything was just tasteless mush" → fires `food_quality` only via "taste"-substring matching `taste`-in-`tasteless`. Random luck.
  - **False positives from substring matching.** "service" appears inside "self-service", "disservice", "serviceable" → fires `service`. "small" fires `portion` even when describing decor ("small cozy room").
  - **Multi-label is shallow.** Multi-category review only flagged if each category's keywords literally appear; misses paraphrase.
  - **Rare classes (Hygiene, Portion) are most underserved** — these get the smallest keyword lists and the rarest exact matches.
- **Resume claim being repaired:** "trained complaint classifier" — currently false. Phase 2a (Day 2) compares against TF-IDF+LightGBM and SBERT+LightGBM and Claude zero-shot.

### `CATEGORY_KEYWORDS` — lines 25–39
All 8 entries are flat substring lists. No regex, no boundaries. (The Phase-3 fallback path will keep this for offline mode, behind the trained model.)

---

## 3. Other pieces of `analyzer.py`

| Function | Lines | Status | Touched in sprint? |
|---|---|---|---|
| `extract_keywords` | 44–49 | Frequency-only, fine for display | No |
| `plot_to_base64` | 75–81 | Utility | No |
| `get_restaurant_info` | 83+ | Multi-CSV restaurant lookup | No (read-only) |
| `generate_visualizations` | (~mid file) | ~900 lines, 18+ matplotlib charts | No (production refactor not in scope) |
| `summarize_reviews_for_recommendations` | (late) | Rule-based business recs | Possibly Day 7 read-through; not core |

---

## 4. RAG — `rag_chat.py`

### Architecture (real)
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, cosine via L2-normalised `IndexFlatIP`.
- **Index:** consolidated `manager_system/vector_db/all_restaurants.faiss` + pickled metadata; appended-to per restaurant.
- **Loaders:** 5 dataset-specific functions (`load_mumbaires_csv`, `load_resreviews_csv`, `load_reviews_csv`, `load_zomato_csv`, `load_zomato2_csv`) feeding `index_documents()`.
- **Retrieval:** `semantic_search(query, top_k=5, restaurant_filter=…)` — searches `top_k * 3`, then post-filters by restaurant.
- **Fallback:** `_search_google_fallback` scrapes Google snippets when FAISS retrieval is empty (works only when Google's HTML schema holds; brittle but out of scope).

**Verdict on retrieval:** legitimate dense retrieval. It is the *generation* step that's fake.

### `_generate_answer` — lines 480–514
- Detects intent by keyword scan (`quality / service / price / hygiene / ambience / recommend`).
- Builds the visible prefix: "Based on N relevant reviews about **X**:" + top-K snippets via `textwrap.shorten`.
- Calls `_synthesize_intelligent_answer(query, retrieved_docs, intent)` — see next section.

### `_synthesize_intelligent_answer(query, retrieved_docs, intent)` — lines 516–606 ★ THIS IS THE FAKE LLM ★
```python
positive_words = ['good', 'great', 'excellent', ...]
negative_words = ['bad', 'poor', 'terrible', ...]
pos_count = sum(all_text.count(w) for w in positive_words)
neg_count = sum(all_text.count(w) for w in negative_words)
...
if intent == 'quality':
    if pos_count > neg_count * 1.5:
        summary = " Food quality is highly praised by customers. "
    elif pos_count > neg_count:
        summary = " Generally good food quality with some positive mentions. "
    else:
        summary = " Mixed reviews about food quality - check specifics. "
```
- **There is no model call.** The "intelligent" answer is one of roughly 18 hardcoded sentences chosen by an `if/elif` ladder over a hand-tuned `pos_count vs neg_count` ratio.
- **Why it looks plausible at a glance:** the prefix lists real retrieved snippets; the appended template paragraph is generic enough that a casual reader assumes summarisation happened.
- **What this means for evaluation:** measuring "RAG faithfulness" against the templates is mostly measuring whether the template happened to align with the question. Day-1 baseline judges only `answer-only` faithfulness + relevancy (cannot meaningfully measure context-precision/recall against a template that ignores the context tokens).
- **Resume claim being repaired:** "RAG with synthesised answers" — currently the synthesis is a switch statement. Phase 2b (Day 3) replaces this with a real Claude / GPT call over retrieved passages, plus reranking.

### `_extract_key_terms` — lines 608–621
Frequency over alpha-only ≥4 char tokens. Used only for display in some of the template branches.

### `_search_google_fallback` — lines 623+
HTML scrape. Fragile, but out of scope.

---

## 5. What sprint Phases 2–7 will change

| Component | Today (Day 1) | Replacement (Phase 2) | Champion lands (Day 4) |
|---|---|---|---|
| Sentiment | VADER (real but limited) | DistilBERT-SST2 + Claude OPUS 4.6 zero-shot | Best of three in `src/sentiment/classifier.py` |
| Complaints | Substring keyword scan (NOT a model) | TF-IDF+LightGBM, SBERT+LightGBM, Claude zero-shot | Trained classifier delegated to from `categorize_complaints` (signature preserved, keyword fallback retained) |
| RAG synthesis | Hardcoded templates (NOT an LLM) | LLM synth on existing chunks; +chunking variant; +cross-encoder rerank; full RAGAS | LLM synthesis delegated to from `_synthesize_intelligent_answer` (signature preserved, template fallback retained) |

Function signatures of `categorize_complaints` and `_synthesize_intelligent_answer` MUST be preserved per Sprint Rule 13 so `app.py` and existing callers keep working without changes.

---

## 6. Files that depend on the audited functions (do not break)

```
app.py                  → imports from manager_system, indirect calls
manager_system/manager.py → calls analyzer.analyze_text_and_keywords, categorize_complaints
manager_system/rag_chat.py → uses its own internal _synthesize_intelligent_answer
templates/*              → render the strings these functions return
```
Day 4 refactor must keep return shapes identical: `(label:str, compound:float, keywords:list[str])` and `list[str]` and `str`.
