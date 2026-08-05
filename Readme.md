# Restaurant-Intelligence-Platform

**A restaurant-intelligence platform with measured, ML-backed NLP — sentiment, complaint classification, and retrieval-augmented chat.**

Restaurant-Intelligence-Platform is a dual system: a **manager dashboard** (Flask, port 5000) for analytics, complaint triage, and RAG-powered chat over customer reviews, and a **customer app** for restaurant discovery and booking. The interesting part of this repo is its honesty story: a 7-day upgrade sprint replaced two components that *looked* like ML but weren't, measured the replacements against real eval sets, and wrapped the champions in a production FastAPI + Redis + Docker service — keeping every existing Flask call-site working.

---

## The headline: what was "AI" vs what is now ML

A Day-1 audit found that of three "AI" components, only one was a real model:

| Component | Before the sprint | After the sprint |
|---|---|---|
| Sentiment | **VADER** (real, lexicon-based) | **NLI zero-shot** (`valhalla/distilbart-mnli-12-3`), VADER kept as fallback |
| Complaint classifier | **keyword substring scan** against an 8-category dict — *not a model* | **TF-IDF + LightGBM** (one-vs-rest, 8 binary heads), keyword kept as fallback |
| RAG "synthesis" | **hardcoded if/elif templates** — *no LLM* | **FAISS retrieval → cross-encoder rerank → flan-t5 synthesis**, templates kept as fallback |

Every champion was integrated **behind the original function signatures** (`manager_system/analyzer.py:categorize_complaints`, `manager_system/rag_chat.py:_synthesize_intelligent_answer`) so the Flask app keeps working unchanged — the trained model is the new default, with the old behavior as an offline fallback.

---

## Final metrics (measured, not claimed)

Where the specialized model *lost*, it's reported honestly.

### Sentiment (200-review eval)
| Model | Macro-F1 | Neutral F1 |
|---|---:|---:|
| VADER (baseline) | 0.466 | 0.081 |
| **NLI zero-shot (champion)** | **0.701** | **0.478** (6× lift) |

The Neutral class is where VADER collapsed — lexicon scoring pushes almost everything to positive/negative. NLI entailment recovers it.

### Complaint classifier (multi-label, 8 categories)
| Model | Macro-F1 (fresh held-out 100) |
|---|---:|
| Keyword substring (baseline) | 0.8335 |
| **TF-IDF + LightGBM (champion)** | **0.8525** |

Per-category, the trained model wins decisively where reviewers *don't* use the literal trigger word, and loses on narrow-lexicon categories — an honest split:

| Category | Champion F1 | Δ vs keyword |
|---|---:|---:|
| Delivery | 0.923 | **+0.386** |
| Portion | 0.927 | **+0.119** |
| Food quality | 0.865 | **+0.077** |
| Variety | 0.909 | +0.052 |
| Service | 0.945 | −0.044 |
| Hygiene | 0.828 | −0.030 |
| Ambience | 0.794 | −0.138 |
| Price | 0.606 | **−0.294** (keyword decisively better) |

Because of this split, the production classifier **unions** the trained output with the keyword fallback by default, preserving keyword recall on the narrow-lexicon categories. Serialized bundle: `models/complaints_classifier.joblib` (1.6 MB), tuned with a 30-trial Optuna sweep (Day 5).

### RAG (50-QA structural eval, RAGAS-proxy)
| Config | Composite | Context recall | Cold latency |
|---|---:|---:|---:|
| Template synthesis (baseline) | 0.680 | 0.655 | 5.0 s |
| **FAISS + rerank + flan-t5 (champion)** | 0.663 | **0.760** (+0.105) | 2.4 s |

Honest read: the structural composite is roughly flat, but the LLM-backed pipeline lifts **context recall** by +0.105 and halves cold latency. With the Redis cache warm, repeat queries return in **<10 ms** (240× faster than the 2.4 s cold path).

### Frontier comparison (Day 6, vs Claude / DistilBERT)
| Task | Specialized | Frontier zero-shot | Winner |
|---|---:|---:|---|
| Complaint macro-F1 | **0.850** (LightGBM) | 0.484 (Claude NLI) | Specialized (+0.366) |
| Complaint subset-accuracy | **0.43** | 0.01 | Specialized |
| Sentiment macro-F1 | **0.607** (NLI) | 0.560 (DistilBERT-SST2) | Specialized |

With ≥100 labeled examples per class on noisy review text, the specialized multi-label classifier beats frontier zero-shot — especially on subset-accuracy, where getting *all* labels right matters.

---

## Architecture

```
                 ┌───────────────────────────────────────────────┐
   Flask :5000   │  Manager dashboard + Customer app (app.py)    │
   (UI, unchanged│  analytics · complaint triage · RAG chat ·    │
    signatures)  │  /manager/model-ops live quality panel        │
                 └───────────────────────────────────────────────┘
                          │ delegates (signatures preserved)
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                       src/ champions                          │
   │  sentiment/classifier.py   NLI zero-shot  (+ VADER fallback)  │
   │  complaints/classifier.py  TF-IDF+LGBM    (+ keyword fallback)│
   │  rag/pipeline.py           FAISS→rerank→flan-t5 (+ template)  │
   │  cache/rag_cache.py        Redis → in-memory LRU              │
   │  observability/ragas_log.py  per-request RAGAS-proxy logging  │
   └──────────────────────────────────────────────────────────────┘
                          ▲
   FastAPI :8000          │  same champions, JSON transport
   (api.py)   /sentiment · /complaints · /rag · /health · /metrics/ragas
```

Two transports (Flask UI + FastAPI JSON) over one set of champion models. Both validate with Pydantic v2 at the API boundary.

---

## Quick start

```bash
# Production stack (FastAPI + Redis) via Docker
docker compose up -d
curl http://localhost:8000/health

# Flask UI (manager + customer) — separate terminal
pip install -r requirements.txt
python app.py                       # http://localhost:5000

# FastAPI service standalone (no Docker)
pip install -r requirements-api.txt
uvicorn api:app --port 8000 --reload

# Tests — 88 tests, ~13 s on CPU
python -m pytest tests/ -v
```

### Reproduce the sprint experiments
```bash
python scripts/build_eval_sets.py                 # Day 1: eval sets
python scripts/day02_phase2a.py                   # sentiment + complaint head-to-head
python scripts/day03_phase2b.py                   # RAG 4-config ablation
python scripts/day04_train_complaints.py          # train complaint classifier
python scripts/day05_phase4_tuning.py             # Optuna sweep (30 trials)
python scripts/day06_phase5_frontier_ablation.py  # fresh held-out + frontier
```

---

## API

`POST /sentiment` · `POST /complaints` · `POST /rag` — all async, Pydantic-validated.
`GET /health`, `GET /health/cache`, `GET /metrics/ragas` — liveness + live RAGAS-proxy means and latency percentiles.

The Flask `/manager/model-ops` page surfaces the champion card, live RAGAS-proxy means, latency percentiles, and RAG cache state.

---

## Tests

88 pytest tests (~12.6 s on CPU), 7 files:

| File | Tests | Covers |
|---|---:|---|
| `test_sentiment.py` | 11 | NLI + VADER fallback, schema, singleton |
| `test_complaints.py` | 14 | keyword path, trained path, blend regression |
| `test_rag.py` | 16 | intent detection, template fallback, CE skip |
| `test_caching.py` | 15 | key determinism, LRU eviction, TTL, Redis fallback |
| `test_observability.py` | 10 | RAGAS sub-scores, JSONL roundtrip |
| `test_api.py` | 16 | async endpoints, 422 validation, cache hit/miss |
| `test_signature_contracts.py` | 4 | Flask call-site signature preservation (regression canary) |

The signature-contract tests are the guarantee that the champion swap never broke the existing Flask app.

---

## Repo layout

```
Restaurant-Intelligence-Platform/
├── app.py                      # Flask entry (manager + customer UI, :5000)
├── api.py                      # FastAPI service (:8000)
├── Dockerfile, docker-compose.yml, requirements-api.txt
├── src/
│   ├── sentiment/classifier.py        # NLI zero-shot + VADER fallback
│   ├── complaints/classifier.py       # TF-IDF + LightGBM + keyword fallback
│   ├── rag/pipeline.py                # FAISS + rerank + flan-t5 + template fallback
│   ├── cache/rag_cache.py             # Redis → in-memory LRU
│   └── observability/ragas_log.py     # per-request RAGAS-proxy logger
├── manager_system/             # Flask manager UI; analyzer.py + rag_chat.py delegate to src/
│   ├── model_ops.py            # live quality dashboard
│   └── vector_db/              # consolidated FAISS index + metadata
├── user_system/, shared/, templates/
├── models/complaints_classifier.joblib
├── datasets/                   # 5 CSV review sources (~248k rows, ~23.8k unique reviews)
├── data/eval/                  # gold eval sets (sentiment / complaint / RAG QA + held-out)
├── tests/                      # 88-test suite
└── scripts/                    # day02..day06 experiment runners
```

---

## Datasets

Five public review CSVs in `datasets/` (~248k rows total, ~23.8k unique reviews after dedup): `zomato.csv`, `zomato2.csv`, `mumbaires.csv`, `Resreviews.csv`, `reviews.csv`. Eval sets sampled stratified across all five.

---

## License

MIT. See [LICENSE](LICENSE).
