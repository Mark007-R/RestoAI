# Restaurant-Intelligence-Platform

**A restaurant-intelligence platform with ML-backed NLP — sentiment, complaint classification, and retrieval-augmented chat.**

A dual system: a **manager dashboard** (Flask, port 5000) for analytics, complaint triage, and RAG-powered chat over customer reviews, and a **customer app** for restaurant discovery and booking, with a FastAPI + Redis serving layer behind them.

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

# Tests
python -m pytest tests/ -v
```

---

## API

`POST /sentiment` · `POST /complaints` · `POST /rag` — all async, Pydantic-validated.
`GET /health`, `GET /health/cache`, `GET /metrics/ragas` — liveness plus live RAGAS-proxy means and latency percentiles.

The Flask `/manager/model-ops` page surfaces the model card, live RAGAS-proxy means, latency percentiles, and RAG cache state.

---

## Datasets

Five public review CSVs in `datasets/` (~248k rows total, ~23.8k unique reviews after dedup): `zomato.csv`, `zomato2.csv`, `mumbaires.csv`, `Resreviews.csv`, `reviews.csv`. Eval sets are sampled stratified across all five.

---

## License

MIT. See [LICENSE](LICENSE).
