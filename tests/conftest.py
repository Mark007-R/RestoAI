"""Pytest fixtures.

These tests avoid loading the heavy NLI / flan-t5 / cross-encoder weights:
- sentiment falls back to VADER when the NLI model isn't loaded (we
  construct the classifier with `prefer_nli=False`).
- complaints uses the trained joblib bundle if present and the keyword
  fallback if not — both paths are exercised.
- RAG synthesis uses the template-fallback path (no LLM weights needed)
  by pointing the synthesizer at non-existent model dirs.
- API tests use the FastAPI TestClient with the RAG endpoint stubbed via
  monkeypatch so we don't need the FAISS index either.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force in-memory cache for the whole test session.
os.environ.pop("REDIS_URL", None)


@pytest.fixture(autouse=True)
def _reset_cache():
    from src.cache import reset_default_cache, get_cache
    reset_default_cache()
    c = get_cache()
    c.clear()
    yield
    reset_default_cache()


@pytest.fixture
def tmp_ragas_log(monkeypatch, tmp_path):
    """Point the RAGAS-proxy logger at a tempfile and reset the singleton."""
    path = tmp_path / "ragas.jsonl"
    monkeypatch.setenv("RAGAS_LOG_PATH", str(path))
    # Reset module-level singleton.
    import src.observability.ragas_log as r
    r._DEFAULT = None
    yield path
    r._DEFAULT = None


@pytest.fixture
def sample_reviews():
    return {
        "positive_food": "The biryani was absolutely delicious and the staff was friendly.",
        "negative_service": "The waiter was rude and we waited 45 minutes for water.",
        "delivery_complaint": "Delivery arrived an hour late and the food was cold.",
        "hygiene_complaint": "The place was dirty and I saw flies near the kitchen.",
        "price_complaint": "Way overpriced for what you get — a small plate for 800 rupees.",
        "neutral": "Average experience, nothing special but nothing terrible either.",
        "multi": "Food was great but service was slow and the place felt cramped.",
        "empty": "",
        "whitespace": "   ",
    }
