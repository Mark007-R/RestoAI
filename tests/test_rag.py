"""RAG synthesizer tests.

We exercise the template-fallback path (no LLM/cross-encoder weights
required). Champion model behaviour is measured in scripts/day03_phase2b.py
and day06_phase5_frontier_ablation.py.
"""

from __future__ import annotations

import pytest

from src.rag.pipeline import (
    RAGAnswer,
    RAGSynthesizer,
    detect_intent,
    template_synthesize,
    _avg_rating,
    _key_terms,
    get_default,
)


SAMPLE_DOCS = [
    {"text": "The food quality was excellent and the dishes were delicious.",
     "metadata": {"rating": 4.5}},
    {"text": "Great service and the staff was friendly. Highly recommend.",
     "metadata": {"rating": 5.0}},
    {"text": "Atmosphere was good but the prices were a bit high.",
     "metadata": {"rating": 3.5}},
]


def test_detect_intent_quality():
    assert detect_intent("How is the food quality?") == "quality"


def test_detect_intent_service():
    assert detect_intent("Is the service good?") == "service"


def test_detect_intent_price():
    assert detect_intent("Is it expensive?") == "price"


def test_detect_intent_recommend():
    assert detect_intent("Would you recommend it?") == "recommend"


def test_detect_intent_unknown_returns_none():
    assert detect_intent("Random gibberish text qqq") is None


def test_detect_intent_empty():
    assert detect_intent("") is None
    assert detect_intent(None) is None  # type: ignore[arg-type]


def test_avg_rating_extracts_floats():
    assert _avg_rating(SAMPLE_DOCS) == pytest.approx((4.5 + 5.0 + 3.5) / 3)


def test_avg_rating_returns_none_when_no_ratings():
    docs = [{"text": "no rating", "metadata": {}}]
    assert _avg_rating(docs) is None


def test_avg_rating_ignores_garbage():
    docs = [{"text": "x", "metadata": {"rating": "not-a-number"}},
            {"text": "y", "metadata": {"rating": 4.0}}]
    assert _avg_rating(docs) == 4.0


def test_key_terms_returns_top_k():
    texts = ["food food food food service service ambience"]
    terms = _key_terms(texts, top_k=3)
    assert len(terms) <= 3
    assert "food" in terms


def test_key_terms_excludes_stopwords():
    terms = _key_terms(["the the the great great delicious"])
    assert "the" not in terms


def test_template_synthesize_quality_intent_mentions_rating():
    answer = template_synthesize("How is the food?", SAMPLE_DOCS, intent="quality")
    assert "rating" in answer.lower() or "/5" in answer


def test_template_synthesize_recommend_intent_strong_positive():
    docs = [
        {"text": "amazing wonderful delicious excellent perfect great love",
         "metadata": {"rating": 5.0}}
    ]
    answer = template_synthesize("recommend?", docs, intent="recommend")
    assert "RECOMMEND" in answer.upper() or "Recommended" in answer


def test_synthesize_empty_retrieved_returns_no_match():
    synth = RAGSynthesizer(flan_dir="/nonexistent", cross_encoder_dir="/nonexistent")
    out = synth.synthesize(query="anything", retrieved_docs=[])
    assert isinstance(out, RAGAnswer)
    assert out.retrieved_count == 0
    assert out.model == "empty"


def test_synthesize_uses_template_fallback_when_llm_missing(monkeypatch):
    """If the LLM loader returns None, the synthesizer must fall back to
    template synthesis (not hit the HF Hub at runtime)."""
    synth = RAGSynthesizer(flan_dir="/nonexistent", cross_encoder_dir="/nonexistent")
    monkeypatch.setattr(synth, "_get_llm", lambda: None)
    monkeypatch.setattr(synth, "_get_ce", lambda: None)
    out = synth.synthesize(query="How is the food?", retrieved_docs=SAMPLE_DOCS)
    assert out.fallback_used is True
    assert "template" in out.model
    assert out.retrieved_count == len(SAMPLE_DOCS)
    assert out.answer.strip() != ""


def test_synthesize_skips_rerank_when_cross_encoder_missing(monkeypatch):
    # rerank_k smaller than candidate set forces _rerank to run; with no CE
    # available it should still return the top-k unranked candidates.
    synth = RAGSynthesizer(flan_dir="/nonexistent", cross_encoder_dir="/nonexistent",
                           rerank_k=2)
    monkeypatch.setattr(synth, "_get_ce", lambda: None)
    monkeypatch.setattr(synth, "_get_llm", lambda: None)
    out = synth.synthesize(query="How is the food?", retrieved_docs=SAMPLE_DOCS)
    assert out.reranked is False  # CE unavailable -> rerank skipped
    assert out.retrieved_count == len(SAMPLE_DOCS)


def test_get_default_returns_singleton():
    a = get_default()
    b = get_default()
    assert a is b
