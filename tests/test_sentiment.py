"""Sentiment classifier tests.

We use `prefer_nli=False` so VADER drives the predictions — the NLI
download (~330MB) isn't required for the test suite. The schema and
fallback contract are the things we verify here. The NLI quality
numbers are measured separately in scripts/day02_phase2a.py.
"""

from __future__ import annotations

import pytest

from src.sentiment.classifier import (
    LABELS,
    SentimentClassifier,
    SentimentPrediction,
    _extract_keywords,
    get_default,
)


@pytest.fixture(scope="module")
def vader_clf():
    return SentimentClassifier(prefer_nli=False)


def test_predict_returns_pydantic_schema(vader_clf, sample_reviews):
    out = vader_clf.predict(sample_reviews["positive_food"])
    assert isinstance(out, SentimentPrediction)
    assert out.label in LABELS
    assert -1.0 <= out.compound <= 1.0
    assert isinstance(out.keywords, list)
    assert out.model == "vaderSentiment"
    assert out.fallback_used is False  # VADER is the requested backend


def test_predict_positive_review(vader_clf, sample_reviews):
    out = vader_clf.predict(sample_reviews["positive_food"])
    assert out.label == "Positive"
    assert out.compound > 0


def test_predict_negative_review(vader_clf, sample_reviews):
    out = vader_clf.predict(sample_reviews["negative_service"])
    assert out.label == "Negative"
    assert out.compound < 0


def test_predict_empty_text(vader_clf, sample_reviews):
    out = vader_clf.predict(sample_reviews["empty"])
    assert out.label == "Neutral"
    assert out.compound == 0.0
    assert out.keywords == []
    assert out.model == "empty"


def test_predict_whitespace_only(vader_clf, sample_reviews):
    out = vader_clf.predict(sample_reviews["whitespace"])
    assert out.model == "empty"


def test_keywords_excludes_stopwords():
    kws = _extract_keywords("the food and the staff were really really good")
    assert "the" not in kws
    assert "really" not in kws
    # length-filter: only 4+ char tokens
    for k in kws:
        assert len(k) >= 4


def test_keywords_topk_limit():
    text = " ".join(f"word{i}" for i in range(50))
    kws = _extract_keywords(text, top_k=8)
    assert len(kws) <= 8


def test_nli_fallback_path_marks_flag(monkeypatch):
    """If NLI is preferred but unavailable, fallback_used must be True
    and the response must still return a valid SentimentPrediction."""
    clf = SentimentClassifier(prefer_nli=True)
    # Force the NLI loader to return None.
    monkeypatch.setattr(clf, "_get_nli", lambda: None)
    clf._nli_load_failed = True
    out = clf.predict("Food was great")
    assert out.fallback_used is True
    assert out.model == "vaderSentiment"
    assert out.label in LABELS


def test_get_default_returns_singleton():
    a = get_default()
    b = get_default()
    assert a is b


def test_long_text_does_not_crash(vader_clf):
    long = "This is a really long review. " * 200
    out = vader_clf.predict(long)
    assert out.label in LABELS


def test_unicode_text(vader_clf):
    out = vader_clf.predict("Café was great — pão de queijo was délicieux")
    assert out.label in LABELS
