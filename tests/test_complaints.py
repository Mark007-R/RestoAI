"""Complaint classifier tests.

Verifies:
- Pydantic response schema
- Keyword fallback when the joblib bundle is unavailable
- Keyword baseline behaviour for narrow-lexicon categories (service,
  hygiene, price) where Day-6 showed keyword still wins
- Trained-head behaviour for broad-context categories when the bundle
  is loaded (delivery, food_quality, portion, variety)
- Empty / whitespace input is handled cleanly
"""

from __future__ import annotations

import os

import pytest

from src.complaints.classifier import (
    CATEGORIES,
    ComplaintClassifier,
    ComplaintPrediction,
    _keyword_predict,
    categorize_complaints_keyword,
    get_default,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(REPO_ROOT, "models", "complaints_classifier.joblib")


@pytest.fixture(scope="module")
def trained_clf():
    return ComplaintClassifier()


def test_categories_constant_matches_analyzer():
    """The CATEGORIES list must stay in sync with manager_system/analyzer.py
    so the Flask app's keyword fallback and the trained model line up."""
    import sys
    mgr = os.path.join(REPO_ROOT, "manager_system")
    if mgr not in sys.path:
        sys.path.insert(0, mgr)
    import analyzer  # noqa: WPS433 — top-level, same as the Flask app
    assert set(CATEGORIES) == set(analyzer.CATEGORY_KEYWORDS.keys())


def test_keyword_predict_service():
    cats = _keyword_predict("The waiter was rude and slow.")
    assert "service" in cats


def test_keyword_predict_hygiene():
    cats = _keyword_predict("The place was dirty and the kitchen was filthy.")
    assert "hygiene" in cats


def test_keyword_predict_price():
    cats = _keyword_predict("Way overpriced for what you get.")
    assert "price" in cats


def test_keyword_predict_returns_unique_ordered_list():
    cats = _keyword_predict("dirty dirty hygiene filthy unhygienic")
    assert cats == list(dict.fromkeys(cats))


def test_keyword_predict_empty():
    assert _keyword_predict("") == []
    assert _keyword_predict(None) == []  # type: ignore[arg-type]


def test_categorize_complaints_keyword_reexport():
    """Public re-export must match the private helper."""
    text = "Food was cold and bland."
    assert categorize_complaints_keyword(text) == _keyword_predict(text)


def test_empty_text_returns_empty_categories(trained_clf):
    out = trained_clf.predict("")
    assert isinstance(out, ComplaintPrediction)
    assert out.categories == []
    assert out.model == "empty"


def test_trained_path_when_bundle_present(trained_clf):
    if not os.path.exists(BUNDLE):
        pytest.skip("trained bundle missing — Day-4 train script not run")
    out = trained_clf.predict("The waiter was rude and the food was cold.")
    assert isinstance(out, ComplaintPrediction)
    assert out.fallback_used is False
    assert "tfidf+lightgbm" in out.model
    assert out.scores is not None
    assert set(out.scores.keys()) == set(CATEGORIES)
    # service or food_quality should fire on this text
    assert any(c in out.categories for c in ("service", "food_quality"))


def test_fallback_when_bundle_missing(monkeypatch):
    clf = ComplaintClassifier(bundle_path="/nonexistent/path/model.joblib")
    out = clf.predict("Food was cold and stale.")
    assert out.fallback_used is True
    assert "keyword" in out.model
    # keyword path picks up "cold" and "stale" -> food_quality
    assert "food_quality" in out.categories


def test_categorize_convenience_returns_list_only(trained_clf):
    cats = trained_clf.categorize("Delivery arrived late and the food was cold.")
    assert isinstance(cats, list)
    assert all(isinstance(c, str) for c in cats)
    assert all(c in CATEGORIES for c in cats)


def test_blend_with_keyword_recovers_keyword_recall():
    """The blended classifier should never *lose* a category that the
    keyword baseline would have caught — that's the contract Day-4
    introduced to preserve subset_acc on the gold-favoured eval set."""
    clf = ComplaintClassifier(blend_with_keyword=True)
    text = "The place was dirty and the staff was rude."
    out = clf.predict(text)
    kw = _keyword_predict(text)
    for c in kw:
        assert c in out.categories, f"blend dropped keyword-hit category {c}"


def test_get_default_returns_singleton():
    a = get_default()
    b = get_default()
    assert a is b


def test_all_categories_valid_set():
    assert set(CATEGORIES) == {
        "service", "food_quality", "hygiene", "price",
        "delivery", "portion", "ambience", "variety",
    }
