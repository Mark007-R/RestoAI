"""Regression tests for Hard Rule 5:

> When modifying `analyzer.py:categorize_complaints` or
> `rag_chat.py:_synthesize_intelligent_answer`, KEEP the existing
> function signature so the Flask app keeps working without changes
> elsewhere.

If either signature changes the Flask routes break silently. These
tests are the canary.
"""

from __future__ import annotations

import inspect
import os
import sys

# Replicate the Flask app's sys.path setup so analyzer.py's
# `from utils.helpers import ...` resolves against manager_system/utils.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MGR = os.path.join(_ROOT, "manager_system")
for p in (_MGR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyzer  # noqa: E402  (loaded as a top-level module, same as the Flask app)


def test_categorize_complaints_signature_takes_single_text_arg():
    """`categorize_complaints(text)` must remain callable with one positional
    string argument and return a list of strings."""
    sig = inspect.signature(analyzer.categorize_complaints)
    params = list(sig.parameters.values())
    # Allow >=1 positional; require the first param to take a text-like value.
    assert len(params) >= 1
    out = analyzer.categorize_complaints("The waiter was rude.")
    assert isinstance(out, list)
    assert all(isinstance(c, str) for c in out)


def test_categorize_complaints_keyword_fallback_still_works():
    """When the trained classifier is forced unavailable, categorize_complaints
    must still return a sensible category list (keyword fallback)."""
    analyzer._COMPLAINT_CLF = False  # sentinel: "tried and failed, don't retry"
    try:
        out = analyzer.categorize_complaints("The food was cold and stale.")
        # Keyword path picks up "cold"/"stale" -> food_quality
        assert "food_quality" in out
    finally:
        analyzer._COMPLAINT_CLF = None  # reset for other tests


def test_analyze_text_and_keywords_signature_preserved():
    """`analyze_text_and_keywords(text)` must keep returning (label, compound, keywords)."""
    label, compound, keywords = analyzer.analyze_text_and_keywords("The food was great.")
    assert label in {"Positive", "Negative", "Neutral"}
    assert -1.0 <= compound <= 1.0
    assert isinstance(keywords, list)


def test_category_keywords_dict_intact():
    """The original CATEGORY_KEYWORDS dict (the fallback lexicon) must still
    expose the 8 production categories."""
    assert set(analyzer.CATEGORY_KEYWORDS.keys()) == {
        "service", "food_quality", "hygiene", "price",
        "delivery", "portion", "ambience", "variety",
    }
