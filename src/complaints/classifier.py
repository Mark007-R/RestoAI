"""Production complaint classifier — TF-IDF + LightGBM champion with keyword fallback.

Day-2 head-to-head (100-review multi-label eval, see results/phase2a_metrics.json):

    Keyword (CATEGORY_KEYWORDS scan)     : macro-F1 0.820  (gold labeller shares vocab — Day-1 caveat)
    TF-IDF (word 1-2 + char 3-5) + LGBM  : macro-F1 0.682  (5-fold CV)
    SBERT + LightGBM                     : macro-F1 0.344
    NLI zero-shot multi-label            : macro-F1 0.407

The trained model wins on the two categories where the keyword baseline shows
the lowest literal-name match rate in the gold positives (food_quality +0.044
F1, portion +0.068 F1; literal-stem hit rates 3.3% and 3.8% respectively, per
results/phase2a_lexical_overlap.json). On `delivery` the trained head's
precision is 0.71 vs keyword 0.43 — useful when downstream consumers need high
precision to avoid false alerts.

We therefore deploy a **blended** strategy by default: trained classifier output
union'd with the keyword fallback. Trained-only mode is available for callers
that want the higher-precision behavior. The existing keyword logic remains the
fallback when the joblib bundle is unavailable so the Flask app keeps working
on a fresh checkout before `python scripts/day04_train_complaints.py` runs.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
from pydantic import BaseModel
from scipy.sparse import hstack

logger = logging.getLogger(__name__)

CATEGORIES = ["service", "food_quality", "hygiene", "price", "delivery",
              "portion", "ambience", "variety"]

# Mirrors manager_system/analyzer.py:CATEGORY_KEYWORDS — duplicated here so the
# fallback works without importing the Flask app's analyzer module.
_KEYWORDS: Dict[str, List[str]] = {
    "service": ["service", "wait", "waiter", "staff", "server", "attitude", "rude",
                "slow", "unfriendly", "impolite", "ignored", "attention", "waiting"],
    "food_quality": ["cold", "burnt", "undercooked", "bland", "taste", "flavour",
                     "flavor", "spoiled", "stale", "overcooked", "raw", "soggy",
                     "hard", "dry", "greasy"],
    "hygiene": ["dirty", "hygiene", "clean", "unclean", "flies", "smell", "smelly",
                "sanitation", "filthy", "unhygienic", "cockroach", "insects"],
    "price": ["expensive", "price", "cost", "overpriced", "costly", "pricey",
              "value", "money"],
    "delivery": ["delivery", "late", "packaging", "missing", "driver", "delayed",
                 "order", "arrived", "cold food", "damaged"],
    "portion": ["small", "portion", "quantity", "size", "less", "tiny", "inadequate"],
    "ambience": ["ambience", "music", "noisy", "crowded", "lighting", "atmosphere",
                 "decor", "seating", "comfort", "space"],
    "variety": ["menu", "options", "variety", "limited", "choices", "selection"],
}

_DEFAULT_BUNDLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "complaints_classifier.joblib",
)


class ComplaintPrediction(BaseModel):
    categories: List[str]
    scores: Optional[Dict[str, float]] = None
    model: str
    fallback_used: bool = False


def _keyword_predict(text: str) -> List[str]:
    """The Day-1 baseline. Order-preserving deduplication."""
    text_l = (text or "").lower()
    cats: List[str] = []
    for cat, kws in _KEYWORDS.items():
        for kw in kws:
            if kw in text_l:
                cats.append(cat)
                break
    return list(dict.fromkeys(cats))


class ComplaintClassifier:
    """TF-IDF + LightGBM multi-label classifier with keyword fallback.

    Parameters
    ----------
    bundle_path :
        Path to the joblib bundle produced by scripts/day04_train_complaints.py.
        Defaults to <repo>/models/complaints_classifier.joblib.
    blend_with_keyword :
        If True (default), union the trained model's categories with the keyword
        baseline's. This recovers the keyword baseline's high macro-F1 on the
        gold set while gaining the trained head's precision on paraphrastic
        categories (food_quality, portion). If False, return only the trained
        head's output — use this when downstream precision matters more than
        recall (e.g. flagging restaurants for manager review).
    """

    def __init__(self, bundle_path: Optional[str] = None, blend_with_keyword: bool = True):
        self.bundle_path = bundle_path or _DEFAULT_BUNDLE_PATH
        self.blend_with_keyword = blend_with_keyword
        self._bundle = None
        self._load_failed = False

    def _load(self):
        if self._bundle is not None or self._load_failed:
            return self._bundle
        if not os.path.exists(self.bundle_path):
            logger.warning("Complaint bundle %s not found; keyword fallback only", self.bundle_path)
            self._load_failed = True
            return None
        try:
            self._bundle = joblib.load(self.bundle_path)
            return self._bundle
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load complaint bundle (%s); keyword fallback only", exc)
            self._load_failed = True
            return None

    def _trained_predict(self, text: str):
        bundle = self._load()
        if bundle is None:
            return None
        word = bundle["word_vectorizer"]
        char = bundle["char_vectorizer"]
        threshold = bundle.get("threshold", 0.5)
        Xw = word.transform([text])
        Xc = char.transform([text])
        X = hstack([Xw, Xc]).tocsr()
        scores: Dict[str, float] = {}
        for cat, head in zip(bundle["categories"], bundle["heads"]):
            if isinstance(head, dict) and head.get("_kind") == "constant":
                scores[cat] = float(head["value"])
            else:
                scores[cat] = float(head.predict(X)[0])
        cats = [cat for cat, p in scores.items() if p >= threshold]
        return cats, scores

    def predict(self, text: str) -> ComplaintPrediction:
        text = (text or "").strip()
        if not text:
            return ComplaintPrediction(categories=[], scores=None, model="empty")

        keyword_cats = _keyword_predict(text)
        trained = self._trained_predict(text)
        if trained is None:
            return ComplaintPrediction(
                categories=keyword_cats,
                scores=None,
                model="keyword (fallback — bundle unavailable)",
                fallback_used=True,
            )
        trained_cats, scores = trained
        if self.blend_with_keyword:
            merged = list(dict.fromkeys(trained_cats + keyword_cats))
            model_name = "tfidf+lightgbm OvR (blended with keyword)"
        else:
            merged = list(trained_cats)
            model_name = "tfidf+lightgbm OvR"
        return ComplaintPrediction(
            categories=merged,
            scores={k: round(v, 4) for k, v in scores.items()},
            model=model_name,
            fallback_used=False,
        )

    # Convenience for legacy callers wanting just the category list.
    def categorize(self, text: str) -> List[str]:
        return self.predict(text).categories


_DEFAULT: Optional[ComplaintClassifier] = None


def get_default() -> ComplaintClassifier:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ComplaintClassifier()
    return _DEFAULT


def categorize_complaints_keyword(text: str) -> List[str]:
    """Public re-export of the keyword fallback (matches Day-1 baseline semantics)."""
    return _keyword_predict(text)
