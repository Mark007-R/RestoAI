"""Production sentiment classifier — NLI zero-shot champion with VADER fallback.

Day-2 head-to-head (200-review eval, see results/phase2a_metrics.json):

    NLI zero-shot (distilbart-mnli-12-3) : macro-F1 0.701, Neutral F1 0.478
    DistilBERT SST-2 (binary -> 3-class) : macro-F1 0.536, Neutral F1 0.113
    VADER (Day-1 baseline)               : macro-F1 0.466, Neutral F1 0.081

The NLI hypothesis template makes "neutral" a first-class option rather than
inferring it from a compound-score corridor — the +0.397 Neutral F1 lift over
VADER is the dominant contributor to the macro-F1 win.

Fallback design
---------------
NLI requires ~600ms/review on CPU and the model weights are ~330MB. If the model
fails to load (offline node, network issue, OOM) the classifier falls back to
the existing VADER path so the API never returns a 500 for an availability bug.
The response includes `fallback_used=True` so callers can downgrade behavior
(e.g. surface the keyword-only complaint path, or warn the user).
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)

NLI_MODEL_ID = "valhalla/distilbart-mnli-12-3"
LABELS = ["Positive", "Neutral", "Negative"]
_HYPOTHESIS = "The sentiment of this restaurant review is {}."

# Lightweight stopword + token regex used for keyword extraction. We deliberately
# don't pull nltk here so the classifier can be used in environments without the
# NLTK corpus downloaded.
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "with", "this",
    "that", "have", "has", "was", "were", "from", "they", "their", "them", "our",
    "very", "really", "just", "like", "also", "into", "your", "what", "when",
    "there", "here", "been", "being", "would", "could", "should",
}
_TOKEN = re.compile(r"\b[a-z]{4,}\b")


class SentimentPrediction(BaseModel):
    label: str
    compound: float
    keywords: List[str]
    model: str
    fallback_used: bool = False


def _extract_keywords(text: str, top_k: int = 8) -> List[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
    return [w for w, _ in Counter(words).most_common(top_k)]


class SentimentClassifier:
    """Singleton-style classifier with lazy NLI model load."""

    def __init__(self, prefer_nli: bool = True):
        self._prefer_nli = prefer_nli
        self._nli = None
        self._nli_load_failed = False
        self._vader = None

    # -- backends ---------------------------------------------------------
    def _get_nli(self):
        if self._nli is not None or self._nli_load_failed:
            return self._nli
        try:
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            from transformers import pipeline
            logger.info("Loading NLI sentiment model %s ...", NLI_MODEL_ID)
            self._nli = pipeline(
                "zero-shot-classification",
                model=NLI_MODEL_ID,
                device=-1,
            )
            return self._nli
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLI sentiment model unavailable (%s); using VADER fallback", exc)
            self._nli_load_failed = True
            return None

    def _get_vader(self):
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    # -- inference --------------------------------------------------------
    def _nli_predict(self, text: str) -> Optional[Tuple[str, float]]:
        clf = self._get_nli()
        if clf is None:
            return None
        candidate = [lab.lower() for lab in LABELS]
        out = clf(text[:2000], candidate_labels=candidate, hypothesis_template=_HYPOTHESIS)
        label_raw = out["labels"][0]
        score = float(out["scores"][0])
        label = label_raw.capitalize() if label_raw.lower() in candidate else "Neutral"
        # Map the NLI top-1 confidence to a [-1, 1] "compound" so downstream
        # callers (analyzer.analyze_text_and_keywords) keep working unchanged.
        if label == "Positive":
            compound = score
        elif label == "Negative":
            compound = -score
        else:
            compound = 0.0
        return label, compound

    def _vader_predict(self, text: str) -> Tuple[str, float]:
        compound = float(self._get_vader().polarity_scores(text).get("compound", 0.0))
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return label, compound

    def predict(self, text: str) -> SentimentPrediction:
        text = (text or "").strip()
        if not text:
            return SentimentPrediction(label="Neutral", compound=0.0, keywords=[],
                                       model="empty", fallback_used=False)

        used_fallback = False
        if self._prefer_nli:
            nli = self._nli_predict(text)
            if nli is not None:
                label, compound = nli
                model_name = NLI_MODEL_ID
            else:
                label, compound = self._vader_predict(text)
                model_name = "vaderSentiment"
                used_fallback = True
        else:
            label, compound = self._vader_predict(text)
            model_name = "vaderSentiment"

        return SentimentPrediction(
            label=label,
            compound=compound,
            keywords=_extract_keywords(text),
            model=model_name,
            fallback_used=used_fallback,
        )


_DEFAULT: Optional[SentimentClassifier] = None


def get_default() -> SentimentClassifier:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SentimentClassifier(prefer_nli=True)
    return _DEFAULT
