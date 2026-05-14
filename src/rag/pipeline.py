"""Production RAG synthesis — flan-t5-base + cross-encoder rerank champion.

Day-3 head-to-head (50-QA eval, RAGAS proxy composite, see results/phase2b_metrics.json):

    Template baseline                           : 0.680  (length-inflated relevancy, see Day-3 finding 3)
    LLM (flan-t5-base) + recursive chunks       : 0.668
    LLM + cross-encoder rerank                  : 0.663   <- selected champion
    LLM + existing chunks                       : 0.653

The four configs land within 0.027 of each other on composite. We pick **rerank**
as the production path because:

  1. Cross-encoder rerank gives the cleanest measurable retrieval-side lift
     (ctx_recall 0.740 -> 0.760).
  2. It's an architectural insert *after* the existing FAISS retrieval, not a
     replacement — the existing per-restaurant vector store keeps doing its
     job, the reranker just picks better top-K from a larger candidate pool.
  3. The recursive-chunks alternative requires re-embedding chunks per
     restaurant at query time; rerank reuses the FAISS-cached embeddings.

When the LLM or cross-encoder cannot load (offline node, missing weights), the
synthesizer falls back to the existing template `if/elif` summary. The full
template logic lives in this module under `template_synthesize` rather than
being imported from rag_chat.py — that lets the synthesizer be used standalone
by the FastAPI service without dragging the Flask app's RAGChat dependencies in.

Pre-cached HuggingFace weights (set via env vars if non-default):
    FLAN_T5_BASE_DIR        default: ~/.cache/huggingface/local/flan-t5-base
    MS_MARCO_CE_DIR         default: ~/.cache/huggingface/local/ce-ms-marco
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_FLAN_DIR = os.environ.get(
    "FLAN_T5_BASE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "local", "flan-t5-base"),
)
DEFAULT_CE_DIR = os.environ.get(
    "MS_MARCO_CE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "local", "ce-ms-marco"),
)

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "quality": ["quality", "taste", "food", "delicious", "flavor"],
    "service": ["service", "staff", "waiter", "wait", "server"],
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "hygiene": ["clean", "hygiene", "dirty", "sanitize"],
    "ambience": ["ambience", "atmosphere", "decor", "vibe"],
    "recommend": ["recommend", "suggest", "best", "should", "worth"],
}


class RAGAnswer(BaseModel):
    answer: str
    intent: Optional[str] = None
    model: str
    reranked: bool = False
    fallback_used: bool = False
    retrieved_count: int = 0


def detect_intent(query: str) -> Optional[str]:
    q = (query or "").lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return intent
    return None


# ---------------------------------------------------------------------------
# Template fallback — verbatim port of rag_chat._synthesize_intelligent_answer's
# template logic. Kept here so the synthesizer is self-contained.
# ---------------------------------------------------------------------------
_POSITIVE = ['good', 'great', 'excellent', 'amazing', 'delicious', 'perfect',
             'wonderful', 'fantastic', 'love', 'best', 'awesome', 'outstanding']
_NEGATIVE = ['bad', 'poor', 'terrible', 'horrible', 'awful', 'worst',
             'disappointing', 'waste', 'avoid', 'never', 'disgusting', 'pathetic']
_TERM = re.compile(r"\b[a-z]{4,}\b")
_TERM_STOP = {'the', 'and', 'this', 'that', 'with', 'from', 'have', 'been',
              'were', 'will', 'would', 'could', 'they', 'them', 'their', 'just',
              'like', 'good', 'bad', 'very', 'really', 'also'}


def _avg_rating(retrieved_docs: List[dict]) -> Optional[float]:
    vals = []
    for d in retrieved_docs:
        r = (d.get("metadata") or {}).get("rating")
        try:
            if r is not None and str(r).replace('.', '').isdigit():
                vals.append(float(r))
        except (TypeError, ValueError):
            continue
    return sum(vals) / len(vals) if vals else None


def _key_terms(texts: List[str], top_k: int = 6) -> List[str]:
    counts: Dict[str, int] = {}
    for t in texts:
        for w in _TERM.findall(t.lower()):
            if w in _TERM_STOP:
                continue
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_k]]


def template_synthesize(query: str, retrieved_docs: List[dict], intent: Optional[str]) -> str:
    """Day-1 template synthesis logic (the fallback)."""
    all_text = " ".join(d["text"].lower() for d in retrieved_docs)
    pos = sum(all_text.count(w) for w in _POSITIVE)
    neg = sum(all_text.count(w) for w in _NEGATIVE)
    avg = _avg_rating(retrieved_docs)
    key = _key_terms([d["text"] for d in retrieved_docs])

    if intent == 'quality':
        if pos > neg * 1.5:
            s = " Food quality is highly praised by customers. "
        elif pos > neg:
            s = " Generally good food quality with some positive mentions. "
        else:
            s = " Mixed reviews about food quality - check specifics. "
        if avg:
            s += f"Average rating: **{avg:.1f}/5**. "
        s += f"\n   Key mentions: {', '.join(key[:5])}"
        return s
    if intent == 'service':
        if 'slow' in all_text or 'long wait' in all_text:
            return " Service speed is a concern mentioned by multiple customers. "
        if 'rude' in all_text or 'unfriendly' in all_text:
            return " Staff attitude needs improvement according to reviews. "
        return " Service is appreciated by most customers. " if pos > neg \
            else " Service quality varies - mixed experiences reported. "
    if intent == 'price':
        if 'expensive' in all_text or 'overpriced' in all_text:
            return " Prices are on the higher side as per customer feedback. "
        if 'value' in all_text and pos > neg:
            return " Good value for money mentioned by customers. "
        return " Pricing is considered reasonable by most reviewers. "
    if intent == 'hygiene':
        return " Hygiene concerns raised - cleanliness needs attention. " if neg > pos \
            else " Cleanliness standards are maintained well. "
    if intent == 'ambience':
        return " Good ambience and atmosphere appreciated by visitors. " if pos > neg \
            else " Ambience feedback is mixed - personal preference varies. "
    if intent == 'recommend':
        ratio = pos / (neg + 1)
        if ratio > 2.0 and (avg is None or avg >= 4.0):
            s = " HIGHLY RECOMMENDED! Strong positive feedback across reviews. "
        elif ratio > 1.2:
            s = " Generally Recommended with mostly positive experiences. "
        elif ratio > 0.8:
            s = " Mixed Reviews - read details before deciding. "
        else:
            s = " Caution Advised - significant negative feedback present. "
        if avg:
            s += f"\n   Overall Rating: **{avg:.1f}/5**"
        return s
    if pos > neg * 1.5:
        s = " Overall positive sentiment in reviews. "
    elif pos > neg:
        s = " Mostly positive feedback with some concerns. "
    else:
        s = " Mixed or negative feedback - proceed with caution. "
    s += f"\n   Frequently mentioned: {', '.join(key[:6])}"
    return s


# ---------------------------------------------------------------------------
# Champion: flan-t5-base + ms-marco cross-encoder rerank
# ---------------------------------------------------------------------------
class RAGSynthesizer:
    """LLM-backed RAG synthesizer with cross-encoder rerank and template fallback.

    Lazy-loads both flan-t5-base (~990 MB) and the ms-marco cross-encoder
    (~90 MB) on first use. If either fails to load, the synthesizer degrades
    gracefully: rerank is skipped when the cross-encoder is unavailable; the
    answer falls back to template synthesis when the LLM is unavailable.
    """

    def __init__(
        self,
        flan_dir: Optional[str] = None,
        cross_encoder_dir: Optional[str] = None,
        rerank_k: int = 5,
        max_new_tokens: int = 120,
        min_length: int = 40,
    ):
        self.flan_dir = flan_dir or DEFAULT_FLAN_DIR
        self.cross_encoder_dir = cross_encoder_dir or DEFAULT_CE_DIR
        self.rerank_k = rerank_k
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self._llm = None
        self._llm_failed = False
        self._ce = None
        self._ce_failed = False

    # -- backends ---------------------------------------------------------
    def _get_llm(self):
        if self._llm is not None or self._llm_failed:
            return self._llm
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            src = self.flan_dir if os.path.isdir(self.flan_dir) else "google/flan-t5-base"
            logger.info("Loading flan-t5-base from %s ...", src)
            tok = AutoTokenizer.from_pretrained(src)
            mdl = T5ForConditionalGeneration.from_pretrained(src)
            mdl.eval()
            self._llm = (tok, mdl, torch)
            return self._llm
        except Exception as exc:  # noqa: BLE001
            logger.warning("flan-t5-base unavailable (%s); template fallback only", exc)
            self._llm_failed = True
            return None

    def _get_ce(self):
        if self._ce is not None or self._ce_failed:
            return self._ce
        try:
            from sentence_transformers import CrossEncoder
            src = self.cross_encoder_dir if os.path.isdir(self.cross_encoder_dir) \
                else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info("Loading cross-encoder from %s ...", src)
            self._ce = CrossEncoder(src)
            return self._ce
        except Exception as exc:  # noqa: BLE001
            logger.warning("cross-encoder unavailable (%s); skipping rerank", exc)
            self._ce_failed = True
            return None

    # -- pipeline ---------------------------------------------------------
    def _rerank(self, query: str, candidates: List[dict]) -> Tuple[List[dict], bool]:
        if len(candidates) <= self.rerank_k:
            return candidates, False
        ce = self._get_ce()
        if ce is None:
            return candidates[: self.rerank_k], False
        pairs = [[query, c["text"]] for c in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
        out: List[dict] = []
        for c, s in ranked[: self.rerank_k]:
            c2 = dict(c)
            c2["rerank_score"] = float(s)
            out.append(c2)
        return out, True

    def _llm_synthesize(self, query: str, restaurant: Optional[str],
                        retrieved: List[dict]) -> Optional[str]:
        llm = self._get_llm()
        if llm is None:
            return None
        tok, mdl, torch = llm
        chunks_block = "\n".join(f"- {d['text'][:280]}" for d in retrieved[:5])
        rest = restaurant or "this restaurant"
        prompt = (
            f"Task: Read the customer reviews about {rest} below and write a "
            f"3-sentence answer to the question.\n\n"
            f"Question: {query}\n\n"
            f"Customer reviews:\n{chunks_block}\n\n"
            f"3-sentence answer (mention food, service, price, ambience, portion etc. "
            f"as relevant; do not quote a single review):"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_length=self.min_length,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        return tok.decode(out[0], skip_special_tokens=True).strip()

    def synthesize(
        self,
        query: str,
        retrieved_docs: List[dict],
        intent: Optional[str] = None,
        restaurant: Optional[str] = None,
    ) -> RAGAnswer:
        if intent is None:
            intent = detect_intent(query)
        n_in = len(retrieved_docs)
        if n_in == 0:
            return RAGAnswer(
                answer=" No matching reviews found in the index. ",
                intent=intent, model="empty", reranked=False, fallback_used=False,
                retrieved_count=0,
            )

        ranked, reranked = self._rerank(query, retrieved_docs)
        llm_text = self._llm_synthesize(query, restaurant, ranked)
        if llm_text:
            return RAGAnswer(
                answer=llm_text,
                intent=intent,
                model="flan-t5-base + ms-marco rerank" if reranked else "flan-t5-base",
                reranked=reranked,
                fallback_used=False,
                retrieved_count=n_in,
            )
        # LLM unavailable -> template fallback
        return RAGAnswer(
            answer=template_synthesize(query, ranked, intent),
            intent=intent,
            model="template (fallback — LLM unavailable)",
            reranked=reranked,
            fallback_used=True,
            retrieved_count=n_in,
        )


_DEFAULT: Optional[RAGSynthesizer] = None


def get_default() -> RAGSynthesizer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RAGSynthesizer()
    return _DEFAULT
