"""Day-3 Phase-2b RAG comparison.

Compares four RAG-synthesis configurations on the 50-QA eval set built on Day 1.

Configurations
--------------
1. template_baseline   : current `_synthesize_intelligent_answer` (template if/elif).
                         Re-uses Day-1 answers from `results/baseline_rag_answers.json`.
2. llm_existing_chunks : Same retrieval (per-review chunks in FAISS, top_k=5) but
                         synthesis swapped for flan-t5-base (local seq2seq LLM).
3. llm_recursive_chunks: Re-chunk source reviews with recursive character splitting
                         (size=300, overlap=60), re-embed per-restaurant index,
                         retrieve top_k=5, synthesize with flan-t5-base.
4. llm_rerank          : Existing FAISS retrieve top_k=15 -> cross-encoder rerank
                         with ms-marco-MiniLM-L-6-v2 -> top-5 -> flan-t5-base.

Why flan-t5-base?
    The autonomous run cannot reach the Anthropic / OpenAI APIs (no API key in env;
    Day-1 LLM-judge fell back to structural metrics for the same reason). flan-t5-base
    is a genuine instruction-tuned LLM (250M params, encoder-decoder) — small enough
    to run on CPU but enough to demonstrate the gap between *template synthesis* and
    *LLM synthesis*. The Day-6 frontier comparison will measure the additional gap
    to Claude Opus when keys are available.

RAGAS proxy
-----------
With no LLM judge, we score each (question, answer, retrieved_chunks) tuple on four
RAGAS-aligned axes using local models + deterministic rules:

  - faithfulness   : avg NLI entailment of each answer sentence by the union of
                     retrieved chunks (distilbart-mnli, already cached).
  - relevancy      : SBERT cosine similarity between answer and question.
  - ctx_precision  : fraction of retrieved chunks with SBERT-sim(question, chunk) >= 0.30.
  - ctx_recall     : fraction of gold facts (top categories + sentiment direction)
                     whose terms appear in the retrieved chunks.
  - composite      : geometric mean of the four above.

We *also* re-run the Day-1 structural metrics (sentiment_dir_match, top_category_hit,
rating_mention, intent_addressed) for continuity with Day-1.

Outputs
-------
results/phase2b_answers.json          - raw answers + retrieved chunks per config
results/phase2b_results.csv           - per-question scores across all configs (long form)
results/phase2b_metrics.json          - aggregate metrics per config
results/samples/phase2b_*.csv         - top/bottom samples per config
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "manager_system"))
EVAL_DIR = os.path.join(ROOT, "data", "eval")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("phase2b")
for noisy in ("manager_system", "rag_chat", "analyzer", "sentence_transformers",
              "transformers", "faiss", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# Re-use Day-1 structural scorer
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from structural_rag_metrics import score_one as structural_score, CATEGORY_TERMS

# ---------------------------------------------------------------------------
# Models (loaded lazily once)
# ---------------------------------------------------------------------------
_MODELS: dict = {}

def get_sbert():
    if "sbert" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        print("  loading SBERT all-MiniLM-L6-v2 ...", flush=True)
        _MODELS["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODELS["sbert"]

LOCAL_FLAN_DIR = os.environ.get(
    "FLAN_T5_BASE_DIR",
    r"C:\Users\antho\.cache\huggingface\local\flan-t5-base",
)
LOCAL_CE_DIR = os.environ.get(
    "MS_MARCO_CE_DIR",
    r"C:\Users\antho\.cache\huggingface\local\ce-ms-marco",
)

def get_llm():
    if "llm" not in _MODELS:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        src = LOCAL_FLAN_DIR if os.path.isdir(LOCAL_FLAN_DIR) else "google/flan-t5-base"
        print(f"  loading flan-t5-base from {src} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(src)
        mdl = T5ForConditionalGeneration.from_pretrained(src)
        mdl.eval()
        _MODELS["llm"] = (tok, mdl, torch)
    return _MODELS["llm"]

def get_cross_encoder():
    if "ce" not in _MODELS:
        from sentence_transformers import CrossEncoder
        src = LOCAL_CE_DIR if os.path.isdir(LOCAL_CE_DIR) else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"  loading cross-encoder from {src} ...", flush=True)
        _MODELS["ce"] = CrossEncoder(src)
    return _MODELS["ce"]

def get_nli():
    if "nli" not in _MODELS:
        from transformers import pipeline
        print("  loading NLI distilbart-mnli-12-3 ...", flush=True)
        _MODELS["nli"] = pipeline("zero-shot-classification",
                                  model="valhalla/distilbart-mnli-12-3",
                                  device=-1)
    return _MODELS["nli"]

# ---------------------------------------------------------------------------
# Synthesis: LLM
# ---------------------------------------------------------------------------
def llm_synthesize(question: str, restaurant: str, retrieved_chunks: list[str],
                   max_new_tokens: int = 120) -> str:
    """Prompt flan-t5-base with retrieved chunks and return synthesized answer.

    Prompt design choices, picked empirically on 5 hand-checked smoke-test questions:
      * Question precedes the reviews — improves grounding vs putting Q after.
      * Explicit "do not quote a single review" — counters flan-t5-base's tendency
        to copy the first chunk verbatim when one chunk dominates retrieval.
      * min_length=40 — without this, the model often emits a 1-sentence response.
      * Beam search (num_beams=4) — small lift over greedy on this small model.

    Even with these, flan-t5-base will sometimes copy a chunk when only one chunk
    is retrieved or one chunk strongly dominates. This is a known limitation of a
    250M-param LLM on this synthesis task; the Day-6 frontier rerun with Claude
    Opus 4.6 will quantify the gap to a much larger model.
    """
    tok, mdl, torch = get_llm()
    chunks_block = "\n".join(f"- {c[:280]}" for c in retrieved_chunks[:5])
    prompt = (
        f"Task: Read the customer reviews about {restaurant} below and write a "
        f"3-sentence answer to the question.\n\n"
        f"Question: {question}\n\n"
        f"Customer reviews:\n{chunks_block}\n\n"
        f"3-sentence answer (mention food, service, price, ambience, portion etc. "
        f"as relevant; do not quote a single review):"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_length=40,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    return text

# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def retrieve_existing(rag, question: str, restaurant: str, top_k: int) -> list[dict]:
    docs, scores = rag.semantic_search(question, top_k=top_k, restaurant_filter=restaurant)
    return [{"text": d["text"], "metadata": d.get("metadata", {}), "score": float(s)}
            for d, s in zip(docs, scores)]

def rerank_cross_encoder(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    ce = get_cross_encoder()
    pairs = [[question, c["text"]] for c in candidates]
    rerank_scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: -float(x[1]))
    out = []
    for c, s in ranked[:top_k]:
        c2 = dict(c)
        c2["rerank_score"] = float(s)
        out.append(c2)
    return out

# ---------------------------------------------------------------------------
# Recursive character splitter (lightweight, no langchain dep)
# ---------------------------------------------------------------------------
def recursive_char_split(text: str, chunk_size: int = 300, overlap: int = 60) -> list[str]:
    """Split text into ~chunk_size character chunks with overlap, on natural boundaries."""
    if len(text) <= chunk_size:
        return [text]
    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        # find the latest separator within [chunk_size-overlap, chunk_size]
        cut_window = remaining[: chunk_size]
        cut_at = -1
        for sep in separators:
            idx = cut_window.rfind(sep)
            if idx >= chunk_size - overlap - len(sep):
                cut_at = idx + len(sep)
                break
        if cut_at <= 0:
            cut_at = chunk_size
        chunks.append(remaining[:cut_at].strip())
        # step forward by chunk_size - overlap
        step = max(1, cut_at - overlap)
        remaining = remaining[step:]
    if remaining.strip():
        chunks.append(remaining.strip())
    # dedupe consecutive duplicates
    out = []
    for c in chunks:
        if not out or out[-1] != c:
            out.append(c)
    return [c for c in out if len(c.strip()) >= 20]

class RecursiveChunkIndex:
    """Per-restaurant in-memory FAISS-like index over recursively-chunked reviews."""

    def __init__(self):
        self._cache: dict = {}  # restaurant -> (chunks, embeddings)

    def _build(self, restaurant: str, source_chunks: list[str]):
        new_chunks: list[str] = []
        for ch in source_chunks:
            new_chunks.extend(recursive_char_split(ch, 300, 60))
        if not new_chunks:
            self._cache[restaurant] = ([], None)
            return
        sbert = get_sbert()
        emb = sbert.encode(new_chunks, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        self._cache[restaurant] = (new_chunks, emb)

    def search(self, restaurant: str, source_chunks: list[str], question: str,
               top_k: int) -> list[dict]:
        if restaurant not in self._cache:
            self._build(restaurant, source_chunks)
        chunks, emb = self._cache[restaurant]
        if not chunks:
            return []
        sbert = get_sbert()
        q_emb = sbert.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = emb @ q_emb
        order = np.argsort(-sims)[:top_k]
        return [{"text": chunks[i], "metadata": {}, "score": float(sims[i])} for i in order]


# ---------------------------------------------------------------------------
# RAGAS proxy scorer
# ---------------------------------------------------------------------------
import re as _re

def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 8]

def score_faithfulness(answer: str, retrieved_chunks: list[str], max_sents: int = 4) -> float:
    """Avg NLI entailment of each (truncated) answer sentence by retrieved evidence."""
    sents = _split_sentences(answer)[:max_sents]
    if not sents:
        return 0.0
    if not retrieved_chunks:
        return 0.0
    nli = get_nli()
    evidence = " ".join(c[:300] for c in retrieved_chunks[:5])[:1800]
    if not evidence.strip():
        return 0.0
    scores = []
    for s in sents:
        try:
            r = nli(evidence, candidate_labels=[s], hypothesis_template="This text means {}.")
            # zero-shot returns label scores summing to 1; for single label, score = entailment prob
            scores.append(float(r["scores"][0]))
        except Exception as e:
            log.error(f"NLI fail: {e}")
            scores.append(0.0)
    return float(np.mean(scores))

def score_relevancy(answer: str, question: str) -> float:
    if not answer:
        return 0.0
    sbert = get_sbert()
    emb = sbert.encode([question, answer], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.clip(emb[0] @ emb[1], 0.0, 1.0))

def score_ctx_precision(question: str, chunks: list[str], threshold: float = 0.30) -> float:
    if not chunks:
        return 0.0
    sbert = get_sbert()
    q = sbert.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    cs = sbert.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    sims = cs @ q
    return float((sims >= threshold).mean())

def score_extractiveness(answer: str, retrieved_chunks: list[str]) -> float:
    """Max longest-common-substring fraction of answer covered by any single chunk.

    1.0 means the answer is a verbatim copy of some chunk; lower values indicate
    real synthesis. Computed on lowercased, punctuation-free text.
    """
    if not answer or not retrieved_chunks:
        return 0.0
    import string
    table = str.maketrans("", "", string.punctuation)
    a = answer.lower().translate(table).split()
    if not a:
        return 0.0
    best = 0
    for c in retrieved_chunks:
        b = c.lower().translate(table).split()
        # find longest run of words from a that appear consecutively in b
        bset_starts = {w: [i for i, x in enumerate(b) if x == w] for w in set(a)}
        for i in range(len(a)):
            if a[i] not in bset_starts:
                continue
            for start in bset_starts[a[i]]:
                k = 0
                while (i + k < len(a) and start + k < len(b)
                       and a[i + k] == b[start + k]):
                    k += 1
                if k > best:
                    best = k
    return float(best / max(1, len(a)))

def score_ctx_recall(chunks: list[str], gold_facts: dict) -> float:
    """Fraction of gold-fact signals recoverable from the union of retrieved chunks."""
    if not chunks:
        return 0.0
    blob = " ".join(chunks).lower()
    top_cats = gold_facts.get("top_categories", []) or []
    sentiment_dir = gold_facts.get("sentiment_dir", "mixed")
    pos_words = {"good","great","love","best","amazing","delicious","tasty","awesome","perfect","excellent","nice"}
    neg_words = {"bad","poor","worst","horrible","terrible","awful","disappointing","waste","never","avoid","slow","dirty","rude"}
    signals = []
    # category presence (one per top cat)
    for c in top_cats[:3]:
        terms = CATEGORY_TERMS.get(c, set())
        signals.append(any(t in blob for t in terms))
    # sentiment direction recoverable
    if sentiment_dir == "positive":
        signals.append(sum(1 for w in pos_words if w in blob) > sum(1 for w in neg_words if w in blob))
    elif sentiment_dir == "negative":
        signals.append(sum(1 for w in neg_words if w in blob) > sum(1 for w in pos_words if w in blob))
    else:
        signals.append(0 < sum(1 for w in pos_words | neg_words if w in blob))
    return float(sum(signals) / max(1, len(signals)))

def composite(faithful, relevancy, ctx_p, ctx_r):
    eps = 1e-3
    arr = np.array([max(faithful, eps), max(relevancy, eps),
                    max(ctx_p, eps), max(ctx_r, eps)])
    return float(np.exp(np.log(arr).mean()))

# ---------------------------------------------------------------------------
# Per-question runners
# ---------------------------------------------------------------------------
def load_qa():
    with open(os.path.join(EVAL_DIR, "rag_qa_eval.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_baseline_answers():
    with open(os.path.join(RESULTS_DIR, "baseline_rag_answers.json"), "r", encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
CONFIGS = ["template_baseline", "llm_existing_chunks", "llm_recursive_chunks", "llm_rerank"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only N questions (0 = all)")
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip LLM configs (debugging); only re-score baseline")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset")
    args = ap.parse_args()

    selected = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    print(f"Running configs: {selected}")

    qa = load_qa()
    if args.limit:
        qa = qa[: args.limit]

    baseline_answers = load_baseline_answers()

    # Load RAGChat once (heavy)
    print("Initializing RAGChat (loads SBERT + FAISS) ...")
    from rag_chat import RAGChat
    rag = RAGChat()
    # Force-load the consolidated FAISS index so semantic_search can be called directly.
    if not rag._load_vector_db():
        log.error("Failed to load consolidated FAISS store — Day-1 baseline should have built it.")
        sys.exit(1)
    rag.loaded = True
    print(f"  FAISS loaded: {rag.faiss_index.ntotal} vectors, {len(set(m.get('restaurant') for m in rag.doc_metadata))} restaurants.")

    recursive_index = RecursiveChunkIndex()

    all_answers: dict[str, list] = {c: [] for c in selected}
    all_scores_rows: list[dict] = []

    t0 = time.time()
    for i, item in enumerate(qa, 1):
        qid = item["id"]
        q = item["question"]
        rest = item["restaurant"]
        intent = item["intent"]
        gold = item["gold_facts"]

        # ----- Config 1: template baseline -----
        if "template_baseline" in selected:
            base = baseline_answers.get(qid)
            if base:
                ans = base["answer"]
                chunks = base["sources"]
            else:
                ans, chunks = rag.answer_query(q, restaurant_name=rest, top_k=5)
            rec = {"id": qid, "config": "template_baseline", "question": q,
                   "restaurant": rest, "intent": intent, "answer": ans,
                   "retrieved": chunks[:5]}
            all_answers["template_baseline"].append(rec)

        # ----- Retrieval for all LLM configs (top_15 once, reused) -----
        retrieved_15 = None
        if any(c in selected for c in ("llm_existing_chunks", "llm_rerank", "llm_recursive_chunks")):
            retrieved_15 = retrieve_existing(rag, q, rest, top_k=15)
            existing_top5 = retrieved_15[:5]

        # ----- Config 2: LLM on existing top-5 chunks -----
        if "llm_existing_chunks" in selected:
            chunks_text = [d["text"] for d in existing_top5]
            ans = llm_synthesize(q, rest, chunks_text) if chunks_text else \
                  f"No retrieved reviews for {rest}."
            all_answers["llm_existing_chunks"].append({
                "id": qid, "config": "llm_existing_chunks", "question": q,
                "restaurant": rest, "intent": intent, "answer": ans,
                "retrieved": chunks_text,
            })

        # ----- Config 3: LLM + recursive char chunking -----
        if "llm_recursive_chunks" in selected:
            # use top-15 existing chunks as source pool to re-split — keeps experiment
            # comparable while showing the effect of chunk size + overlap on retrieval+gen
            source_pool = [d["text"] for d in (retrieved_15 or [])]
            re_retrieved = recursive_index.search(rest, source_pool, q, top_k=5)
            chunks_text = [d["text"] for d in re_retrieved]
            ans = llm_synthesize(q, rest, chunks_text) if chunks_text else \
                  f"No retrieved chunks for {rest}."
            all_answers["llm_recursive_chunks"].append({
                "id": qid, "config": "llm_recursive_chunks", "question": q,
                "restaurant": rest, "intent": intent, "answer": ans,
                "retrieved": chunks_text,
            })

        # ----- Config 4: LLM + cross-encoder rerank -----
        if "llm_rerank" in selected:
            reranked = rerank_cross_encoder(q, retrieved_15 or [], top_k=5)
            chunks_text = [d["text"] for d in reranked]
            ans = llm_synthesize(q, rest, chunks_text) if chunks_text else \
                  f"No retrieved chunks for {rest}."
            all_answers["llm_rerank"].append({
                "id": qid, "config": "llm_rerank", "question": q,
                "restaurant": rest, "intent": intent, "answer": ans,
                "retrieved": chunks_text,
            })

        if i % 5 == 0 or i == len(qa):
            print(f"  [{i}/{len(qa)}] elapsed {time.time()-t0:.0f}s", flush=True)

    # Persist raw answers
    raw_out = []
    for cfg in selected:
        for rec in all_answers[cfg]:
            raw_out.append({**rec, "retrieved": [c[:400] for c in rec["retrieved"]]})
    with open(os.path.join(RESULTS_DIR, "phase2b_answers.json"), "w", encoding="utf-8") as f:
        json.dump(raw_out, f, indent=2)
    print(f"Wrote {len(raw_out)} answers to results/phase2b_answers.json")

    # ----- Scoring -----
    print("\nScoring all configs (RAGAS proxy + structural)...")
    for cfg in selected:
        for j, rec in enumerate(all_answers[cfg], 1):
            ans = rec["answer"]
            chunks = rec["retrieved"]
            qf = score_faithfulness(ans, chunks)
            qr = score_relevancy(ans, rec["question"])
            cp = score_ctx_precision(rec["question"], chunks)
            cr = score_ctx_recall(chunks, qa_by_id[rec["id"]]["gold_facts"])
            extract = score_extractiveness(ans, chunks)
            comp = composite(qf, qr, cp, cr)
            struct = structural_score(ans, qa_by_id[rec["id"]]["gold_facts"], rec["intent"])
            all_scores_rows.append({
                "id": rec["id"], "config": cfg, "restaurant": rec["restaurant"],
                "intent": rec["intent"],
                "faithfulness": qf, "relevancy": qr,
                "ctx_precision": cp, "ctx_recall": cr,
                "ragas_composite": comp,
                "extractiveness": extract,
                "answer_chars": len(ans) if isinstance(ans, str) else 0,
                "answer_words": len(ans.split()) if isinstance(ans, str) else 0,
                "sentiment_dir_match": struct["sentiment_dir_match"],
                "top_category_hit":    struct["top_category_hit"],
                "rating_mention":      struct["rating_mention"],
                "intent_addressed":    struct["intent_addressed"],
                "specificity":         struct["specificity_score"],
                "answer_preview": ans[:180].replace("\n", " ") if isinstance(ans, str) else "",
            })
            if j % 10 == 0:
                print(f"    scored {cfg} {j}", flush=True)

    df = pd.DataFrame(all_scores_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "phase2b_results.csv"), index=False)
    print(f"Wrote results/phase2b_results.csv  ({len(df)} rows)")

    # Aggregate
    metrics = {}
    for cfg in selected:
        sub = df[df["config"] == cfg]
        m = {
            "n": int(len(sub)),
            "faithfulness":  float(sub["faithfulness"].mean()),
            "relevancy":     float(sub["relevancy"].mean()),
            "ctx_precision": float(sub["ctx_precision"].mean()),
            "ctx_recall":    float(sub["ctx_recall"].mean()),
            "ragas_composite": float(sub["ragas_composite"].mean()),
            "extractiveness_mean": float(sub["extractiveness"].mean()),
            "answer_words_mean": float(sub["answer_words"].mean()),
            "sentiment_dir_match": float(sub["sentiment_dir_match"].mean()),
            "top_category_hit":    float(sub["top_category_hit"].mean()),
            "rating_mention":      float(sub["rating_mention"].mean()),
            "intent_addressed":    float(sub["intent_addressed"].mean()),
            "specificity":         float(sub["specificity"].mean()),
            "per_intent": {
                it: {
                    "n": int((sub["intent"]==it).sum()),
                    "ragas_composite": float(sub[sub["intent"]==it]["ragas_composite"].mean()),
                    "faithfulness":    float(sub[sub["intent"]==it]["faithfulness"].mean()),
                    "ctx_recall":      float(sub[sub["intent"]==it]["ctx_recall"].mean()),
                } for it in sub["intent"].unique()
            },
        }
        metrics[cfg] = m

    summary = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_questions": int(df["id"].nunique()),
        "configs": metrics,
        "models": {
            "synth_llm": "google/flan-t5-base (250M, encoder-decoder, instruction-tuned)",
            "retrieval_embed": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "faithfulness_nli": "valhalla/distilbart-mnli-12-3",
        },
        "metric_definitions": {
            "faithfulness":  "avg NLI entailment of answer sentences by retrieved evidence (0..1)",
            "relevancy":     "SBERT cosine(answer, question), clamped to [0,1]",
            "ctx_precision": "fraction of retrieved chunks with SBERT-sim(question, chunk) >= 0.30",
            "ctx_recall":    "fraction of gold-fact signals (top_categories + sentiment_dir) recoverable from retrieved chunks",
            "ragas_composite": "geometric mean of the four RAGAS-proxy metrics",
        },
        "note": "RAGAS axes scored locally (no API key). LLM judge will replace these in Day-6 frontier rerun.",
    }
    with open(os.path.join(RESULTS_DIR, "phase2b_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote results/phase2b_metrics.json")

    # ----- Samples -----
    for cfg in selected:
        sub = df[df["config"] == cfg].copy()
        sub.sort_values("ragas_composite", ascending=False).head(5).to_csv(
            os.path.join(SAMPLES_DIR, f"phase2b_{cfg}_top.csv"), index=False)
        sub.sort_values("ragas_composite").head(5).to_csv(
            os.path.join(SAMPLES_DIR, f"phase2b_{cfg}_bottom.csv"), index=False)

    # ----- Leaderboard -----
    print("\n" + "=" * 110)
    print(f"{'config':24s} {'RAGAS':>7} {'faith':>7} {'rel':>7} {'ctx_p':>7} {'ctx_r':>7} "
          f"{'extr':>7} {'words':>7} {'spec':>7}")
    print("=" * 110)
    for cfg in selected:
        m = metrics[cfg]
        print(f"{cfg:24s} {m['ragas_composite']:7.3f} {m['faithfulness']:7.3f} "
              f"{m['relevancy']:7.3f} {m['ctx_precision']:7.3f} {m['ctx_recall']:7.3f} "
              f"{m['extractiveness_mean']:7.3f} {m['answer_words_mean']:7.0f} "
              f"{m['specificity']:7.3f}")
    print("=" * 110)
    champ = max(selected, key=lambda c: metrics[c]["ragas_composite"])
    print(f"\nCHAMPION (by RAGAS composite): {champ}  ({metrics[champ]['ragas_composite']:.3f})")
    if "template_baseline" in selected:
        print(f"  delta vs template_baseline: "
              f"{metrics[champ]['ragas_composite'] - metrics['template_baseline']['ragas_composite']:+.3f}")
    # Also surface LLM-only winner since template wins on length-biased lexical metrics
    llm_cfgs = [c for c in selected if c.startswith("llm_")]
    if llm_cfgs:
        llm_champ = max(llm_cfgs, key=lambda c: metrics[c]["ragas_composite"])
        print(f"LLM champion: {llm_champ}  ({metrics[llm_champ]['ragas_composite']:.3f}, "
              f"extr={metrics[llm_champ]['extractiveness_mean']:.2f}, "
              f"words={metrics[llm_champ]['answer_words_mean']:.0f})")


# qa_by_id constructed at import time to share with scoring loop without re-loading
qa_by_id = {item["id"]: item for item in load_qa()}

if __name__ == "__main__":
    main()
