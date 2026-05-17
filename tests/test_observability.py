"""RAGAS-proxy logging tests."""

from __future__ import annotations

import json
import os

from src.observability.ragas_log import (
    RAGASProxyLogger,
    score_request,
    _jaccard,
    _overlap_ratio,
    _tokens,
)


def test_tokens_excludes_stopwords():
    out = _tokens("The food was great and the staff was friendly")
    assert "the" not in out
    assert "and" not in out
    assert "food" in out
    assert "staff" in out


def test_jaccard_empty_returns_zero():
    assert _jaccard([], ["a"]) == 0.0
    assert _jaccard(["a"], []) == 0.0


def test_jaccard_full_overlap():
    assert _jaccard(["a", "b"], ["a", "b"]) == 1.0


def test_overlap_ratio_full():
    assert _overlap_ratio(["a", "b"], ["a", "b", "c"]) == 1.0


def test_overlap_ratio_partial():
    # one of two needles present
    assert _overlap_ratio(["a", "b"], ["a"]) == 0.5


def test_score_request_returns_all_five_scores():
    s = score_request(
        query="How is the food?",
        answer="The food was great and tasty.",
        retrieved_texts=["food was great", "tasty dishes"],
    )
    assert set(s.keys()) == {
        "faithfulness", "relevancy", "context_precision",
        "context_recall", "composite",
    }
    for v in s.values():
        assert 0.0 <= v <= 1.0


def test_score_request_handles_empty_inputs():
    s = score_request(query="", answer="", retrieved_texts=[])
    # composite is bounded below by 1e-3 floor; just verify shape
    assert "composite" in s
    assert 0.0 <= s["composite"] <= 1.0


def test_score_request_perfect_alignment():
    s = score_request(
        query="food great",
        answer="food great",
        retrieved_texts=["food great"],
    )
    assert s["composite"] > 0.5  # all four sub-scores are 1.0 -> composite 1.0


def test_logger_writes_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    lg = RAGASProxyLogger(str(p))
    lg.log(query="q", restaurant=None, answer="a b c",
           retrieved_texts=["a b", "c"], model="m", reranked=False,
           fallback_used=False, cache_hit=False, latency_ms=10.0)
    assert p.exists()
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model"] == "m"
    assert rec["latency_ms"] == 10.0
    assert "composite" in rec


def test_logger_appends_multiple_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    lg = RAGASProxyLogger(str(p))
    for i in range(3):
        lg.log(query=f"q{i}", restaurant=None, answer="a",
               retrieved_texts=["a"], model="m", reranked=False,
               fallback_used=False, cache_hit=False, latency_ms=1.0)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 3


def test_logger_creates_parent_dir(tmp_path):
    target = tmp_path / "logs" / "deeper" / "r.jsonl"
    lg = RAGASProxyLogger(str(target))
    lg.log(query="q", restaurant=None, answer="a",
           retrieved_texts=["a"], model="m", reranked=False,
           fallback_used=False, cache_hit=False, latency_ms=1.0)
    assert target.exists()
