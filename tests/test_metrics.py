"""Unit tests for Tier 1 retrieval metrics."""

import math

import pytest

from doceval.evaluation.metrics import (
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["c1", "c2", "c3", "c4", "c5"]


def test_recall_respects_k():
    assert recall_at_k(RANKED, {"c1", "c5"}, 1) == 0.5
    assert recall_at_k(RANKED, {"c1", "c5"}, 5) == 1.0


def test_precision_counts_retrieved():
    assert precision_at_k(RANKED, {"c1"}, 5) == pytest.approx(0.2)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"c2"}) == 0.5
    assert reciprocal_rank(RANKED, {"nope"}) == 0.0


def test_ndcg_perfect_and_normalized():
    assert ndcg_at_k(RANKED, {"c1"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, {"c1", "c2"}, 10) == pytest.approx(1.0)


def test_ndcg_rank_sensitive():
    assert ndcg_at_k(RANKED, {"c1"}, 5) > ndcg_at_k(RANKED, {"c5"}, 5)


def test_empty_relevant_is_nan():
    assert math.isnan(recall_at_k(RANKED, set(), 5))


def test_evaluate_run_skips_ungrounded_questions():
    out = evaluate_run({"q1": RANKED, "q2": RANKED}, {"q1": {"c1"}}, ks=(1,))
    assert out["n_queries"] == 1
    assert out["recall@1"] == 1.0
