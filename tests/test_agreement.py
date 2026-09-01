"""Tier 4 statistics. Offline and deterministic."""

import math

import pytest

from doceval.evaluation.agreement import (
    cohens_kappa,
    exact_agreement,
    flip_rate,
    mean_signed_error,
    within_one,
)


def test_perfect_agreement_is_one():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert cohens_kappa(a, a) == pytest.approx(1.0)


def test_constant_judge_earns_no_credit():
    """A judge that always says 4 agrees often but knows nothing; kappa must not
    reward that."""
    human = [4, 4, 4, 5, 3, 4, 2]
    judge = [4] * len(human)
    assert exact_agreement(judge, human) > 0.5      # raw agreement looks fine
    k = cohens_kappa(judge, human)
    assert math.isnan(k) or k <= 0.0                # kappa is not fooled


def test_quadratic_weighting_punishes_distant_errors_more():
    human = [1, 1, 1, 1]
    near = [2, 2, 2, 2]
    far = [5, 5, 5, 5]
    cats = [1, 2, 5]
    assert cohens_kappa(near, human, categories=cats) >= cohens_kappa(far, human, categories=cats)


def test_mean_signed_error_detects_generosity():
    assert mean_signed_error([5, 5, 4], [4, 4, 3]) == pytest.approx(1.0)
    assert mean_signed_error([3, 3], [4, 4]) == pytest.approx(-1.0)


def test_within_one_is_looser_than_exact():
    judge, human = [4, 3, 5], [5, 3, 4]
    assert within_one(judge, human) == 1.0
    assert exact_agreement(judge, human) == pytest.approx(1 / 3)


def test_flip_rate_measures_order_sensitivity():
    assert flip_rate([1, 1, 0, 1], [1, 0, 0, 1]) == 0.25
    assert flip_rate([1, 1], [1, 1]) == 0.0


def test_degenerate_inputs_are_nan_not_crashes():
    assert math.isnan(cohens_kappa([], []))
    assert math.isnan(cohens_kappa([3], [3]))
