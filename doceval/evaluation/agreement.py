"""
agreement.py — Statistics for Tier 4: does the judge deserve to be trusted?

Most projects that use an LLM judge simply assume it is right. These functions
exist to test that assumption, and they are pure/offline so they are unit-testable
and run in CI without API access.

Cohen's kappa rather than raw agreement, because raw agreement is inflated by the
base rate: a judge that always answers 4 will "agree" with a human most of the time
on a corpus where 4 is common, while carrying no information at all. Kappa corrects
for agreement expected by chance.

Quadratic weighting is the default for the 1-5 rubric because the scale is ordinal:
confusing 4 with 5 is a much smaller error than confusing 1 with 5, and unweighted
kappa treats those identically.
"""

from collections import Counter


def confusion(a: list[int], b: list[int], categories: list[int]) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(categories)}
    m = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b, strict=True):
        m[idx[x]][idx[y]] += 1
    return m


def cohens_kappa(a: list[int], b: list[int], weights: str = "quadratic",
                 categories: list[int] | None = None) -> float:
    """
    Weighted Cohen's kappa between two raters.

    weights: "quadratic" (ordinal, default), "linear", or "none" (nominal).
    Returns NaN when undefined (fewer than 2 items, or no expected disagreement).
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    cats = categories or sorted(set(a) | set(b))
    if len(cats) < 2:
        return float("nan")

    n = len(a)
    obs = confusion(a, b, cats)
    ca, cb = Counter(a), Counter(b)
    k = len(cats)

    def w(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        d = abs(cats[i] - cats[j])
        span = cats[-1] - cats[0]
        if span == 0:
            return 0.0
        return (d / span) ** 2 if weights == "quadratic" else d / span

    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ca[cats[i]] * cb[cats[j]] / n for i in range(k) for j in range(k))
    if den == 0:
        return float("nan")
    return 1.0 - num / den


def exact_agreement(a: list[int], b: list[int]) -> float:
    if not a:
        return float("nan")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def within_one(a: list[int], b: list[int]) -> float:
    """Fraction of pairs within one scale point -- the practical bar for a 1-5 rubric."""
    if not a:
        return float("nan")
    return sum(abs(x - y) <= 1 for x, y in zip(a, b, strict=True)) / len(a)


def mean_signed_error(judge: list[int], human: list[int]) -> float:
    """Positive means the judge is systematically more generous than the human."""
    if not judge:
        return float("nan")
    return sum(j - h for j, h in zip(judge, human, strict=True)) / len(judge)


def flip_rate(first: list[int], second: list[int]) -> float:
    """
    Fraction of paired judgments that changed when only presentation order changed.

    Used for the position-bias probe. Anything materially above zero means the
    judge's ranking depends on argument order rather than content.
    """
    if not first:
        return float("nan")
    return sum(x != y for x, y in zip(first, second, strict=True)) / len(first)
