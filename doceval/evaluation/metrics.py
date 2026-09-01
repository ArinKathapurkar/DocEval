"""
metrics.py — Tier 1 retrieval metrics.

Evidence sets are small (usually 1-3 chunks) and exhaustively annotated, so unlike
a click-log benchmark these qrels are genuinely complete: an unretrieved evidence
chunk is a real miss, not an unlabeled maybe.
"""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not ranked[:k]:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(ranked[:k])


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if c in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else float("nan")


KS = (1, 3, 5, 10)


def evaluate_run(runs: dict[str, list[str]], qrels: dict[str, set[str]],
                 ks: tuple[int, ...] = KS) -> dict[str, float]:
    qids = [q for q in runs if qrels.get(q)]
    if not qids:
        return {}
    out: dict[str, float] = {"n_queries": len(qids)}
    for k in ks:
        out[f"recall@{k}"] = _mean(recall_at_k(runs[q], qrels[q], k) for q in qids)
        out[f"ndcg@{k}"] = _mean(ndcg_at_k(runs[q], qrels[q], k) for q in qids)
    out["precision@5"] = _mean(precision_at_k(runs[q], qrels[q], 5) for q in qids)
    out["mrr"] = _mean(reciprocal_rank(runs[q], qrels[q]) for q in qids)
    return out


def _mean(vals) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")
