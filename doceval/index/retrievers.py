"""
retrievers.py — Dense, lexical, and hybrid retrieval, scoped per paper.

QASPER is single-document QA: the question is about one specific paper, so
retrieval runs over that paper's chunks (median 64) rather than the whole corpus.
That keeps the task honest -- retrieving the right paragraph from 64 candidates is
the actual product problem, and inflating the candidate pool would only manufacture
impressive-looking numbers.
"""

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class PaperIndex:
    """Retrieval over the chunks of a single paper."""

    def __init__(self, chunk_ids: list[str], texts: list[str], emb: np.ndarray):
        self.chunk_ids = chunk_ids
        self.emb = emb                       # (n_chunks, D), L2-normalized
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def dense(self, qvec: np.ndarray, k: int) -> list[str]:
        scores = self.emb @ qvec
        return [self.chunk_ids[i] for i in np.argsort(-scores)[:k]]

    def lexical(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        return [self.chunk_ids[i] for i in np.argsort(-scores)[:k]]

    def hybrid(self, query: str, qvec: np.ndarray, k: int, k_const: int = 60) -> list[str]:
        return rrf([self.dense(qvec, k * 2), self.lexical(query, k * 2)], k_const, k)


def rrf(runs: list[list[str]], k_const: int = 60, top_k: int = 10) -> list[str]:
    """
    Reciprocal rank fusion.

    Fusing by rank rather than by score avoids having to calibrate BM25 scores
    against cosine similarities, which are on incompatible scales.
    """
    scores: dict[str, float] = defaultdict(float)
    for run in runs:
        for rank, cid in enumerate(run, start=1):
            scores[cid] += 1.0 / (k_const + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def build_indexes(chunks, emb: np.ndarray) -> dict[str, PaperIndex]:
    """One PaperIndex per paper, sharing rows of the global embedding matrix."""
    out = {}
    for pid, grp in chunks.groupby("paper_id", sort=False):
        rows = grp.index.to_numpy()
        out[pid] = PaperIndex(grp.chunk_id.tolist(), grp.text.tolist(), emb[rows])
    return out
