"""
retrieval_eval.py — Tier 1: retrieval quality. Free, deterministic, no API calls.

Run after ingest and embed:
    python -m doceval.evaluation.retrieval_eval [--top-k 10]

Produces:
    artifacts/results/retrieval.json
    artifacts/results/retrieval.md

Tier 1 of four. The tiers are separated by cost on purpose: this one and the
deterministic generation checks run on every commit in CI, while the LLM-judge
tiers run on demand. That split is how production eval pipelines are actually
structured, and it means the cheap signal is always available.
"""

import argparse
import json

import numpy as np
import pandas as pd

from doceval import paths
from doceval.evaluation.metrics import evaluate_run
from doceval.index.embed import DEFAULT_MODEL, encode_queries, load_model
from doceval.index.retrievers import build_indexes


def table(rows: dict[str, dict], metrics=("recall@1", "recall@3", "recall@5",
                                          "recall@10", "ndcg@10", "mrr")) -> str:
    lines = ["| System | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{m[k]:.4f}" if k in m else "-" for k in metrics) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    paths.ensure_dirs()
    chunks = pd.read_parquet(paths.CHUNKS).reset_index(drop=True)
    questions = pd.read_parquet(paths.QUESTIONS)
    emb = np.load(paths.EMBEDDINGS)

    # Only questions with resolvable evidence can be scored for retrieval.
    grounded = questions[questions.evidence_chunk_ids.apply(len) > 0].reset_index(drop=True)
    qrels = {r.question_id: set(r.evidence_chunk_ids) for r in grounded.itertuples()}
    print(f"{len(grounded):,} grounded questions over {len(chunks):,} chunks "
          f"in {chunks.paper_id.nunique():,} papers")

    indexes = build_indexes(chunks, emb)
    model = load_model(args.model)
    qvecs = encode_queries(model, grounded.question.tolist())

    runs: dict[str, dict[str, list[str]]] = {"Dense (bge-small)": {}, "BM25": {}, "Hybrid RRF": {}}
    for i, row in enumerate(grounded.itertuples()):
        idx = indexes[row.paper_id]
        qid, qvec, q = row.question_id, qvecs[i], row.question
        runs["Dense (bge-small)"][qid] = idx.dense(qvec, args.top_k)
        runs["BM25"][qid] = idx.lexical(q, args.top_k)
        runs["Hybrid RRF"][qid] = idx.hybrid(q, qvec, args.top_k)

    results = {name: evaluate_run(run, qrels) for name, run in runs.items()}

    (paths.RESULTS / "retrieval.json").write_text(json.dumps(results, indent=2))
    (paths.RESULTS / "retrieval.md").write_text(
        f"# Tier 1 — Retrieval ({len(grounded):,} grounded questions)\n\n" + table(results) + "\n"
    )
    print("\n" + table(results))
