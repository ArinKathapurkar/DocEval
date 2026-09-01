"""
run_pipeline.py — Retrieve, generate, and score with the free deterministic checks.

    python -m doceval.evaluation.run_pipeline --limit 150 --top-k 5

Produces (in artifacts/results/):
    answers_k<k>.jsonl    one generated answer + its context, per question
    tier2_k<k>.json       aggregated deterministic grounding scores

The --top-k sweep is one of the ablation rows: more context should raise
completeness while lowering faithfulness and raising cost, and this is where
that trade-off gets measured rather than asserted.

Cost is reported at the end of every run. Generation is the only paid step here;
Tier 2 scoring is free.
"""

import argparse
import concurrent.futures as cf
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from doceval import paths
from doceval.evaluation.grounding import check
from doceval.generate.answer import Generator
from doceval.index.embed import DEFAULT_MODEL, encode_queries, load_model
from doceval.index.retrievers import build_indexes

# claude-haiku-4-5 list price, USD per million tokens.
PRICE_IN, PRICE_OUT = 1.00, 5.00


def aggregate(reports: list[dict]) -> dict:
    """Mean each field, ignoring NaN, and keep the n that contributed."""
    out = {}
    for key in reports[0]:
        vals = [r[key] for r in reports if r[key] == r[key]]  # drop NaN
        out[key] = sum(vals) / len(vals) if vals else float("nan")
        out[f"{key}_n"] = len(vals)
    out["n_answers"] = len(reports)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_dotenv()
    paths.ensure_dirs()

    chunks = pd.read_parquet(paths.CHUNKS).reset_index(drop=True)
    questions = pd.read_parquet(paths.QUESTIONS)
    emb = np.load(paths.EMBEDDINGS)
    text_of = dict(zip(chunks.chunk_id, chunks.text, strict=True))

    # Include unanswerable questions: they are the abstention test.
    pool = questions.sample(min(args.limit, len(questions)), random_state=args.seed)
    print(f"{len(pool)} questions | retriever={args.retriever} | top_k={args.top_k}")
    print(f"  unanswerable in sample: {pool.unanswerable.sum()}")

    indexes = build_indexes(chunks, emb)
    qvecs = encode_queries(load_model(DEFAULT_MODEL), pool.question.tolist())

    def retrieve(row, qvec) -> list[str]:
        idx = indexes[row.paper_id]
        if args.retriever == "dense":
            return idx.dense(qvec, args.top_k)
        if args.retriever == "bm25":
            return idx.lexical(row.question, args.top_k)
        return idx.hybrid(row.question, qvec, args.top_k)

    gen = Generator()

    def work(i_row):
        i, row = i_row
        cids = retrieve(row, qvecs[i])
        contexts = [(c, text_of[c]) for c in cids]
        ans = gen.answer(row.question_id, row.question, contexts)
        rep = check(
            ans.answer, ans.citations, ans.sufficient,
            dict(contexts), unanswerable=bool(row.unanswerable),
        )
        return ans, rep, set(cids)

    rows = list(enumerate(pool.itertuples()))
    answers, reports = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        for ans, rep, _ in tqdm(
            pool_exec.map(work, rows), total=len(rows), unit="q"
        ):
            answers.append(ans)
            reports.append(rep.to_dict())

    out_a = paths.RESULTS / f"answers_k{args.top_k}_{args.retriever}.jsonl"
    out_a.write_text("\n".join(json.dumps(a.to_dict()) for a in answers))

    agg = aggregate(reports)
    tin = sum(a.usage.get("input_tokens", 0) for a in answers)
    tout = sum(a.usage.get("output_tokens", 0) for a in answers)
    agg["cost_usd"] = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    agg["top_k"], agg["retriever"] = args.top_k, args.retriever

    out_t = paths.RESULTS / f"tier2_k{args.top_k}_{args.retriever}.json"
    out_t.write_text(json.dumps(agg, indent=2))

    print(f"\nTier 2 (deterministic, free) over {agg['n_answers']} answers")
    for k in ("citation_validity", "has_citations", "citation_grounding",
              "abstention_correct", "unsupported_numbers", "unsupported_entities"):
        print(f"  {k:<24} {agg[k]:.4f}   (n={agg[f'{k}_n']})")
    print(f"\nGeneration cost: ${agg['cost_usd']:.4f}  ({tin:,} in / {tout:,} out tokens)")
    print(f"Wrote {out_a}\n      {out_t}")
