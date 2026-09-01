"""
judge_validation.py — Tier 4: is the judge itself trustworthy?

    python -m doceval.evaluation.judge_validation --limit 60

Produces:
    artifacts/results/judge_validation.json

Runs three probes that need no human labels, plus a kappa-vs-human comparison
when hand labels are available.

  1. Position bias. Judge the same answer twice with the context passages in
     reversed order. A content-driven judge is invariant to that; a
     position-sensitive one is not.

  2. Verbosity bias. Re-judge the answer with irrelevant-but-true padding
     appended. The padding adds no information and contradicts nothing, so a
     well-behaved judge should not move. LLM judges commonly reward length.

  3. Self-consistency. Judge the same item repeatedly at temperature 1.0 and
     report the spread. A judge whose own variance approaches the effect you are
     trying to measure cannot resolve that effect.

Human agreement (Cohen's kappa) requires labels this tool cannot invent. Use
make_label_set.py to export items, label them by hand, and the kappa section
fills in automatically.
"""

import argparse
import json
import statistics
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from doceval import paths
from doceval.evaluation.agreement import (
    cohens_kappa,
    exact_agreement,
    flip_rate,
    mean_signed_error,
    within_one,
)
from doceval.evaluation.judge import AXES, Judge

# True of essentially any NLP paper, and irrelevant to any specific question.
PADDING = (
    " It is worth noting that the authors describe their methodology in the paper, "
    "that the work is situated within the natural language processing literature, "
    "and that the experiments were carried out by the researchers involved."
)


def load_answers(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(args) -> None:
    load_dotenv()
    paths.ensure_dirs()

    src = paths.RESULTS / args.answers
    if not src.exists():
        raise SystemExit(f"{src} missing. Run run_pipeline.py first.")

    answers = load_answers(src)[: args.limit]
    chunks = pd.read_parquet(paths.CHUNKS)
    questions = pd.read_parquet(paths.QUESTIONS).set_index("question_id")
    text_of = dict(zip(chunks.chunk_id, chunks.text, strict=True))

    judge = Judge(temperature=0.0)
    hot = Judge(temperature=1.0)
    out: dict = {"n_items": len(answers)}

    base, reversed_, padded = defaultdict(list), defaultdict(list), defaultdict(list)
    consistency = defaultdict(list)

    for a in tqdm(answers, unit="item"):
        qid = a["question_id"]
        ref = questions.loc[qid, "answer_text"] if qid in questions.index else ""
        ctx = [(c, text_of[c]) for c in a["context_ids"] if c in text_of]
        if not ctx:
            continue

        j1 = judge.judge(qid, a["question"], ctx, a["answer"], ref)
        j2 = judge.judge(qid, a["question"], list(reversed(ctx)), a["answer"], ref)
        j3 = judge.judge(qid, a["question"], ctx, a["answer"] + PADDING, ref)

        for axis in AXES:
            base[axis].append(getattr(j1, axis))
            reversed_[axis].append(getattr(j2, axis))
            padded[axis].append(getattr(j3, axis))

        if args.consistency_runs > 1:
            for _ in range(args.consistency_runs):
                consistency[qid].append(
                    hot.judge(qid, a["question"], ctx, a["answer"], ref).faithfulness
                )

    # ── probe 1: position bias ──────────────────────────────────────────────
    out["position_bias"] = {
        axis: {
            "flip_rate": flip_rate(base[axis], reversed_[axis]),
            "within_one": within_one(base[axis], reversed_[axis]),
            "kappa": cohens_kappa(base[axis], reversed_[axis]),
            "mean_shift": mean_signed_error(reversed_[axis], base[axis]),
        }
        for axis in AXES
    }

    # ── probe 2: verbosity bias ─────────────────────────────────────────────
    out["verbosity_bias"] = {
        axis: {
            "mean_shift_when_padded": mean_signed_error(padded[axis], base[axis]),
            "flip_rate": flip_rate(base[axis], padded[axis]),
            "exact_agreement": exact_agreement(base[axis], padded[axis]),
        }
        for axis in AXES
    }

    # ── probe 3: self-consistency ───────────────────────────────────────────
    if consistency:
        spreads = [max(v) - min(v) for v in consistency.values()]
        stdevs = [statistics.pstdev(v) for v in consistency.values() if len(v) > 1]
        out["self_consistency"] = {
            "runs_per_item": args.consistency_runs,
            "mean_range": sum(spreads) / len(spreads),
            "mean_stdev": sum(stdevs) / len(stdevs) if stdevs else float("nan"),
            "pct_items_unstable": sum(s > 0 for s in spreads) / len(spreads),
        }

    # ── human agreement, when labels exist ──────────────────────────────────
    labeled = paths.GOLDEN / "labeled.jsonl"
    if labeled.exists():
        human_rows = [json.loads(x) for x in labeled.read_text().splitlines() if x.strip()]
        by_qid = {r["question_id"]: r for r in human_rows}
        out["human_agreement"] = {}
        # Align the temperature-0 judge scores with the hand labels by question_id.
        qids = [a["question_id"] for a in answers]
        for axis in AXES:
            j, h = [], []
            for qid, score in zip(qids, base[axis], strict=False):
                if qid in by_qid and by_qid[qid].get(axis) is not None:
                    j.append(score)
                    h.append(int(by_qid[qid][axis]))
            if len(j) >= 2:
                out["human_agreement"][axis] = {
                    "n": len(j),
                    "kappa_quadratic": cohens_kappa(j, h),
                    "exact_agreement": exact_agreement(j, h),
                    "within_one": within_one(j, h),
                    "judge_minus_human": mean_signed_error(j, h),
                }
    else:
        out["human_agreement"] = (
            f"no labels at {labeled}; run make_label_set.py and label by hand"
        )

    dest = paths.RESULTS / "judge_validation.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + json.dumps(out, indent=2, default=str))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="answers_k5_hybrid.jsonl")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--consistency-runs", type=int, default=3)
    main(ap.parse_args())
