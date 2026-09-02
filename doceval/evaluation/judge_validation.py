"""
judge_validation.py — Tier 4: is the judge itself trustworthy?

Two modes.

    # full audit: bias probes + agreement (many API calls)
    python -m doceval.evaluation.judge_validation --limit 60

    # after hand-labeling: agreement only, reusing cached judgments (cheap)
    python -m doceval.evaluation.judge_validation --reuse-judgments

Produces:
    artifacts/results/judge_validation.json
    artifacts/results/judgments.json      cached temperature-default judgments

Probes that need no human labels:

  1. Position bias. Judge the same answer twice with the context passages reversed.
     A content-driven judge is invariant to that; a position-sensitive one is not.

  2. Verbosity bias. Re-judge with irrelevant-but-true padding appended. The padding
     adds no information and contradicts nothing, so a well-behaved judge should not
     move. LLM judges commonly reward length.

  3. Self-consistency. Judge the same item repeatedly and report the spread. Run this
     FIRST when interpreting anything else: a probe effect smaller than the judge's
     own run-to-run noise is not a detected effect, it is an underpowered experiment.

     anthropic SDK 1.3.0 removed the `temperature` argument, so this measures the
     nondeterminism a deployment would actually experience rather than variance
     induced by a setting nobody ships with.

Human agreement (Cohen's kappa) needs labels this tool cannot invent. Use
label_cli.py to score them by hand, then rerun with --reuse-judgments.
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

JUDGMENTS = paths.RESULTS / "judgments.json"


# ── io helpers ──────────────────────────────────────────────────────────────


def load_answers(name: str) -> list[dict]:
    src = paths.RESULTS / name
    if not src.exists():
        raise SystemExit(f"{src} missing. Run run_pipeline.py first.")
    return [json.loads(x) for x in src.read_text().splitlines() if x.strip()]


def load_cache() -> dict[str, dict]:
    return json.loads(JUDGMENTS.read_text()) if JUDGMENTS.exists() else {}


def save_cache(cache: dict[str, dict]) -> None:
    JUDGMENTS.write_text(json.dumps(cache, indent=2))


def load_labels() -> dict[str, dict]:
    path = paths.GOLDEN / "labeled.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["question_id"]: r for r in rows}


def contexts_for(answer: dict, text_of: dict[str, str]) -> list[tuple[str, str]]:
    return [(c, text_of[c]) for c in answer["context_ids"] if c in text_of]


# ── agreement ───────────────────────────────────────────────────────────────


def agreement_report(judge_scores: dict[str, dict], labels: dict[str, dict]) -> dict:
    """Cohen's kappa and friends, over the items that have BOTH a judgment and a label."""
    out: dict = {}
    for axis in AXES:
        j, h = [], []
        for qid, label in labels.items():
            if qid not in judge_scores or label.get(axis) is None:
                continue
            j.append(int(judge_scores[qid][axis]))
            h.append(int(label[axis]))
        if len(j) < 2:
            out[axis] = f"only {len(j)} labeled item(s); need at least 2"
            continue
        out[axis] = {
            "n": len(j),
            "kappa_quadratic": cohens_kappa(j, h),
            "kappa_unweighted": cohens_kappa(j, h, weights="none"),
            "exact_agreement": exact_agreement(j, h),
            "within_one": within_one(j, h),
            "judge_minus_human": mean_signed_error(j, h),
        }
    return out


# ── main ────────────────────────────────────────────────────────────────────


def main(args) -> None:
    load_dotenv()
    paths.ensure_dirs()

    answers = load_answers(args.answers)
    chunks = pd.read_parquet(paths.CHUNKS)
    questions = pd.read_parquet(paths.QUESTIONS).set_index("question_id")
    text_of = dict(zip(chunks.chunk_id, chunks.text, strict=True))

    def reference(qid: str) -> str:
        return questions.loc[qid, "answer_text"] if qid in questions.index else ""

    judge = Judge()
    cache = load_cache()
    labels = load_labels()

    # ── agreement-only mode ─────────────────────────────────────────────────
    if args.reuse_judgments:
        if not labels:
            raise SystemExit(
                f"No labels at {paths.GOLDEN / 'labeled.jsonl'}.\n"
                "Run: python -m doceval.evaluation.label_cli"
            )
        by_qid = {a["question_id"]: a for a in answers}
        missing = [q for q in labels if q in by_qid and q not in cache]
        if missing:
            print(f"Judging {len(missing)} labeled item(s) not in cache "
                  f"({len(labels) - len(missing)} reused)")
            for qid in tqdm(missing, unit="item"):
                a = by_qid[qid]
                ctx = contexts_for(a, text_of)
                if not ctx:
                    continue
                j = judge.judge(qid, a["question"], ctx, a["answer"], reference(qid))
                cache[qid] = {ax: getattr(j, ax) for ax in AXES}
            save_cache(cache)
        else:
            print(f"All {len(labels)} labeled items already cached; no API calls needed.")

        report = agreement_report(cache, labels)
        dest = paths.RESULTS / "judge_validation.json"
        existing = json.loads(dest.read_text()) if dest.exists() else {}
        existing["human_agreement"] = report
        dest.write_text(json.dumps(existing, indent=2, default=str))
        print("\n" + json.dumps(report, indent=2, default=str))
        print(f"\nWrote {dest}")
        return

    # ── full audit ──────────────────────────────────────────────────────────
    answers = answers[: args.limit]
    out: dict = {"n_items": len(answers)}
    base, reversed_, padded = defaultdict(list), defaultdict(list), defaultdict(list)
    consistency = defaultdict(list)

    for a in tqdm(answers, unit="item"):
        qid = a["question_id"]
        ref = reference(qid)
        ctx = contexts_for(a, text_of)
        if not ctx:
            continue

        j1 = judge.judge(qid, a["question"], ctx, a["answer"], ref)
        j2 = judge.judge(qid, a["question"], list(reversed(ctx)), a["answer"], ref)
        j3 = judge.judge(qid, a["question"], ctx, a["answer"] + PADDING, ref)

        cache[qid] = {ax: getattr(j1, ax) for ax in AXES}
        for axis in AXES:
            base[axis].append(getattr(j1, axis))
            reversed_[axis].append(getattr(j2, axis))
            padded[axis].append(getattr(j3, axis))

        for _ in range(max(0, args.consistency_runs)):
            consistency[qid].append(
                judge.judge(qid, a["question"], ctx, a["answer"], ref).faithfulness
            )

    save_cache(cache)

    out["position_bias"] = {
        axis: {
            "flip_rate": flip_rate(base[axis], reversed_[axis]),
            "within_one": within_one(base[axis], reversed_[axis]),
            "kappa": cohens_kappa(base[axis], reversed_[axis]),
            "mean_shift": mean_signed_error(reversed_[axis], base[axis]),
        }
        for axis in AXES
    }
    out["verbosity_bias"] = {
        axis: {
            "mean_shift_when_padded": mean_signed_error(padded[axis], base[axis]),
            "flip_rate": flip_rate(base[axis], padded[axis]),
            "exact_agreement": exact_agreement(base[axis], padded[axis]),
        }
        for axis in AXES
    }
    if consistency:
        spreads = [max(v) - min(v) for v in consistency.values()]
        stdevs = [statistics.pstdev(v) for v in consistency.values() if len(v) > 1]
        out["self_consistency"] = {
            "runs_per_item": args.consistency_runs,
            "mean_range": sum(spreads) / len(spreads),
            "mean_stdev": sum(stdevs) / len(stdevs) if stdevs else float("nan"),
            "pct_items_unstable": sum(s > 0 for s in spreads) / len(spreads),
        }

    out["human_agreement"] = (
        agreement_report(cache, labels)
        if labels
        else f"no labels at {paths.GOLDEN / 'labeled.jsonl'}; run label_cli.py"
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
    ap.add_argument("--reuse-judgments", action="store_true",
                    help="agreement only: reuse cached judgments, judge only what is missing")
    main(ap.parse_args())
