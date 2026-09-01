"""
make_label_set.py — Export a sample for hand labeling.

    python -m doceval.evaluation.make_label_set --n 50

Writes golden/to_label.jsonl with the question, the retrieved context, the
system answer, the reference answer, and blank score fields.

Score each item 1-5 on faithfulness / relevance / completeness using the SAME
rubric the judge sees (doceval/evaluation/judge.py), save as golden/labeled.jsonl,
and judge_validation.py will report Cohen's kappa against your labels.

Label before looking at the judge's scores. Seeing them first anchors the labels
and inflates the agreement you are trying to measure.
"""

import argparse
import json
import random

import pandas as pd

from doceval import paths


def main(args) -> None:
    paths.ensure_dirs()
    src = paths.RESULTS / args.answers
    if not src.exists():
        raise SystemExit(f"{src} missing. Run run_pipeline.py first.")

    answers = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    chunks = pd.read_parquet(paths.CHUNKS)
    questions = pd.read_parquet(paths.QUESTIONS).set_index("question_id")
    text_of = dict(zip(chunks.chunk_id, chunks.text, strict=True))

    random.seed(args.seed)
    sample = random.sample(answers, min(args.n, len(answers)))

    out = paths.GOLDEN / "to_label.jsonl"
    with out.open("w") as fh:
        for a in sample:
            qid = a["question_id"]
            fh.write(json.dumps({
                "question_id": qid,
                "question": a["question"],
                "context": [
                    {"chunk_id": c, "text": text_of.get(c, "")} for c in a["context_ids"]
                ],
                "system_answer": a["answer"],
                "system_citations": a["citations"],
                "reference_answer": (
                    questions.loc[qid, "answer_text"] if qid in questions.index else ""
                ),
                # Fill these in by hand, 1-5.
                "faithfulness": None,
                "relevance": None,
                "completeness": None,
            }) + "\n")

    print(f"Wrote {len(sample)} items to {out}")
    print("Label by hand, save as golden/labeled.jsonl, then run judge_validation.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="answers_k5_hybrid.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())
