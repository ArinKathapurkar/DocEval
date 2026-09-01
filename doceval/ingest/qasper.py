"""
qasper.py — Turn the raw QASPER release into chunks and questions.

Run first:
    python -m doceval.ingest.qasper

Produces (in artifacts/):
    chunks.parquet     chunk_id, paper_id, kind, section, text
    questions.parquet  question_id, paper_id, question, evidence_chunk_ids,
                       answer_text, unanswerable

Why QASPER: every answer ships with annotated EVIDENCE PARAGRAPHS. That gives
exact ground truth for retrieval and for citation grounding, so a large share of
the evaluation is deterministic and free rather than dependent on an LLM judge.

Chunking decision, measured rather than assumed
-----------------------------------------------
Evidence strings are matched back to chunks by exact text. Indexing only body
paragraphs matches just 83.7% of evidence spans, which would silently cap
retrieval recall at 0.837 no matter how good the retriever got. The misses are
almost entirely table/figure captions (stored separately in figures_and_tables
and referenced as "FLOAT SELECTED: <caption>") and section headers.

Indexing captions and section names alongside paragraphs lifts the match rate to
97.3%. Anything still unmatched is dropped from the qrels rather than counted as
an unreachable target.
"""

import json

import pandas as pd

from doceval import paths

FLOAT_PREFIX = "FLOAT SELECTED: "


def load_papers(split: str = "dev") -> dict:
    path = paths.QASPER[split]
    if not path.exists():
        raise SystemExit(
            f"{path} missing. Download and extract:\n"
            "  curl -sL -o data/qasper.tgz "
            "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz\n"
            "  tar -xzf data/qasper.tgz -C data/"
        )
    return json.loads(path.read_text())


def paper_chunks(paper_id: str, paper: dict) -> list[dict]:
    """Every retrievable unit of one paper, in document order."""
    rows: list[dict] = []

    def add(kind: str, section: str, text: str) -> None:
        if text and text.strip():
            rows.append(
                {
                    "chunk_id": f"{paper_id}::{len(rows)}",
                    "paper_id": paper_id,
                    "kind": kind,
                    "section": section,
                    "text": text,
                }
            )

    add("abstract", "Abstract", paper.get("abstract", ""))
    for sec in paper.get("full_text", []):
        name = sec.get("section_name") or ""
        if name:
            add("section_header", name, name)
        for para in sec.get("paragraphs", []):
            add("paragraph", name, para)
    for ft in paper.get("figures_and_tables", []):
        cap = ft.get("caption", "")
        if cap:
            add("caption", "Figures and Tables", cap)
    return rows


def paper_questions(paper_id: str, paper: dict, text_to_chunk: dict[str, str]) -> list[dict]:
    """
    One row per question, merging its (possibly several) annotator answers.

    Evidence from every annotator is unioned: an evidence paragraph any annotator
    marked is a legitimate retrieval target.
    """
    rows = []
    for qa in paper.get("qas", []):
        evidence: set[str] = set()
        answers: list[str] = []
        unanswerable_votes = 0

        for ann in qa.get("answers", []):
            a = ann["answer"]
            if a.get("unanswerable"):
                unanswerable_votes += 1
            for ev in a.get("evidence", []):
                key = ev[len(FLOAT_PREFIX):] if ev.startswith(FLOAT_PREFIX) else ev
                if (cid := text_to_chunk.get(key)) is not None:
                    evidence.add(cid)
            if a.get("free_form_answer"):
                answers.append(a["free_form_answer"])
            elif a.get("extractive_spans"):
                answers.append("; ".join(a["extractive_spans"]))
            elif a.get("yes_no") is not None:
                answers.append("Yes" if a["yes_no"] else "No")

        n_ann = max(len(qa.get("answers", [])), 1)
        rows.append(
            {
                "question_id": qa["question_id"],
                "paper_id": paper_id,
                "question": qa["question"],
                "evidence_chunk_ids": sorted(evidence),
                "answer_text": answers[0] if answers else "",
                "all_answers": answers,
                # Majority of annotators said the paper does not answer it.
                "unanswerable": unanswerable_votes > n_ann / 2,
            }
        )
    return rows


def build(split: str = "dev") -> tuple[pd.DataFrame, pd.DataFrame]:
    papers = load_papers(split)
    all_chunks, all_questions = [], []

    for pid, paper in papers.items():
        chunks = paper_chunks(pid, paper)
        # Exact-text lookup, scoped to this paper: QASPER is single-document QA.
        lookup = {c["text"]: c["chunk_id"] for c in chunks}
        all_chunks.extend(chunks)
        all_questions.extend(paper_questions(pid, paper, lookup))

    return pd.DataFrame(all_chunks), pd.DataFrame(all_questions)


if __name__ == "__main__":
    paths.ensure_dirs()
    chunks, questions = build("dev")

    chunks.to_parquet(paths.CHUNKS, index=False)
    questions.to_parquet(paths.QUESTIONS, index=False)

    grounded = questions[questions.evidence_chunk_ids.apply(len) > 0]
    print(f"papers:    {chunks.paper_id.nunique():,}")
    print(f"chunks:    {len(chunks):,}")
    print(chunks.kind.value_counts().to_string())
    print(f"\nquestions: {len(questions):,}")
    print(f"  with resolvable evidence: {len(grounded):,} "
          f"({100 * len(grounded) / len(questions):.1f}%)")
    print(f"  unanswerable (majority):  {questions.unanswerable.sum():,}")
    print(f"  median chunks per paper:  {chunks.groupby('paper_id').size().median():.0f}")
    print(f"\nWrote {paths.CHUNKS}\n      {paths.QUESTIONS}")
