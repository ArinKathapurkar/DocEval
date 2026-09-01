# DocEval

Document-QA over research papers where **the evaluation harness is the product**, not
the chatbot.

Anyone can wire up a RAG pipeline in an afternoon. The hard part — and the part that
decides whether such a system can be shipped — is knowing whether its answers are
actually grounded, and knowing whether the thing measuring that is itself trustworthy.

---

## The four tiers

The tiers are separated **by cost**, deliberately. Tiers 1 and 2 are deterministic and
free, so they run in CI on every commit. Tiers 3 and 4 spend API tokens and run on
demand. That split is how production eval pipelines are actually structured, and it
means the cheap signal is always available rather than being something you run once
before a demo.

| Tier | What it measures | Cost | Runs |
|---|---|---|---|
| 1 — Retrieval | Recall@K, nDCG, MRR against annotated evidence | free | every commit |
| 2 — Grounding | citation validity, citation grounding, unsupported claims, abstention | free | every commit |
| 3 — Judge | faithfulness / relevance / completeness, 1–5 rubric | ~$ | on demand |
| 4 — Judge validation | **is the judge itself trustworthy?** | ~$ | on demand |

CI deliberately runs **without** `ANTHROPIC_API_KEY` present, so a test that silently
started calling the API would fail rather than quietly spend money.

## Why QASPER

[QASPER](https://allenai.org/data/qasper) — 281 dev papers, 1,005 questions. Every
answer ships with **annotated evidence paragraphs**, which gives exact ground truth for
retrieval *and* for citation grounding. That is what lets most of this evaluation be
deterministic instead of judge-dependent.

62 questions are labeled unanswerable from the paper, which is a free, exactly-labeled
abstention test — does the system correctly decline rather than confabulate?

### The chunking decision was measured, not assumed

Evidence spans are matched back to chunks by exact text. Indexing only body paragraphs
matches **83.7%** of evidence — which would have silently capped retrieval recall at
0.837 no matter how good the retriever became.

The misses turned out to be almost entirely table/figure captions (stored separately
under `figures_and_tables` and referenced as `FLOAT SELECTED: <caption>`) and section
headers. Indexing those alongside paragraphs lifts the match rate to **97.3%**.
Evidence still unmatched is dropped from the qrels rather than left as an unreachable
target that quietly penalizes every system equally.

Result: 19,271 chunks — 13,266 paragraphs, 3,795 section headers, 1,929 captions, 281
abstracts. 922 of 1,005 questions (91.7%) have resolvable evidence.

Retrieval is scoped **per paper** (median 64 chunks), because QASPER is single-document
QA. Widening the candidate pool would only manufacture better-looking numbers.

## Generation with machine-checkable citations

The model must return, via **forced tool use**:

```json
{"answer": "...", "citations": ["paper::12"], "sufficient": true}
```

Because citations come back as structured chunk ids rather than prose, grounding is
verifiable deterministically and for free. No judge is needed to check whether a cited
chunk exists, or whether the numbers in an answer appear in the passage it cites.

## Tier 2: what "grounded" means here

Tier 2 does not try to decide whether an answer is *good*. It decides whether it is
*grounded* — a narrower question, but one that can be answered exactly, and one that
catches a large share of hallucinations on its own.

Two choices worth calling out:

- **Grounding requires specifics to appear in a *cited* passage**, not merely somewhere
  in the context. An answer whose claim is true of the context but absent from what it
  actually cited is still a citation failure. A test pins that distinction.
- **Answers containing nothing checkable score NaN, not 1.0.** Otherwise vague answers
  farm a perfect grounding score by saying nothing falsifiable.

## Tier 4: auditing the judge

This is the part most LLM-judge setups skip entirely. Three probes need no human labels:

**Position bias** — judge the same answer twice with the context passages reversed. A
content-driven judge is invariant; a position-sensitive one is not.

**Verbosity bias** — re-judge with irrelevant-but-true padding appended. The padding
adds no information and contradicts nothing, so a well-behaved judge should not move.
LLM judges commonly reward length.

**Self-consistency** — judge repeatedly at temperature 1.0 and report the spread. A
judge whose own variance approaches the effect you are trying to measure cannot resolve
that effect.

And one that does need labels: **Cohen's kappa against hand labels**, quadratically
weighted. Kappa rather than raw agreement, because raw agreement is inflated by base
rate — a judge that always answers 4 will "agree" most of the time on a corpus where 4
is common while carrying zero information. There is a unit test pinning exactly that:
the constant judge scores >0.5 raw agreement and ≤0 kappa.

Labels cannot be invented by the tool. `make_label_set.py` exports items with blank
score fields; label them by hand against the same rubric the judge sees, and the kappa
section fills in automatically. Label **before** reading the judge's scores — seeing
them first anchors the labels and inflates the very agreement you are measuring.

## Results

*(Populated by the commands below; see `artifacts/results/`.)*

<!-- RESULTS -->

## Reproducing

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env       # add ANTHROPIC_API_KEY for tiers 3-4 only

curl -sL -o data/qasper.tgz \
  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar -xzf data/qasper.tgz -C data/

python -m doceval.ingest.qasper                      # free
python -m doceval.index.embed                        # free, local, MPS
python -m doceval.evaluation.retrieval_eval          # Tier 1, free
python -m doceval.evaluation.run_pipeline --limit 150 --top-k 5   # Tier 2 (+ generation)
python -m doceval.evaluation.make_label_set --n 50   # export for hand labeling
python -m doceval.evaluation.judge_validation        # Tiers 3-4
```

Note: `allenai/qasper` on HuggingFace is a script-based dataset and no longer loads
under `datasets` 3.x, so this reads the raw AllenAI release directly — which is more
reproducible anyway.

## Layout

```
doceval/
  ingest/     QASPER parsing, measured chunking
  index/      local embeddings, dense + BM25 + RRF, per-paper scoping
  generate/   forced-tool-use answers with structured citations
  evaluation/ metrics, grounding, judge, agreement statistics, runners
```
