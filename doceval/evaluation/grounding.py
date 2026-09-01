"""
grounding.py — Tier 2: deterministic generation checks. Free, no API calls.

Every check here is a pure function over (answer, citations, context). No model is
involved, so the whole tier runs in CI on every commit at zero cost. That is the
point of the tier split: the cheap signal should always be available, and API
spend should be reserved for the judgments that genuinely require a model.

These checks do not attempt to decide whether an answer is *good*. They decide
whether it is *grounded* -- a narrower question, but one that can be answered
exactly, and one that catches a large share of hallucinations on its own.
"""

import re
from dataclasses import dataclass

# Numbers, including decimals, percentages, and negatives.
NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
# Candidate name-like tokens. A token counts as an entity if it carries two or
# more capitals, which catches acronyms (BERT, NLP), CamelCase (WordPiece), and
# the mixed forms common in ML papers (SQuAD, RoBERTa, GPT-3) that a pure
# acronym-or-CamelCase regex misses.
WORD = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*\b")
MIN_CAPS = 2

# Numeric tokens too generic to be evidence of grounding.
TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5", "10", "100"}


@dataclass
class GroundingReport:
    """Per-answer deterministic verdicts. All fields are 1.0/0.0 or NaN if N/A."""

    citation_validity: float
    has_citations: float
    citation_grounding: float
    unsupported_numbers: int
    unsupported_entities: int
    abstention_correct: float
    n_citations: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def numbers_in(text: str) -> set[str]:
    return {n for n in NUMBER.findall(text) if n.strip("%") not in TRIVIAL_NUMBERS}


def entities_in(text: str) -> set[str]:
    return {
        w for w in WORD.findall(text)
        if sum(c.isupper() for c in w) >= MIN_CAPS
    }


def check(
    answer: str,
    citations: list[str],
    sufficient: bool,
    context: dict[str, str],
    unanswerable: bool | None = None,
) -> GroundingReport:
    """
    Score one answer against the context it was shown.

    context: chunk_id -> text, exactly what the model was given.
    unanswerable: gold label, when known, for the abstention check.
    """
    valid = [c for c in citations if c in context]

    # Do the specifics in the answer actually appear in the cited passages?
    cited_text = _normalize(" ".join(context[c] for c in valid))
    all_text = _normalize(" ".join(context.values()))

    ans_numbers = numbers_in(answer)
    ans_entities = entities_in(answer)

    unsupported_nums = sum(1 for n in ans_numbers if n.lower() not in all_text)
    unsupported_ents = sum(1 for e in ans_entities if e.lower() not in all_text)

    # Grounding: every specific claim traceable to a CITED passage, not merely
    # to the context at large. Answers with nothing specific to check are NaN
    # rather than a free pass.
    specifics = ans_numbers | ans_entities
    if not specifics:
        grounding = float("nan")
    elif not valid:
        grounding = 0.0
    else:
        hit = sum(1 for s in specifics if s.lower() in cited_text)
        grounding = hit / len(specifics)

    abstention = float("nan")
    if unanswerable is not None:
        abstention = 1.0 if (not sufficient) == unanswerable else 0.0

    return GroundingReport(
        citation_validity=(len(valid) / len(citations)) if citations else float("nan"),
        has_citations=1.0 if citations else 0.0,
        citation_grounding=grounding,
        unsupported_numbers=unsupported_nums,
        unsupported_entities=unsupported_ents,
        abstention_correct=abstention,
        n_citations=len(citations),
    )
