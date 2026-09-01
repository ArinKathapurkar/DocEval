"""Tier 2 deterministic checks. These must run without any API access."""

import math

from doceval.evaluation.grounding import check, entities_in, numbers_in

CONTEXT = {
    "p::1": "We evaluate on SQuAD and report an F1 of 88.5 across three runs.",
    "p::2": "The BERT baseline reaches 79.2 F1 on the same split.",
}


def test_numbers_ignores_trivial_values():
    assert numbers_in("we ran 3 seeds and got 88.5") == {"88.5"}


def test_entities_catches_acronyms_and_camelcase():
    assert entities_in("SQuAD and BERT and RoBERTa") >= {"SQuAD", "BERT", "RoBERTa"}


def test_fully_grounded_answer_scores_one():
    r = check("The model reaches 88.5 F1 on SQuAD.", ["p::1"], True, CONTEXT)
    assert r.citation_validity == 1.0
    assert r.citation_grounding == 1.0
    assert r.unsupported_numbers == 0


def test_hallucinated_number_is_flagged():
    r = check("The model reaches 92.7 F1.", ["p::1"], True, CONTEXT)
    assert r.unsupported_numbers == 1
    assert r.citation_grounding < 1.0


def test_citation_to_nonexistent_chunk_is_invalid():
    r = check("Some claim about 88.5.", ["p::99"], True, CONTEXT)
    assert r.citation_validity == 0.0
    assert r.citation_grounding == 0.0


def test_claim_in_context_but_not_in_cited_chunk_is_ungrounded():
    """79.2 appears in p::2, but the answer cites only p::1."""
    r = check("The baseline gets 79.2 F1.", ["p::1"], True, CONTEXT)
    assert r.unsupported_numbers == 0      # it is somewhere in the context
    assert r.citation_grounding == 0.0     # but not in what was cited


def test_answer_without_specifics_is_nan_not_a_free_pass():
    r = check("The paper describes an approach.", ["p::1"], True, CONTEXT)
    assert math.isnan(r.citation_grounding)


def test_abstention_scored_against_gold():
    assert check("Not stated.", [], False, CONTEXT, unanswerable=True).abstention_correct == 1.0
    assert check("Not stated.", [], False, CONTEXT, unanswerable=False).abstention_correct == 0.0
