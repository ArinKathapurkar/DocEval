"""
judge.py — Tier 3: LLM-as-judge. Costs API calls, so it runs on demand, not in CI.

Scores an answer on three axes against the retrieved context and the gold answer:

    faithfulness  is every claim supported by the context?
    relevance     does it actually answer the question asked?
    completeness  does it cover what the gold answer covers?

Scores are 1-5 with explicit anchors in the prompt. Vague rubrics ("rate quality
0-10") produce judges that cluster everything at 7 and correlate with nothing;
anchoring each point to an observable condition is what makes the scores mean
something, and Tier 4 exists to check whether that worked.

Structured output comes back via forced tool use, so a malformed judgment is
impossible rather than merely unlikely.

Note on sampling: anthropic SDK 1.3.0 no longer exposes a `temperature` argument on
messages.create (output_config carries `effort` and `format` instead). The
self-consistency probe in Tier 4 therefore repeats identical calls under production
defaults and reports the variance actually observed, rather than variance induced by
a temperature setting a deployment would not use anyway.
"""

import json
import os
from dataclasses import dataclass

MODEL = "claude-haiku-4-5"

SYSTEM = """You are a strict evaluator of question-answering systems. You are given \
a question, the context passages the system was shown, the system's answer, and a \
reference answer written by a human expert.

Score on three axes, 1-5:

FAITHFULNESS - is every claim in the answer supported by the context passages?
  5: every claim directly supported by the passages
  4: all claims supported; minor paraphrase drift
  3: mostly supported; one unsupported peripheral claim
  2: a central claim is unsupported by the passages
  1: substantially fabricated

RELEVANCE - does the answer address the question actually asked?
  5: directly and fully answers the question
  4: answers it, with some unnecessary material
  3: partially answers it
  2: related but does not answer it
  1: off topic

COMPLETENESS - does it cover what the reference answer covers?
  5: covers everything in the reference
  4: covers the main point, omits a detail
  3: covers roughly half
  2: covers a minor fragment
  1: misses the substance entirely

Judge only what is present. Do not reward length. A short correct answer must \
score higher than a long one padded with irrelevant true statements."""

TOOL = {
    "name": "submit_judgment",
    "description": "Submit rubric scores with a brief justification.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
            "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning": {"type": "string", "description": "One or two sentences."},
        },
        "required": ["faithfulness", "relevance", "completeness", "reasoning"],
    },
}

AXES = ("faithfulness", "relevance", "completeness")


@dataclass
class Judgment:
    question_id: str
    faithfulness: int
    relevance: int
    completeness: int
    reasoning: str
    usage: dict

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_prompt(question: str, contexts: list[tuple[str, str]],
                 answer: str, reference: str) -> str:
    blocks = "\n\n".join(f"[{cid}]\n{text}" for cid, text in contexts)
    return (
        f"Question: {question}\n\n"
        f"Context passages:\n\n{blocks}\n\n"
        f"System answer: {answer}\n\n"
        f"Reference answer: {reference}"
    )


class Judge:
    """Wraps the Anthropic client. Inject `client` in tests to avoid API calls."""

    def __init__(self, client=None, model: str = MODEL):
        if client is None:
            import anthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise SystemExit("ANTHROPIC_API_KEY not set. See .env.example.")
            client = anthropic.Anthropic(api_key=key)
        self.client = client
        self.model = model

    def judge(self, question_id: str, question: str, contexts: list[tuple[str, str]],
              answer: str, reference: str) -> Judgment:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "submit_judgment"},
            messages=[{"role": "user",
                       "content": build_prompt(question, contexts, answer, reference)}],
        )
        p = _tool_input(resp)
        u = getattr(resp, "usage", None)
        return Judgment(
            question_id=question_id,
            faithfulness=int(p["faithfulness"]),
            relevance=int(p["relevance"]),
            completeness=int(p["completeness"]),
            reasoning=p.get("reasoning", ""),
            usage={"input_tokens": getattr(u, "input_tokens", 0),
                   "output_tokens": getattr(u, "output_tokens", 0)} if u else {},
        )


def _tool_input(resp) -> dict:
    for b in resp.content:
        if getattr(b, "type", None) == "tool_use":
            return b.input if isinstance(b.input, dict) else json.loads(b.input)
    raise ValueError("Judge returned no tool_use block despite forced tool choice")
