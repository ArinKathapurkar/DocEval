"""
answer.py — Generate an answer with machine-checkable citations.

The output contract is what makes Tier 2 possible. The model must return

    {answer: str, citations: [chunk_id], sufficient: bool}

via FORCED tool use, not free text. Because citations come back as structured
chunk ids rather than prose, grounding can be verified deterministically and for
free -- no judge required to check whether a cited chunk exists or whether the
numbers in the answer appear in it.

`sufficient` lets the model abstain. QASPER marks 62 dev questions as
unanswerable from the paper, which gives a free, exactly-labeled abstention test.
"""

import json
import os
from dataclasses import dataclass, field

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

SYSTEM = """You answer questions about a research paper using ONLY the numbered \
context passages provided.

Rules:
- Every factual claim must come from the passages. Never use outside knowledge.
- Cite the chunk_id of every passage you used. Cite only passages you actually used.
- If the passages do not contain the answer, set sufficient=false and say so in the \
answer field. Do not guess.
- Keep the answer concise and specific."""

TOOL = {
    "name": "submit_answer",
    "description": "Submit the grounded answer and its supporting citations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The answer, grounded in the passages."},
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "chunk_id of every passage used.",
            },
            "sufficient": {
                "type": "boolean",
                "description": "False if the passages do not contain the answer.",
            },
        },
        "required": ["answer", "citations", "sufficient"],
    },
}


@dataclass
class Answer:
    """One generated answer plus the context it was given, for later scoring."""

    question_id: str
    question: str
    answer: str
    citations: list[str]
    sufficient: bool
    context_ids: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    """contexts: (chunk_id, text) in rank order."""
    blocks = "\n\n".join(f"[{cid}]\n{text}" for cid, text in contexts)
    return f"Context passages:\n\n{blocks}\n\nQuestion: {question}"


class Generator:
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

    def answer(self, question_id: str, question: str,
               contexts: list[tuple[str, str]]) -> Answer:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=[TOOL],
            # Forcing the tool guarantees a parseable object rather than prose.
            tool_choice={"type": "tool", "name": "submit_answer"},
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        )
        payload = _extract_tool_input(resp)
        return Answer(
            question_id=question_id,
            question=question,
            answer=payload.get("answer", ""),
            citations=list(payload.get("citations", [])),
            sufficient=bool(payload.get("sufficient", True)),
            context_ids=[cid for cid, _ in contexts],
            usage=_usage(resp),
        )


def _extract_tool_input(resp) -> dict:
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input if isinstance(block.input, dict) else json.loads(block.input)
    raise ValueError("Model returned no tool_use block despite forced tool choice")


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {"input_tokens": getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0)}
