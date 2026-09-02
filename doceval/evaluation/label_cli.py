"""
label_cli.py — Interactive labeling for the judge-validation golden set.

    python -m doceval.evaluation.label_cli            # label 25 (default)
    python -m doceval.evaluation.label_cli --n 50     # label all
    python -m doceval.evaluation.label_cli --review   # re-read what you already scored

Reads golden/to_label.jsonl, writes golden/labeled.jsonl after EVERY item, so
quitting mid-session loses nothing and restarting resumes exactly where you stopped.

Scoring one item means reading ~2,400 characters and making three calls, so budget
1-3 minutes each. Labeling 25 rather than 50 still yields a real human kappa with
wider confidence intervals, which is the honest trade and half the time.

Type three digits (`5 4 3` or `543`) to score all axes at once, or use the
single-key commands listed in the prompt.
"""

import argparse
import json
import shutil
import textwrap

from doceval import paths

AXES = ("faithfulness", "relevance", "completeness")

RUBRIC = {
    "faithfulness": [
        "5  every claim directly supported by the passages",
        "4  all claims supported; minor paraphrase drift",
        "3  mostly supported; one unsupported peripheral claim",
        "2  a central claim is unsupported by the passages",
        "1  substantially fabricated",
    ],
    "relevance": [
        "5  directly and fully answers the question",
        "4  answers it, with some unnecessary material",
        "3  partially answers it",
        "2  related but does not answer it",
        "1  off topic",
    ],
    "completeness": [
        "5  covers everything in the reference",
        "4  covers the main point, omits a detail",
        "3  covers roughly half",
        "2  covers a minor fragment",
        "1  misses the substance entirely",
    ],
}

GUIDANCE = """\
Judge only what is present. Do NOT reward length: a short correct answer must score
higher than a long one padded with irrelevant true statements.

If the system abstained ("the passages do not contain...") and the reference has a real
answer, that is usually faithfulness 5 (it claimed nothing unsupported) and completeness
1 (it conveyed nothing). That split is the point of scoring three axes separately.

If the reference answer is empty, press `n` for completeness rather than guessing."""


def width() -> int:
    return min(shutil.get_terminal_size((100, 24)).columns, 100)


def rule(char: str = "─") -> str:
    return char * width()


def wrap(text: str, indent: str = "") -> str:
    return "\n".join(
        textwrap.fill(line, width=width(), initial_indent=indent, subsequent_indent=indent)
        or indent
        for line in text.splitlines() or [""]
    )


def load_items() -> list[dict]:
    src = paths.GOLDEN / "to_label.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} missing. Run: python -m doceval.evaluation.make_label_set")
    return [json.loads(x) for x in src.read_text().splitlines() if x.strip()]


def load_done() -> dict[str, dict]:
    dst = paths.GOLDEN / "labeled.jsonl"
    if not dst.exists():
        return {}
    rows = [json.loads(x) for x in dst.read_text().splitlines() if x.strip()]
    return {r["question_id"]: r for r in rows}


def save(done: dict[str, dict]) -> None:
    dst = paths.GOLDEN / "labeled.jsonl"
    dst.write_text("\n".join(json.dumps(r) for r in done.values()) + "\n")


def show_item(item: dict, n: int, total: int) -> None:
    cited = set(item["system_citations"])
    print("\n" + rule("═"))
    print(f"  ITEM {n} of {total}")
    print(rule("═"))
    print("\nQUESTION")
    print(wrap(item["question"], "  "))

    print(f"\nCONTEXT  ({len(item['context'])} passages; ▶ = cited by the system)")
    for c in item["context"]:
        mark = "▶" if c["chunk_id"] in cited else " "
        print(f"\n {mark} [{c['chunk_id']}]")
        print(wrap(c["text"], "     "))

    print("\n" + rule())
    print("SYSTEM ANSWER")
    print(wrap(item["system_answer"], "  "))
    print(f"\n  cited: {', '.join(item['system_citations']) or '(nothing)'}")

    ref = item["reference_answer"].strip()
    print("\nREFERENCE ANSWER (human-written)")
    print(wrap(ref if ref else "(empty — press n for completeness)", "  "))
    print(rule())


def show_rubric() -> None:
    print()
    for axis in AXES:
        print(f"  {axis.upper()}")
        for line in RUBRIC[axis]:
            print(f"    {line}")
        print()
    print(wrap(GUIDANCE, "  "))
    print()


def parse_scores(raw: str) -> list[int | None] | None:
    """Accept '5 4 3', '543', or '5,4,3'. `n` marks an axis not applicable."""
    toks = raw.replace(",", " ").split()
    if len(toks) == 1 and len(toks[0]) == 3 and all(c in "12345n" for c in toks[0]):
        toks = list(toks[0])
    if len(toks) != 3:
        return None
    out: list[int | None] = []
    for t in toks:
        if t.lower() == "n":
            out.append(None)
        elif t in "12345":
            out.append(int(t))
        else:
            return None
    return out


def prompt_one(axis: str) -> int | None | str:
    while True:
        raw = input(f"    {axis:<13} 1-5 (n=n/a, ?=rubric, s=skip, q=quit): ").strip().lower()
        if raw in ("q", "s", "?"):
            return raw
        if raw == "n":
            return None
        if raw in ("1", "2", "3", "4", "5"):
            return int(raw)
        print("      enter 1-5, or n / ? / s / q")


def main(args) -> None:
    paths.ensure_dirs()
    items = load_items()
    done = load_done()

    todo = [i for i in items if i["question_id"] not in done]
    target = max(0, args.n - len(done))
    if args.review:
        todo, target = [], 0

    print(rule("═"))
    print("  DocEval — judge validation labeling")
    print(rule("═"))
    print(f"\n  already labeled : {len(done)}")
    print(f"  this session    : up to {target}")
    print(f"  saved to        : {paths.GOLDEN / 'labeled.jsonl'} (after every item)")
    print("\n  Enter three scores at once, e.g.  5 4 3   or   543")
    print("  Single keys:  ? rubric   s skip   b back   q quit (progress is kept)")
    show_rubric()

    if target == 0:
        print(f"  Nothing to do. {len(done)} items labeled.")
        print("  Next: python -m doceval.evaluation.judge_validation --reuse-judgments")
        return

    i = 0
    labeled_this_session = 0
    while i < len(todo) and labeled_this_session < target:
        item = todo[i]
        show_item(item, len(done) + 1, args.n)

        raw = input("\n  scores (f r c) > ").strip().lower()

        if raw == "q":
            break
        if raw == "?":
            show_rubric()
            continue
        if raw == "s":
            i += 1
            continue
        if raw == "b":
            if done:
                last = list(done)[-1]
                del done[last]
                save(done)
                labeled_this_session = max(0, labeled_this_session - 1)
                todo.insert(i, next(x for x in items if x["question_id"] == last))
                print("  removed the previous label; re-scoring it now")
            continue

        scores = parse_scores(raw)
        if scores is None:
            # Anything unparseable falls back to one axis at a time.
            scores, control = [], None
            for axis in AXES:
                v = prompt_one(axis)
                if isinstance(v, str):        # q / s / ?
                    control = v
                    break
                scores.append(v)
            if control == "q":
                break
            if control == "s":
                i += 1
                continue
            if control == "?":
                show_rubric()
                continue
            if len(scores) != 3:
                continue

        row = {"question_id": item["question_id"], "question": item["question"]}
        row.update(dict(zip(AXES, scores, strict=True)))
        done[item["question_id"]] = row
        save(done)

        shown = "  ".join(
            f"{a}={'n/a' if s is None else s}"
            for a, s in zip(AXES, scores, strict=True)
        )
        print(f"  ✓ saved  {shown}   ({len(done)} labeled)")
        i += 1
        labeled_this_session += 1

    print("\n" + rule("═"))
    print(f"  {len(done)} items labeled -> {paths.GOLDEN / 'labeled.jsonl'}")
    if len(done) >= 2:
        print("\n  Next (no new API calls):")
        print("    python -m doceval.evaluation.judge_validation --reuse-judgments")
    else:
        print("\n  Label at least 2 items before computing agreement.")
    print(rule("═"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="target number of labeled items")
    ap.add_argument("--review", action="store_true", help="show progress without labeling")
    main(ap.parse_args())
