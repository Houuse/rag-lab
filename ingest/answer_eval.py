"""Score the answers, not just the retrieval.

    python answer_eval.py                    # all 150, about 3 hours on CPU
    python answer_eval.py --limit 20         # a sample
    python answer_eval.py --group direct

`eval.py` measures whether the right evidence was retrieved. This measures what
comes out of the whole pipeline — route, retrieve, generate, verify — against
FinanceBench's answers.

Every question lands in exactly one bucket:

| outcome | meaning |
|---|---|
| `correct` | a figure matching the reference answer, and nothing ungrounded |
| `wrong` | a figure, confidently given, that does not match — the failure that matters |
| `refused` | said INSUFFICIENT EVIDENCE |
| `ungrounded` | answered, but invented a citation or a number |
| `unscored` | narrative answer; correctness needs a judge, which does not exist yet |

`refused` is not the same as `wrong`, and collapsing them hides the only thing
this design buys. A refusal means retrieval failed and the system said so. A
wrong answer means it did not.

One structural result to expect rather than discover: the 56 computed questions
ask for ratios and growth rates whose values appear nowhere in any filing.
Answering them requires arithmetic, and the system forbids the model from
calculating (ADR 0001). Until an arithmetic step exists in code, the correct
behaviour on all 56 is to refuse, and a `correct` there would mean the model
broke the rule and got lucky. They are scored separately for that reason.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ask
from eval import EXPECTED_SPLIT, classify, matches, numbers, overlap

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE.parent / "financebench" / "data" / "financebench_open_source.jsonl"
RUNS = HERE / "answer-runs"


def outcome(row: dict, answer: str, problems: list[str]) -> tuple[str, str]:
    """Which bucket, and why. Returns (outcome, note)."""
    if answer.startswith(ask.REFUSAL):
        return "refused", ""
    if problems:
        return "ungrounded", problems[0]

    if row["group"] == "narrative":
        # No reliable automatic grade. Report the overlap with the reference so
        # the number is at least inspectable, but do not pretend it is a score.
        ov = overlap(row["answer"], answer)
        return "unscored", f"overlap {ov:.2f}"

    target = numbers(row["answer"])
    # Citations carry digits; they are ids, not claims about the world.
    prose = ask.CITE_RE.sub(" ", answer)
    found = numbers(prose)
    if not found:
        return "wrong", "answered with no figure"
    if any(matches(t, f) for t in target for f in found):
        return "correct", ""
    return "wrong", f"expected {sorted(target)[:2]}, said {sorted(found)[:3]}"


def run(row: dict, args) -> dict:
    question = row["question"]
    t0 = time.perf_counter()
    kind, company, year = ask.route(question)
    ctx = ask.retrieve(question, kind, company, year, args.facts, args.chunks)
    prompt = ask.render(question, ctx)
    try:
        answer = ask.generate(prompt, args.model, args.host, stream=False,
                          backend=args.backend)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        # Any failure here is one question's problem, not the run's. A read
        # timeout killed a 50-question batch at number 40 and took the eleven
        # after it down with it: urlopen raises TimeoutError, an OSError and
        # not a URLError, so neither handler caught it.
        return {"id": row["financebench_id"], "group": row["group"],
                "outcome": "error", "note": f"{type(e).__name__}: {str(e)[:100]}",
                "seconds": round(time.perf_counter() - t0, 1)}
    problems = ask.verify(answer, ctx)
    kind_of, note = outcome(row, answer, problems)
    return {
        "id": row["financebench_id"],
        "group": row["group"],
        "outcome": kind_of,
        "note": note,
        "company": row["company"],
        "routed_company": company,
        "routed_fy": year,
        "oracle_doc": row["oracle_doc"],
        "n_facts": len(ctx.facts),
        "n_chunks": len(ctx.chunks),
        "seconds": round(time.perf_counter() - t0, 1),
        "expected": row["answer"][:120],
        "answer": " ".join(answer.split())[:300],
        "problems": "; ".join(problems)[:200],
    }


ORDER = ["correct", "wrong", "ungrounded", "refused", "unscored", "error"]


def summarise(results: list[dict], args) -> str:
    out = [
        "# Answer evaluation",
        "",
        f"Run {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"model {args.model}, "
        + (f"{args.facts} facts and {args.chunks} passages" if args.facts
           else "route-dependent evidence budget (see ask.BUDGET)"),
        "",
        "End to end: route, retrieve, generate, verify — scored against",
        "FinanceBench's answers. `eval.py` measures retrieval; this measures what",
        "actually comes out.",
        "",
    ]
    for group in ("direct", "computed", "narrative"):
        rows = [r for r in results if r["group"] == group]
        if not rows:
            continue
        c = Counter(r["outcome"] for r in rows)
        out += [f"## {group} (n={len(rows)})", ""]
        out.append("| outcome | n | share |")
        out.append("|---|---|---|")
        for k in ORDER:
            if c[k]:
                out.append(f"| {k} | {c[k]} | {c[k] / len(rows):.0%} |")
        out.append("")
        if group == "computed":
            out += [
                "Computed answers are ratios and growth rates whose values appear in no",
                "filing. Answering them needs arithmetic, and the model is forbidden to",
                "calculate (ADR 0001). Refusal is therefore the *correct* behaviour here",
                "until an arithmetic step exists in code — and a `correct` in this group",
                "means the model broke the rule and happened to be right.",
                "",
            ]
        if group == "narrative":
            out += [
                "`unscored` is honest, not a gap in the harness: grading a prose answer",
                "needs a judge, and a judge needs validating before its scores mean",
                "anything. The trigram overlap in `note` is inspectable, not a grade.",
                "",
            ]

    scored = [r for r in results if r["group"] == "direct"]
    if scored:
        c = Counter(r["outcome"] for r in scored)
        wrong, refused = c["wrong"] + c["ungrounded"], c["refused"]
        out += [
            "## The number that matters",
            "",
            f"On the {len(scored)} direct questions: **{c['correct']} correct, {wrong} wrong,",
            f"{refused} refused**.",
            "",
            "Wrong and refused are not interchangeable. A refusal means retrieval failed",
            "and the system said so, which is the designed behaviour and costs a user",
            "nothing but time. A wrong figure with a citation is the failure this whole",
            "design exists to prevent, because someone acts on it.",
            "",
        ]
        secs = [r["seconds"] for r in results if r.get("seconds")]
        if secs:
            out.append(f"Median {statistics.median(secs):.0f}s per question, "
                       f"{sum(secs) / 3600:.1f} hours total.")
            out.append("")
    return "\n".join(out) + "\n"


FIELDS = ["id", "group", "outcome", "note", "company", "routed_company", "routed_fy",
          "oracle_doc", "n_facts", "n_chunks", "seconds", "expected", "answer", "problems"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N, so a long run can be done in chunks")
    ap.add_argument("--group", choices=("direct", "computed", "narrative"))
    ap.add_argument("--ids", help="comma-separated financebench ids")
    ap.add_argument("--facts", type=int, default=None)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--model", default=ask.DEFAULT_MODEL)
    ap.add_argument("--host", default=ask.DEFAULT_HOST)
    ap.add_argument("--backend", default=ask.DEFAULT_BACKEND)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [json.loads(x) for x in QUESTIONS.read_text().splitlines() if x.strip()]
    for r in rows:
        r["group"] = classify(r)
        r["oracle_doc"] = r["evidence"][0]["doc_name"] if r["evidence"] else None
    if dict(Counter(r["group"] for r in rows)) != EXPECTED_SPLIT:
        sys.exit("group split changed — fix eval.py before trusting this")

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        rows = [r for r in rows if r["financebench_id"] in wanted]
    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]

    budget = f"{args.facts} facts" if args.facts else "route-dependent budget"
    print(f"{len(rows)} questions, {args.model}, {budget}")
    print(f"roughly {len(rows) * 80 / 3600:.1f} hours at 80s per question\n", flush=True)
    ask.warm_up()

    out = args.out or RUNS / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for i, row in enumerate(rows, 1):
            r = run(row, args)
            results.append(r)
            print(f"  [{i}/{len(rows)}] {r['outcome']:10} {r['group']:9} "
                  f"{r['seconds'] if 'seconds' in r else '?'}s  {r['id']}"
                  + (f"  {r['note'][:60]}" if r.get("note") else ""), flush=True)
            # Write as we go: three hours of results must not depend on the run
            # surviving to the end.
            with (out / "rows.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(results)
    except KeyboardInterrupt:
        print("\ninterrupted — partial results kept", flush=True)

    (out / "summary.md").write_text(summarise(results, args))
    print(f"\n{len(results)} rows -> {out}/rows.csv")
    print(summarise(results, args))


if __name__ == "__main__":
    main()
