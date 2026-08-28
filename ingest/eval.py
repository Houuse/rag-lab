"""Score retrieval against the 150 FinanceBench questions.

    python eval.py                         # both conditions, all 150
    python eval.py --limit 0               # classify only, run nothing
    python eval.py --group direct --limit 10
    python eval.py --condition oracle

Implements `docs/eval-plan.md`. Read that first — it decides the metrics and
why the three question groups are never blended into one number. This file is
the mechanism, not the argument.

Retrieval only. Answer correctness, groundedness and the LLM judge need the
generator and are out of scope.

Two conditions, always reported as a pair. **Oracle** restricts retrieval to
the filing FinanceBench names for the question, measuring ranking with entity
and period resolution removed. **Full corpus** restricts nothing, measuring
the system as a user meets it. The gap between them is the cost of resolving
which company and which year — the failure the manual probes point at, where
"capital expenditures 2016" returns 2018 values.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from search import resolve_scope, search, search_facts

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE.parent / "financebench" / "data" / "financebench_open_source.jsonl"
RUNS = HERE / "eval-runs"

KS = (1, 5, 10, 20)
TRIGRAM_THRESHOLD = 0.25

# The documented split. Asserted rather than trusted: the groups are derived
# from the data, so a change to number parsing could reshape them silently and
# every metric below would still produce confident numbers.
EXPECTED_SPLIT = {"direct": 70, "computed": 56, "narrative": 24}


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def numbers(text: str | None) -> set[float]:
    """Every number in a string, as unsigned magnitudes.

    Zero counts. Dropping it with a truthiness test moves
    financebench_id_01319, whose answer is exactly "0", out of the direct group
    and into narrative — which is the difference between reproducing the
    documented 70/56/24 split and quietly getting 69/56/25.
    """
    out: set[float] = set()
    for m in NUM_RE.findall(text or ""):
        s = m.strip("%").replace("$", "").replace(",", "").strip("()")
        try:
            out.add(round(abs(float(s)), 4))
        except ValueError:
            continue
    return out


def matches(target: float, found: float) -> bool:
    """Same value, tolerating the scale a filing reported it in.

    A filing states 1,577 in a table headed "(Millions)"; the answer says
    1577.00. Elsewhere the same figure appears as 1,577,000 thousands. Scale is
    stored per fact but is resolved for only about 60% of them, so comparing
    magnitudes across a factor of 1000 is the honest test.
    """
    if abs(target - found) < 0.01:
        return True
    return abs(target * 1000 - found) < 0.01 or abs(target / 1000 - found) < 1e-6


def any_match(targets: set[float], found: float) -> bool:
    return any(matches(t, found) for t in targets)


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------


def classify(row: dict) -> str:
    """direct | computed | narrative, derived from the data (eval-plan.md).

    Narrative means the answer is not numeric. Direct means the answer's value
    is printed in the evidence. Computed means it is not — a ratio or a growth
    rate, which appears nowhere in the filing, so the answer value cannot be
    the retrieval target and the *inputs* are scored instead.
    """
    answer = numbers(row["answer"])
    if not answer:
        return "narrative"
    evidence: set[float] = set()
    for e in row["evidence"]:
        evidence |= numbers(e["evidence_text"])
    if any(any_match({a}, e) for a in answer for e in evidence):
        return "direct"
    return "computed"


def load_questions() -> list[dict]:
    if not QUESTIONS.exists():
        sys.exit(
            f"no questions at {QUESTIONS}\n"
            "financebench must be cloned alongside this repo:\n"
            "  git clone https://github.com/patronus-ai/financebench"
        )
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    for r in rows:
        r["group"] = classify(r)
        r["oracle_doc"] = r["evidence"][0]["doc_name"] if r["evidence"] else None
        # Whether the answer is a bare figure or a sentence. Not used to
        # classify — the split must stay as eval-plan.md defines it — but
        # reported, because it turns out to decide what the direct group is
        # actually measuring. See the validity section of the summary.
        r["answer_words"] = len(re.findall(r"[A-Za-z]{3,}", r["answer"]))
    return rows


# --------------------------------------------------------------------------
# narrative overlap
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-z0-9]+")


def trigrams(text: str) -> set[tuple[str, str, str]]:
    """Word trigrams of normalised text.

    Character offsets are not an option: chunks store pages, not offsets, and
    Docling's text differs from the transcription in whitespace and word breaks
    ("Cash Flow s" for "Cash Flows"). Normalising to words and comparing
    trigrams survives that; exact alignment would not.
    """
    w = WORD_RE.findall(text.lower())
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def overlap(evidence: str, chunk: str) -> float:
    ev = trigrams(evidence)
    if not ev:
        return 0.0
    return len(ev & trigrams(chunk)) / len(ev)


# --------------------------------------------------------------------------
# scoring one question
# --------------------------------------------------------------------------


def score_direct(row: dict, hits: list) -> dict:
    """Rank of the first retrieved fact whose value matches the answer's."""
    target = numbers(row["answer"])
    for rank, (score, ftext, signed, scale, page, doc, fid) in enumerate(hits, 1):
        if any_match(target, abs(float(signed))):
            return {
                "rank": rank,
                "score": round(score, 4),
                "matched": ftext,
                "page": page,
                "found_doc": doc,
                "expected": sorted(target)[0] if target else None,
                "found": abs(float(signed)),
            }
    return {"rank": None, "expected": sorted(target)[0] if target else None}


def score_computed(row: dict, hits: list) -> dict:
    """Fraction of the evidence's numbers present among retrieved fact values.

    Approximate in one direction, and only one: evidence_text carries numbers
    the computation does not need, so coverage understates performance. Report
    the fraction, and report "any" beside "all" — "all inputs retrieved" is
    what the generator actually needs to compute an answer.
    """
    wanted: set[float] = set()
    for e in row["evidence"]:
        wanted |= numbers(e["evidence_text"])
    if not wanted:
        return {"coverage": {k: None for k in KS}, "n_inputs": 0}

    cov = {}
    for k in KS:
        found = {abs(float(h[2])) for h in hits[:k]}
        cov[k] = sum(1 for w in wanted if any(matches(w, f) for f in found)) / len(wanted)
    return {"coverage": cov, "n_inputs": len(wanted)}


def score_narrative(row: dict, hits: list) -> dict:
    """Best trigram overlap between the evidence and any retrieved chunk."""
    evidence = " ".join(e["evidence_text"] for e in row["evidence"])
    best, best_rank, best_page, best_doc = 0.0, None, None, None
    scores = []
    for rank, (cid, score, kind, page, item, context, doc, text) in enumerate(hits, 1):
        o = overlap(evidence, text)
        scores.append(o)
        if o > best:
            best, best_rank, best_page, best_doc = o, rank, page, doc
    return {
        "best_overlap": round(best, 4),
        "rank": best_rank if best >= TRIGRAM_THRESHOLD else None,
        "page": best_page,
        "found_doc": best_doc,
        "all_overlaps": scores,
    }


def evaluate(row: dict, condition: str, k: int) -> dict:
    """Score one question under one condition.

    Three conditions, and the middle one is the only shippable system:

    - **oracle**  restricted to the filing FinanceBench names. Uses a ground
      truth label, so it is a diagnostic ceiling and never a product number.
    - **routed**  restricted to whatever `resolve_scope` reads out of the
      question text. Uses nothing a real user would not supply.
    - **corpus**  unrestricted.

    corpus -> routed is what routing actually buys. routed -> oracle is what
    remains: the questions whose company or period the router could not read.
    """
    group = row["group"]
    doc = row["oracle_doc"] if condition == "oracle" else None
    company = fy = None
    if condition == "routed":
        company, fy = resolve_scope(row["question"])

    if group == "narrative":
        hits = search(row["question"], k=k, doc_name=doc, company=company, fiscal_year=fy)
        result = score_narrative(row, hits)
    else:
        hits = search_facts(row["question"], k=k, doc_name=doc, company=company, fiscal_year=fy)
        result = score_direct(row, hits) if group == "direct" else score_computed(row, hits)

    result |= {
        "id": row["financebench_id"],
        "group": group,
        "condition": condition,
        "company": row["company"],
        "oracle_doc": row["oracle_doc"],
        "n_retrieved": len(hits),
        "answer_words": row["answer_words"],
        "routed_company": company,
        "routed_fy": fy,
    }
    return result


# --------------------------------------------------------------------------
# aggregation and output
# --------------------------------------------------------------------------


def summarise(results: list[dict]) -> str:
    out = ["# Retrieval evaluation", ""]
    out.append(f"Run {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append("")
    out.append("Retrieval only. Three conditions:")
    out.append("")
    out.append("- **oracle** — restricted to the filing FinanceBench names. Uses a ground")
    out.append("  truth label, so it is a diagnostic ceiling, never a product number.")
    out.append("- **routed** — restricted to the company and fiscal year read out of the")
    out.append("  question text. Uses nothing a real user would not supply. This is the")
    out.append("  only shippable row.")
    out.append("- **corpus** — unrestricted.")
    out.append("")
    out.append("corpus to routed is what routing buys. routed to oracle is what remains:")
    out.append("questions whose company or period the router could not read.")
    out.append("")

    for group in ("direct", "computed", "narrative"):
        rows = [r for r in results if r["group"] == group]
        if not rows:
            continue
        out += [f"## {group} (n={len(rows) // max(1, len({r['condition'] for r in rows}))})", ""]

        if group == "computed":
            out.append("| condition | " + " | ".join(f"any@{k}" for k in KS)
                       + " | " + " | ".join(f"all@{k}" for k in KS) + " | mean cov@20 |")
            out.append("|---" * (2 + 2 * len(KS)) + "|")
        elif group == "narrative":
            out.append(f"| condition | hit@20 (overlap>={TRIGRAM_THRESHOLD}) | median overlap | p90 overlap |")
            out.append("|---|---|---|---|")
        else:
            out.append("| condition | " + " | ".join(f"R@{k}" for k in KS) + " | MRR |")
            out.append("|---" * (2 + len(KS)) + "|")

        for condition in ("oracle", "routed", "corpus"):
            sub = [r for r in rows if r["condition"] == condition]
            if not sub:
                continue
            if group == "direct":
                recall = [sum(1 for r in sub if r["rank"] and r["rank"] <= k) / len(sub) for k in KS]
                mrr = sum(1 / r["rank"] for r in sub if r["rank"]) / len(sub)
                out.append(f"| {condition} | " + " | ".join(f"{x:.3f}" for x in recall)
                           + f" | {mrr:.3f} |")
            elif group == "computed":
                anyk = [sum(1 for r in sub if (r["coverage"][k] or 0) > 0) / len(sub) for k in KS]
                allk = [sum(1 for r in sub if (r["coverage"][k] or 0) >= 0.999) / len(sub) for k in KS]
                mean20 = statistics.mean(r["coverage"][20] or 0 for r in sub)
                out.append(f"| {condition} | " + " | ".join(f"{x:.3f}" for x in anyk)
                           + " | " + " | ".join(f"{x:.3f}" for x in allk)
                           + f" | {mean20:.3f} |")
            else:
                hit = sum(1 for r in sub if r["rank"]) / len(sub)
                ov = sorted(r["best_overlap"] for r in sub)
                med = statistics.median(ov)
                p90 = ov[int(0.9 * (len(ov) - 1))]
                out.append(f"| {condition} | {hit:.3f} | {med:.3f} | {p90:.3f} |")
        out.append("")

    # ------------------------------------------------------------------
    # What the numbers above are actually measuring. Both of these were found
    # by running the harness, not by reading the plan, and both are properties
    # of how eval-plan.md defines the groups rather than of retrieval.
    # ------------------------------------------------------------------
    direct = [r for r in results if r["group"] == "direct" and r["condition"] == "routed"]
    computed = [r for r in results if r["group"] == "computed" and r["condition"] == "routed"]
    if direct or computed:
        out += ["## Measurement validity", ""]

    if direct:
        prose = [r for r in direct if r.get("answer_words", 0) >= 4]
        bare = [r for r in direct if r.get("answer_words", 0) < 4]
        hit = lambda rs: sum(1 for r in rs if r["rank"])  # noqa: E731
        out += [
            "**The direct group is mostly not a direct lookup.** A question lands in it",
            "when a number in its answer also appears in the evidence — but most answers",
            "are sentences that happen to contain one. \"The consumer segment shrunk by",
            "0.9% organically\" is scored by looking for the value 0.9; one answer yields",
            "1.5 from a bond coupon. Value-matching cannot measure those.",
            "",
            "| direct/routed | n | found@20 |",
            "|---|---|---|",
            f"| answer is a sentence (>=4 words) | {len(prose)} | "
            f"{hit(prose)}/{len(prose)} = {hit(prose) / max(1, len(prose)):.3f} |",
            f"| answer is a bare figure | {len(bare)} | "
            f"{hit(bare)}/{len(bare)} = {hit(bare) / max(1, len(bare)):.3f} |",
            "",
            "The second row is the real numeric-lookup rate. The headline figure is",
            "dominated by the first.",
            "",
        ]

    if computed:
        n = [r["n_inputs"] for r in computed if r.get("n_inputs")]
        if n:
            over = sum(1 for x in n if x > max(KS))
            out += [
                "**`all@k` for the computed group is unreachable by arithmetic, not by",
                f"retrieval quality.** `evidence_text` holds a median of {statistics.median(n):.0f} distinct",
                f"numbers (max {max(n)}), and {over} of {len(n)} questions need more than the k={max(KS)}",
                "facts retrieved. A score of 0.000 is the definition biting, not a result.",
                "Read `any@k` and mean coverage; treat `all@k` as broken until the metric",
                "targets the numbers a computation actually needs.",
                "",
            ]

    out += [
        "## Read these numbers with the known limits",
        "",
        "- A coincidental value match scores as a hit. 1,577 appears on several",
        "  pages of the 3M filing, some occurrences unrelated to capex.",
        "- Some questions have more than one defensible answer — net income 5,363",
        "  including noncontrolling interest against 5,349 attributable to 3M — so",
        "  a miss can be the reference being one of several valid values.",
        "- Computed coverage understates performance: `evidence_text` contains",
        "  numbers the computation does not need.",
        f"- The narrative threshold ({TRIGRAM_THRESHOLD}) is arbitrary, which is why the",
        "  overlap distribution is reported beside the hit rate.",
        "",
        "Per-question rows are in `rows.csv`. At n=70 for the largest group a few",
        "points is inside the noise, and the rows are what make a change explicable.",
    ]
    return "\n".join(out) + "\n"


FIELDS = [
    "id", "group", "condition", "company", "oracle_doc", "n_retrieved", "answer_words",
    "routed_company", "routed_fy",
    "rank", "score", "expected", "found", "matched", "page", "found_doc",
    "best_overlap", "n_inputs", "cov@1", "cov@5", "cov@10", "cov@20",
]


def write_rows(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            for k in KS:
                row[f"cov@{k}"] = (r.get("coverage") or {}).get(k)
            w.writerow(row)


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=max(KS), help="retrieval depth")
    ap.add_argument("--limit", type=int, help="first N questions; 0 = classify only")
    ap.add_argument("--group", choices=("direct", "computed", "narrative"))
    ap.add_argument("--condition", choices=("oracle", "routed", "corpus"))
    ap.add_argument("--out", type=Path, help="results directory")
    args = ap.parse_args()

    rows = load_questions()
    split = Counter(r["group"] for r in rows)
    print(f"questions    {len(rows)}")
    print(f"split        {dict(split)}")
    if dict(split) != EXPECTED_SPLIT:
        sys.exit(
            f"group split changed: expected {EXPECTED_SPLIT}, got {dict(split)}.\n"
            "The groups are derived from the data, so this means number parsing "
            "moved — fix that before trusting any metric below it."
        )
    print("split matches eval-plan.md")

    if args.limit == 0:
        return

    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    if args.limit:
        rows = rows[: args.limit]
    conditions = (args.condition,) if args.condition else ("oracle", "routed", "corpus")

    results = []
    total = len(rows) * len(conditions)
    for i, row in enumerate(rows, 1):
        for condition in conditions:
            results.append(evaluate(row, condition, args.k))
        print(f"  [{i}/{len(rows)}] {row['financebench_id']} {row['group']}", flush=True)

    out = args.out or RUNS / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out.mkdir(parents=True, exist_ok=True)
    write_rows(results, out / "rows.csv")
    (out / "summary.md").write_text(summarise(results))

    print(f"\n{total} rows -> {out}/rows.csv")
    print(f"summary      -> {out}/summary.md\n")
    print(summarise(results))


if __name__ == "__main__":
    main()
