"""Compute derived financial metrics in code, from filed values.

    python metrics.py 3M 2018 --metric "current ratio"
    python metrics.py 3M 2018            # everything computable

ADR 0001 forbids the model from calculating, and 56 of FinanceBench's 150
questions — 37% — ask for a ratio or a margin whose value appears in no filing.
Until arithmetic exists somewhere, the correct behaviour on all of them is to
refuse, which is honest and answers nothing. This is that somewhere.

The rule it preserves: every number in an answer traces to a table cell. A
computed metric is not exempt, it just traces to several — so a result carries
the inputs it came from, each with its fact id and page, and the arithmetic is
Python rather than a language model's impression of division.

Two things make the lookup trustworthy rather than approximate:

**Inputs must come from one table.** A current ratio built from assets on one
page and liabilities on another may be mixing a consolidated statement with a
segment note, and would be wrong in a way nothing downstream could detect.
Candidate tables are those containing *every* input for the period; a metric
with no such table is not computed at all.

**The period is matched on the column, not assumed.** A balance sheet shows two
years side by side, and `period_end` is unresolved for most facts, so the
fiscal year is matched against `column_label`. No match, no answer.
"""

import argparse
import os
import re
from dataclasses import dataclass

DSN = os.environ.get("RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab")


# Canonical inputs, and the row labels filings actually use for them. Order
# matters: the first pattern that matches wins, so the most specific comes
# first. "Total revenue" before "revenue" — otherwise a segment revenue line
# satisfies a metric that wanted the consolidated one.
LINE_ITEMS: dict[str, list[str]] = {
    "current_assets":      ["total current assets"],
    "current_liabilities": ["total current liabilities"],
    "cash":                ["cash and cash equivalents", "cash and equivalents"],
    # ILIKE matches the whole value, so a pattern without a trailing wildcard
    # is an exact match. "accounts receivable%net" found nothing: the real
    # label is "Accounts receivable - net of allowances of $95 and $103".
    "receivables":         ["accounts receivable%", "receivables - net%",
                            "receivables%"],
    "inventory":           ["total inventories", "inventories", "inventory%net"],
    "revenue":             ["total net sales", "net sales", "total revenues",
                            "total revenue", "revenues", "revenue"],
    "cogs":                ["cost of sales", "cost of goods sold",
                            "total cost of sales"],
    "gross_profit":        ["gross profit", "gross margin"],
    "operating_income":    ["operating income", "income from operations",
                            "operating income (loss)"],
    "net_income":          ["net income", "net income (loss)",
                            "net income attributable%"],
    "total_assets":        ["total assets"],
    "total_liabilities":   ["total liabilities"],
    "equity":              ["total stockholders%equity", "total shareholders%equity",
                            "total equity"],
}


@dataclass
class Metric:
    inputs: list[str]
    formula: object
    fmt: str
    note: str = ""


# Deliberately small. Each entry is a claim about what a metric means, and a
# wrong formula produces a confident wrong number with real citations attached
# — worse than the refusal it replaces. Add one only with its definition
# checked, not from memory.
METRICS: dict[str, Metric] = {
    "current ratio": Metric(
        ["current_assets", "current_liabilities"],
        lambda a, b: a / b, "{:.2f}",
        "total current assets / total current liabilities"),
    "quick ratio": Metric(
        ["cash", "receivables", "current_liabilities"],
        lambda c, r, l: (c + r) / l, "{:.2f}",
        "(cash + receivables) / total current liabilities; excludes inventory, "
        "so it differs from a definition that subtracts inventory from current "
        "assets when there are other current assets"),
    "working capital": Metric(
        ["current_assets", "current_liabilities"],
        lambda a, b: a - b, "{:,.0f}",
        "total current assets - total current liabilities"),
    "gross margin": Metric(
        ["gross_profit", "revenue"],
        lambda g, r: 100 * g / r, "{:.1f}%",
        "gross profit / revenue"),
    "operating margin": Metric(
        ["operating_income", "revenue"],
        lambda o, r: 100 * o / r, "{:.1f}%",
        "operating income / revenue"),
    "net margin": Metric(
        ["net_income", "revenue"],
        lambda n, r: 100 * n / r, "{:.1f}%",
        "net income / revenue"),
    "debt to equity": Metric(
        ["total_liabilities", "equity"],
        lambda d, e: d / e, "{:.2f}",
        "total liabilities / total equity"),
    "return on assets": Metric(
        ["net_income", "total_assets"],
        lambda n, a: 100 * n / a, "{:.1f}%",
        "net income / total assets; not averaged over the year"),
}


def _candidates(cur, company: str, fiscal_year: int, key: str):
    """Every fact that could be this line item, for this company and year."""
    out = []
    for rank, pat in enumerate(LINE_ITEMS[key]):
        cur.execute(
            """
            SELECT f.fact_id, f.doc_id, f.table_ordinal, f.row_label,
                   f.column_label, f.signed, f.scale, f.page, d.doc_name
            FROM facts f JOIN documents d USING (doc_id)
            WHERE d.company = %s AND d.fiscal_year = %s
              AND f.row_label ILIKE %s
              AND f.column_label ILIKE %s
            """,
            (company, fiscal_year, pat, f"%{fiscal_year}%"),
        )
        for row in cur.fetchall():
            out.append((rank, *row))
    return out


@dataclass
class Computed:
    metric: str
    value: float
    formatted: str
    note: str
    scale: str | None
    inputs: list  # (key, fact_id, row_label, value, page, doc_name)


def compute(company: str, fiscal_year: int, metric: str) -> Computed | None:
    """One metric, or None if the filings do not support it.

    None is a real answer. A metric whose inputs are not all present in one
    table of one filing is not computable from what was extracted, and saying
    so beats assembling a number from wherever the pieces happened to be.
    """
    import psycopg

    spec = METRICS[metric]
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        found = {k: _candidates(cur, company, fiscal_year, k) for k in spec.inputs}

    if any(not v for v in found.values()):
        return None

    # Group by the table each candidate came from, and keep only tables that
    # supply every input. This is what stops a consolidated figure being
    # divided by a segment one.
    tables: dict[tuple, dict] = {}
    for key, rows in found.items():
        for rank, fid, doc_id, tbl, row_label, col, value, scale, page, doc in rows:
            slot = tables.setdefault((doc_id, tbl), {})
            prev = slot.get(key)
            # Best pattern rank wins; ties break on the shorter label, which is
            # the plainer line item rather than a qualified variant.
            if prev is None or (rank, len(row_label)) < (prev[0], len(prev[2])):
                slot[key] = (rank, fid, row_label, float(value), page, doc, scale)

    complete = [(t, s) for t, s in tables.items() if len(s) == len(spec.inputs)]
    if not complete:
        return None
    # Prefer the table whose labels matched the most specific patterns.
    _, slot = min(complete, key=lambda ts: sum(v[0] for v in ts[1].values()))

    values = [slot[k][3] for k in spec.inputs]
    try:
        result = spec.formula(*values)
    except ZeroDivisionError:
        return None

    return Computed(
        metric=metric,
        value=result,
        formatted=spec.fmt.format(result),
        note=spec.note,
        scale=slot[spec.inputs[0]][6],
        inputs=[(k, slot[k][1], slot[k][2], slot[k][3], slot[k][4], slot[k][5])
                for k in spec.inputs],
    )


# Longest first: "operating margin" must be tried before "margin" would be, and
# a question saying "current ratio" must not match "ratio" alone.
_METRIC_RE = re.compile(
    "|".join(re.escape(m) for m in sorted(METRICS, key=len, reverse=True)), re.I
)


def detect(question: str) -> str | None:
    m = _METRIC_RE.search(question.lower())
    return m.group(0).lower() if m else None


def render(c: Computed) -> str:
    """The result, its formula, and every input it came from."""
    unit = f" {c.scale}" if c.scale and c.metric == "working capital" else ""
    lines = [f"{c.metric}: {c.formatted}{unit}", f"  {c.note}", "  computed from:"]
    for key, fid, label, value, page, doc in c.inputs:
        lines.append(f"    [F{fid}] {label} = {value:,.0f}  ({doc} p{page})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("company", help="as stored, e.g. 3M or JOHNSON JOHNSON")
    ap.add_argument("fiscal_year", type=int)
    ap.add_argument("--metric", help="default: everything computable")
    args = ap.parse_args()

    wanted = [args.metric] if args.metric else list(METRICS)
    any_ok = False
    for m in wanted:
        if m not in METRICS:
            print(f"unknown metric {m!r}; known: {', '.join(METRICS)}")
            return
        c = compute(args.company, args.fiscal_year, m)
        if c:
            any_ok = True
            print(render(c))
            print()
        elif args.metric:
            print(f"{m}: not computable — the filings do not carry all inputs "
                  f"in one table for {args.company} {args.fiscal_year}")
    if not any_ok and not args.metric:
        print(f"nothing computable for {args.company} {args.fiscal_year}")


if __name__ == "__main__":
    main()
