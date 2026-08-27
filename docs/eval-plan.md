# Eval plan

Measures retrieval only. Generation and judging come later.

## The 150 questions split three ways

Value-matching is valid for one group and invalid for another, so they are
scored separately and never blended into one number.

| Group | n | Ground truth |
|---|---|---|
| Direct | 70 | the answer's value appears in `evidence_text` |
| Computed | 56 | the answer is a ratio or growth rate; the value appears nowhere in the filing |
| Narrative | 24 | the answer is not numeric |

Counts come from comparing the numbers in each `answer` against the numbers in
its `evidence_text`, allowing for thousand/million scaling.

## Group 1 — direct (70)

Retrieve top-k facts for the question text. A hit is a fact whose absolute
value matches the answer's, tolerating scale (`v`, `v*1000`, `v/1000`).

Report Recall@k for k in 1, 5, 10, 20; MRR; and the rank of the first hit per
question.

Two known weaknesses, which is why per-question output is mandatory:

A coincidental value match scores as a hit. 1,577 appears on several pages of
the 3M filing, some occurrences unrelated to capex.

Some questions have more than one defensible answer. "Net income 2018" is
5,363 including noncontrolling interest and 5,349 attributable to 3M. A miss
may be the reference being one of several valid values, not a retrieval
failure.

## Group 2 — computed (56)

The answer value is not in the filing, so the target changes: score the
*inputs*. Metric is the fraction of the distinctive numbers in `evidence_text`
that appear among the retrieved facts' values, at each k.

This is approximate in one direction: `evidence_text` contains numbers the
computation does not need, so coverage understates performance. Report the
fraction rather than a pass/fail, and report "at least one input retrieved"
alongside "all inputs retrieved" — the second is what the generator actually
needs.

## Group 3 — narrative (24)

No character-offset alignment. Chunks store pages, not offsets, and Docling's
text differs from the transcription in whitespace and word breaks
(`Cash Flow s` for `Cash Flows`), so exact alignment is fragile.

Instead: normalise both texts (lowercase, collapse whitespace, strip
punctuation) and compute what fraction of `evidence_text`'s word trigrams
appear in each retrieved chunk. A chunk is a hit above a fixed threshold.
Report the threshold used and the distribution of overlap scores, not only the
hit rate — a metric whose result depends on an arbitrary cut-off must show
where the cut-off sits.

## Two conditions, always reported as a pair

**Oracle document.** Restrict retrieval to the document FinanceBench names for
that question. Measures retrieval quality with entity and period resolution
removed.

**Full corpus.** No restriction. Measures the system as a user would meet it.

The gap between the two is the cost of entity and period resolution, which is
the failure the measurements so far point at (`capital expenditures 2016`
returning 2018 values). Reporting only the oracle condition would hide it;
reporting only the full-corpus condition would confound it with ranking
quality.

## Output

One row per question — id, group, condition, rank or coverage, the matched
fact's label and page, expected and found values — written to CSV, plus a
summary table per group and condition. Aggregates alone are not enough: at
n=70 for the largest group, a few points of difference is inside the noise,
and the per-question rows are what make a change explicable.

## Not in scope

Answer correctness, groundedness and the LLM judge. Those need the generator,
and the judge needs its own validation before its scores mean anything.
