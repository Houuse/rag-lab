# Roadmap

    ingest → retrieve → route → augment → generate → evaluate

## ingest — done, 366 of 368 documents

    366 documents    122,396 chunks    387,378 facts

`ingest/convert.py` turns a PDF into a cached DoclingDocument (~5 min per
160-page filing, batched and resumable). `ingest/probe.py` inspects that cache
and grades extraction against FinanceBench's transcriptions. `ingest/load.py`
writes documents, chunks and facts to Postgres in one transaction. Chunking,
fact extraction and embedding live in `docling-extract` (ADR 0006).

The two missing documents are `INTEL_2023_8K_dated-2023-02-10` and
`INTEL_2023_8K_dated-2023-08-16`, both truncated in the FinanceBench clone —
no `%%EOF`, no `startxref`, the file ends mid-xref-table. They are the only
two of 368 in that state and no question references either, so they are
distractors and the gap is cosmetic. Re-cloning the corpus would fix it.

Open: `column_label` on facts is not a normalised period, so a lookup cannot
filter by year without string matching. Scale resolves for about 60% of facts;
the rest are mostly non-monetary tables. Two of 116 tables have no heading
trail.

## retrieve — 3 of 5

Vector search over chunks, vector search over facts, and the metadata
pre-filter are built. Lexical search over `tsv` and hybrid fusion are not.

`search.resolve_scope` reads a company and a fiscal year out of the question
text and turns them into SQL predicates. It parses the question and never
FinanceBench's `company` or `doc_name` fields — those are labels the benchmark
supplies and a user does not, so using them at query time would measure a
system that cannot exist.

Measured across all 150 questions (the `routed` condition below), against the
unrestricted baseline: computed `any@20` 0.661 to 0.875, narrative hit@20 0.417
to 0.458, direct R@20 0.157 to 0.186. On the computed group routing reaches the
oracle ceiling, and slightly passes it — filtering to company plus year admits
a few sibling filings, and an input the named document lacks is sometimes in
one of them.

Still open on this stage: the router resolves both company and year for 49 of
70 direct questions and 10 of 24 narrative ones, so the rest fall back to a
broader search. Jargon is untouched — "capex" still fails, because it needs
lexical matching or query rewriting, not a filter.

Caveat worth keeping: this corpus is 366 curated filings where company plus
year nearly always identifies one document. Against 10,000 filings the same
predicate is far less selective, so the gain measured here overstates the gain
in production.

## route — built

`ask.route` classifies a question as numeric or narrative by rule, and resolves
its scope. Both kinds get facts and chunks: the sentence around a number is
what makes it checkable, and a narrative answer usually needs figures in it.
That makes a misroute degrade the answer rather than break it, which is why a
rule is enough here and a second model call is not.

## augment — built

`ask.render` assembles the context. Every line carries an id — `[F12345]` for a
fact, `[C678]` for a chunk — and the answer is required to cite them, which is
what makes `ask.verify` possible.

Still open: context budget and what gets dropped, ordering, deduplication of
the same figure appearing on several pages, and the sign convention for
`(1,577)`.

Original note: chunk IDs must be visible in the context or citations cannot
be checked. Facts must be rendered as text, and that rendering determines
whether the sign of `(1,577)` survives. Also: context budget and what gets
dropped, ordering, deduplication of the same figure appearing on several pages,
and the refusal instruction. Groundedness is made possible or impossible here.

## generate — built, unmeasured

`ask.py` generates against a local model over Ollama, so no filing leaves the
machine and there is no per-question cost. Temperature 0 and a fixed seed,
because an answer that changes between runs cannot be regression-tested.

Grounding is checked after generation rather than requested in the prompt.
`ask.verify` rejects a citation that was never supplied, an answer carrying no
citations, and any figure appearing in no retrieved fact — that last one
catches the model computing or recalling, which is the failure ADR 0001 exists
to prevent. A prompt asking for citations is a hope; this is a check.

Not yet measured. Answer correctness needs a judge, and the judge needs its own
validation before its scores mean anything.

## evaluate — harness built, baseline recorded

`ingest/eval.py` scores all 150 questions in both conditions and writes
`ingest/eval-runs/<timestamp>/{rows.csv,summary.md}`. Under a minute for a full
run. Implements `docs/eval-plan.md`; the group split is derived from the data
and asserted against the documented 70/56/24, so a change to number parsing
fails loudly instead of quietly reshaping every metric.

Baseline, 2026-08-28, 366 documents:

| group | condition | headline |
|---|---|---|
| direct (70) | oracle | R@1 0.086, R@20 0.286, MRR 0.121 |
| direct (70) | corpus | R@1 0.029, R@20 0.157, MRR 0.063 |
| computed (56) | oracle | any@20 0.857, mean coverage 0.103 |
| computed (56) | corpus | any@20 0.661, mean coverage 0.050 |
| narrative (24) | oracle | hit@20 0.500, median overlap 0.230 |
| narrative (24) | corpus | hit@20 0.417, median overlap 0.058 |

Oracle beats corpus on every metric. That gap is the cost of entity and period
resolution and is the number the metadata pre-filter has to move.

Two defects in the metrics, found by running them, reported in every
`summary.md` rather than left to be rediscovered:

- **The direct group is mostly not a direct lookup.** A question joins it when
  a number in its answer also appears in the evidence, but 62 of 70 answers are
  sentences that merely contain one — "The consumer segment shrunk by 0.9%
  organically" is scored by hunting for the value 0.9, and one answer yields
  1.5 from a bond coupon. The 8 answers that are bare figures hit 5/8 at k=20,
  against 15/62 for the rest. The headline is dominated by questions
  value-matching cannot measure.
- **`all@k` for computed is unreachable by arithmetic.** `evidence_text` holds
  a median of 88 distinct numbers and 49 of 56 questions need more than k=20,
  so 0.000 is the definition biting rather than a result.

Fixing either means revising `eval-plan.md`, so the baseline above implements
it as written and flags the problem instead of quietly redefining the metric.

Also surfaced: 19 of 366 documents produced zero facts, two of them documents
the questions reference (`AMCOR_2022_8K_dated-2022-07-01`,
`FOOTLOCKER_2022_8K_dated_2022-08-19`). A numeric question about those cannot
be answered through the fact path at all.
