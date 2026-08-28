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

## retrieve — next

Five pieces: vector search over chunks, lexical search over `tsv`, hybrid
fusion of the two, fact lookup by row label, and a metadata pre-filter on
company and period. The pre-filter is what stops a 2016 filing answering a
2018 question.

Build vector search first, then the eval harness, then add the rest so each
addition is a measured before/after.

## route

Decide whether a question is a fact lookup or a narrative search. A numeric
question sent down the chunk path produces a plausible unverified figure,
which is the failure ADR 0001 exists to prevent, so this needs its own
evaluation separate from retrieval quality.

## augment

Prompt assembly. Chunk IDs must be visible in the context or citations cannot
be checked. Facts must be rendered as text, and that rendering determines
whether the sign of `(1,577)` survives. Also: context budget and what gets
dropped, ordering, deduplication of the same figure appearing on several pages,
and the refusal instruction. Groundedness is made possible or impossible here.

## generate

Answer with citations, refuse when the context does not support an answer.
Arithmetic happens in code, not in the model.

## evaluate

The 150 FinanceBench questions as a regression suite. Retrieval is scored
against `evidence_text` character spans, not page numbers, so the metric is
chunker-agnostic; report recall at a fixed context budget so larger chunks
cannot win by size alone. Numeric answers are checked exactly. Keep
per-question results, not just aggregates: at n=150 a few points of difference
is inside the noise.
