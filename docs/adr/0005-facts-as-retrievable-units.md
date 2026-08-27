# 0005. Facts are retrievable units, not SQL lookups

*2026-08-26*

## Context

Revises the retrieval mechanism in ADR 0001; the decision to store tables twice stands.

ADR 0001 sent numeric questions to a SQL lookup. That needs a hand-curated alias table mapping "capex" to filer row labels, plus company and period predicates — a report with extra steps, and not retrieval.

Measured instead: embedding each fact as a short string ("3M · Business Segment Information · Total Company · Capital Expenditures · $ 1,577") moved the correct answer for "3M capital expenditures 2018" from rank 10 to rank 1. Whole-table chunks failed because a line item is a fortieth of the text; in a fact string it is most of it.

## Decision

Embed every fact as its own retrievable unit: `facts.fact_text` plus `facts.embedding vector(768)`, HNSW cosine index. Numeric questions retrieve facts by nearest neighbour, not by SQL predicate. Exactness comes from the retrieved unit being a filed value with its page, not from the query being SQL.

## Consequences

No alias table and no metric dictionary to maintain. The model is handed a value and a page rather than a paragraph to read a number out of.

Two failures are untouched because they have different causes: the period is still near-invisible ("capital expenditures 2016" ranks 20, returning 2018 values) and jargon still fails ("capex" ranks 62). Neither is a granularity problem.

Facts are embedded in a second pass, after chunks, because a fact string needs its table's heading trail.

Retrieval now returns the same figure under several labels — `Capital Expenditures` and `Capital Spending` both give 1,577 — with inconsistent signs against the cash flow statement's `(1,577)`. Deduplication and the sign convention move into the augment stage.
