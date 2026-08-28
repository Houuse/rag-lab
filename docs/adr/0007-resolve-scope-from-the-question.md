# 0007. Company and period are predicates, read from the question

*2026-08-28*

## Context

ADR 0005 made each fact its own retrievable unit and said plainly that the
period problem was untouched by it: "capital expenditures 2016" still ranked 20
and returned 2018 values.

Loading the full corpus turned that from a footnote into the main failure. With
63 documents, "3M capital expenditures 2018" put the correct 1,577 at rank 1.
With 366 — including 3M's 2015 through 2022 filings — it fell to rank 3, behind
1,699 and 1,436 from other years. Nothing regressed; the earlier number was
flattering because the confusable neighbours had not been loaded.

The embedding cannot fix this. Measured on our own model, adding a year to a
query moves the similarity by 0.014, which is noise. Several filings by one
company differ mainly in their numbers, so they are near-identical to a vector.

Measured across all 150 questions, restricting retrieval to the filing
FinanceBench names roughly doubles what is found: direct R@20 0.157 to 0.286,
computed any@20 0.661 to 0.857. That is the size of the prize, and it is only
available to something that knows which company and which year.

## Decision

Read the company and the fiscal year out of the question text and apply them as
SQL predicates before ranking. `search.resolve_scope` normalises the question
and matches against the companies actually in the corpus, longest name first,
then takes a year, preferring an explicit `FY2018` over a bare `2018`.

Parse the question, never FinanceBench's `company` or `doc_name` fields. Those
are labels the benchmark supplies and a user does not. Using them at query time
would measure a system that cannot exist, and would make every later number a
lie by construction.

Either value may come back `None`, and `None` means no filter on that axis
rather than a guess. Narrowing to the wrong company is worse than not
narrowing: the answer becomes unreachable at any k.

The eval gains a third condition, `routed`, between the `oracle` ceiling and
the unrestricted `corpus` floor. It is the only row that describes a shippable
system, and it exists so that nobody quotes the oracle number.

## Consequences

Measured, routed against corpus: computed any@20 0.661 to 0.875, narrative
hit@20 0.417 to 0.458, direct R@20 0.157 to 0.186. On the computed group
routing reaches the oracle ceiling and slightly passes it, because company plus
year admits sibling filings and an input the named document lacks is sometimes
in one of them.

The router resolves both axes for 49 of 70 direct questions and 10 of 24
narrative ones. The rest fall back to a broad search, and that unresolved
remainder is most of the gap left to oracle.

A filtered vector search had to be made exact. An HNSW scan finds its nearest
candidates first and filters afterwards, so a selective predicate can return
nothing: filtering facts to one company pulled 45 candidates from the index,
none matching, and returned zero rows against 15,113 qualifying facts. Whether
that happens is the planner's choice — filtering to one document happened to
produce a correct exact scan, filtering by company did not. Filtered searches
now materialise their candidate set and scan it exhaustively.

This corpus is 366 curated filings where company plus year nearly always
identifies one document. Against 10,000 the predicate is far less selective, so
the gain measured here overstates the production gain.

Jargon is untouched and this does not address it. "capex" fails because the
filing says "Purchases of property, plant and equipment" — a scope predicate
cannot help, and neither can lexical matching. That needs a synonym step.
