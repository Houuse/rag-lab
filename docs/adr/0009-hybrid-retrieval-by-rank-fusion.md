# 0009. Vector and keyword search are fused by rank, not by score

*2026-08-28*

## Context

An embedding matches meaning, which is what makes it useful and what makes it
miss. Asked for AMCOR's Adjusted EBITDA it returns four *Adjusted EBIT* rows —
a different line item with a different value — and never surfaces the figure at
all. The vector is behaving correctly: EBIT and EBITDA mean nearly the same
thing. They are not the same number.

A financial statement is full of names that must match exactly rather than
approximately. "Adjusted EBITDA", "Net cash provided by operating activities",
"Total current liabilities" are labels, not concepts to be approached.

## Decision

Run both and fuse them by reciprocal rank. `search.lexical` and
`search.lexical_facts` use Postgres full-text search; `hybrid` and
`hybrid_facts` fuse each with its vector counterpart, scoring an item
`1/(60 + rank)` in every list it appears in.

Rank fusion rather than a weighted sum of scores. Cosine similarity here runs
0.5 to 0.85 and is dense; `ts_rank` runs near zero and is sparse. Any weighting
of the two is really a weighting of their scales, and tuning that weight means
tuning against whatever the current corpus does to those distributions. Ranks
have no scale to get wrong.

Each list is fetched deeper than k before fusing, because an item ranked 15th
by one method and 3rd by the other should surface, and cannot if both lists
stop at k.

Facts get a functional GIN index on `to_tsvector('english', fact_text)` rather
than a generated `tsv` column. Adding a column rewrites the table; an index
does not, and 387k rows were loaded and being queried at the time.

## Consequences

Not the default. It is built, and one demonstration is not a result: measuring
it needs a full eval run, which the answer evaluation had the CPU for. It sits
behind `--retrieval hybrid` until that number exists. A change to retrieval
that has not been through the harness is a guess, and the harness was built
precisely so guesses stop being shipped.

On the one case examined, AMCOR FY2023 Adjusted EBITDA, the correct 2,018 moves
from absent to the top three.

`websearch_to_tsquery` ANDs its terms, so passing a whole question demands one
fact containing "what", "was", "in" *and* "FY2023", and it returned zero rows
for every real question. A two-word probe passed and hid this completely.
Content words are now OR-ed and `ts_rank` orders the result, which means
lexical search here is a bag-of-words scorer rather than a phrase matcher.

This does not fix jargon, and it was twice claimed during the day that it
would. "capex" fails because the filing says "Purchases of property, plant and
equipment" and never says capex. Lexical matching finds nothing that is not
literally present. That needs a synonym or query-rewriting step, which is a
separate decision.
