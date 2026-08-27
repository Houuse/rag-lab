# Ingestion hazards

Confirmed while building `ingest/`. Most of these fail silently.

## Text can be truncated with no error

`HybridChunker` ignores the token budget of a plain `transformers` tokenizer.
It needs `docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer`
with an explicit `max_tokens`. Handed a raw tokenizer it produced a
10,481-token chunk against an 8,192 limit.

`SentenceTransformer` carries its own `max_seq_length`, independent of the
model's `model_max_length`. If it is lower, long text is truncated at encode
time with no warning at all. `load.py` raises it explicitly.

Both failures produce embeddings of text that was cut short, with nothing in
the output to say so.

## PDF text can contain NUL bytes

The checkbox glyph on a 10-K cover page renders as `\x00` in some PDFs' font
encoding and as a normal character in others. PostgreSQL rejects NUL in text
columns outright, so a whole filing fails to load over four characters.
`scrub()` in `load.py` strips them, applied before embedding as well as before
insert so the vector and the stored string come from identical text.

## FinanceBench page numbers are off by one

`evidence_page_num` is one less than the physical page: its "59" is page 60.
The PDF's own printed labels match the physical pages, so the offset is in the
dataset, not the document. Locate evidence pages by content, not by the stated
number.

## A table's identity is not inside the table

Five business segments each carry a row labelled `Sales (millions)` for 2018
with different values. Nothing in those tables names the segment — that is a
heading printed above them. Facts are therefore stored with the heading trail
of their table's chunk, and a fact without one cannot be disambiguated.

Docling classifies identically formatted headings inconsistently: four of the
five segment headings came back as `section_header`, the fifth as `text`.
`table_heading_trails()` also accepts short lines ending in a colon.

## Scale markers appear in three places

`(Millions)` may be the table's corner cell, somewhere in the header row, or
inside the row label itself (`Sales (millions)`) — segment tables use the last.
Resolution order is row label, then table, then short text items on the page.
Facts without a scale are not necessarily wrong; roughly 40% of them come from
tables with no monetary scale at all.
