# 0004. Postgres with pgvector as the single store

*2026-08-25*

## Context

ADR 0001 requires two representations of every table that must not drift. Separate stores (numpy index plus SQLite, or a vector DB plus a relational DB) make that a two-phase write with no transaction across it. Postgres holds vectors via pgvector, facts relationally, and full-text search via `tsvector` — which also supplies the lexical half of hybrid retrieval without a second system.

## Decision

Run `pgvector/pgvector:pg17` in Podman with a named volume for durability. Chunks with `vector(768)` and an HNSW cosine index, facts as an ordinary table, `tsvector` generated column with a GIN index for lexical search.

## Consequences

Both table representations are written in one transaction, so they cannot drift. Hybrid retrieval needs no extra infrastructure. Provenance joins are ordinary SQL.

HNSW is approximate, so recall is not guaranteed and its build parameters affect measured retrieval quality — a confound to control when comparing retrievers. Exact search is available via a sequential scan on this corpus size if a ground-truth comparison is needed.

The database is now a runtime dependency of every experiment; a stopped container means no retrieval at all.
