# 0001. Dual ingestion of tables: chunks and a fact store

*2026-08-25*

## Context

A table's embedding is dominated by its row labels, so "3M capex 2018" and "3M capex 2016" retrieve the same chunk. Vector search finds the statement, never the cell. Chunks-only lets the generator misread figures undetectably; fact-store-only leaves tables unsearchable.

## Decision

Ingest every table twice: as vector chunks, and as fact records of company, metric, period, value, scale, provenance. A router sends numeric questions to lookup, the rest to search.

## Consequences

Figures become exact and page-cited; arithmetic happens in code. Ingestion doubles per table and the two copies can drift. The router becomes the new failure mode and needs its own eval. Still undecided: mapping "capex" to filer-specific row labels, and the sign of `(1,577)`.
