# 0002. Chunk prose with Docling's HybridChunker, tag with Item section

*2026-08-25*

## Context

Fixed-size chunking splits mid-sentence and needs tuning to defend. Item sections (`Item 1A`, `Item 7`) are real boundaries but an MD&A is far too large for one chunk, so they are metadata, not a chunker.

## Decision

Use `docling_core.transforms.chunker.hybrid_chunker.HybridChunker` for boundaries. Tag every chunk with its enclosing Item section, alongside the heading trail and page provenance the chunker already supplies.

## Consequences

Chunks carry citations and are filterable by section without extra work. The chunker's tokenizer must match the embedding model or the token budget is wrong. No fixed-size baseline exists to compare against, so chunking cannot be shown to be better than the obvious alternative.
