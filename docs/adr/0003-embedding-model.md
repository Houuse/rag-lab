# 0003. Embed with nomic-embed-text-v1.5

*2026-08-25*

## Context

The BGE and E5 families cap at 512 tokens, which splits serialized financial tables — the tables ADR 0001 keeps whole. `nomic-embed-text-v1.5` takes 8192 tokens at 768 dimensions, Apache 2.0, runs locally. Anthropic has no embeddings endpoint.

## Decision

Use `nomic-ai/nomic-embed-text-v1.5` via sentence-transformers, loaded with `trust_remote_code=True`. Set the HybridChunker's token budget from this model's tokenizer.

## Consequences

A whole financial statement fits in one chunk. Matryoshka training means the index can later be truncated to 512/256/128 dimensions without re-embedding.

The mandatory `search_document:` and `search_query:` prefixes are asymmetric and fail silently when wrong, degrading retrieval with no error. They must be applied inside the indexing and query code, never left to callers.

`trust_remote_code=True` executes model-repo code, and is required until transformers v5.5 / sentence-transformers v5.3.
