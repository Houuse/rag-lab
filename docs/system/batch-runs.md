# Running the batch

## It is CPU-bound, and oversubscription is slower than sequential

One document uses about five of eight cores. Parallelism helps only until the
cores are full, so `--workers` and `--threads` must multiply to roughly the
core count.

Measured: `--workers 6` with the default `--threads 8` — 48 threads on 8 cores
— completed 5 documents in 72 minutes, where sequential would take about 55.
`--workers 4 --threads 2` gives about 6.3 minutes per document, close to the
floor implied by 52 core-minutes of work per document.

Memory is not the binding constraint. Six workers peaked well inside 30 GB.

## A discrete GPU adds little

On a machine with an NVIDIA dGPU the GPU sat at 2% while the CPU worked.
Getting CUDA torch installed at all requires pinning Python to 3.12 — there is
no CUDA wheel for 3.14, and the resolver silently installs a CPU-only build
that fails later with "Torch not compiled with CUDA enabled".

## Three independent resume points

Rerunning `batch.py` with the same arguments is always safe.

| Layer | Granularity | Location |
|---|---|---|
| Conversion | 12-page batch | `ingest/cache/parts/<stem>/` |
| Embedding | whole document, keyed by content hash | `ingest/cache/<stem>.emb.<hash>.npy` |
| Load | whole document | skipped if its facts are already embedded in Postgres |

The embedding cache is keyed on the concatenated text, so a failed database
write costs no re-encoding, but any change to chunking invalidates it.

## Corpus selection

`--corpus N` takes the documents the questions reference, then fills to N with
other filings by the *same companies*. A distractor only tests retrieval if it
is confusable: 3M's 2016 10-K beside its 2018 one differs mainly in its
numbers. An unrelated company tests nothing.
