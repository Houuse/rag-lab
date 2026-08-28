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

## A discrete GPU adds little to *conversion*

On a machine with an NVIDIA dGPU the GPU sat at 2% while the CPU worked.
Getting CUDA torch installed at all requires pinning Python to 3.12 — there is
no CUDA wheel for 3.14, and the resolver silently installs a CPU-only build
that fails later with "Torch not compiled with CUDA enabled".

This section used to say "a discrete GPU adds little", full stop. That
generalised one measurement of one model to a pipeline that runs two, and it
cost a day.

Ingestion embeds as well as converts, and embedding is the workload a GPU
suits: a few hundred thousand batched transformer forward passes. Measured on
the database machine, which has no GPU — four workers, 35 minutes, the first
four filings, nothing committed. The remaining 303 documents average 149 pages
and 86 tables, against 160 and 116 for the 3M 10-K, so they are typical rather
than outliers. Extrapolated: about 88 hours.

Embedding now lives in `docling-extract` (ADR 0006) and runs where the hardware
is: `./setup.sh --embed`. Copy the `*.emb.*.npy` files back next to the JSON
and `load.py` finds them cached.

Measure the two models separately. They have nothing in common but a
dependency on torch.

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
