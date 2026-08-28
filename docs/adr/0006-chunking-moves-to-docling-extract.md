# 0006. Chunking, facts and embedding move to docling-extract

*2026-08-27*

## Context

Relocates the implementation of ADRs 0001, 0002, 0005 and the embedding half of 0003. Those decisions stand unchanged; only where the code lives changes.

Ingestion runs two neural models. Conversion — Docling's layout model and TableFormer — was already split into `docling-extract` so it could run on whatever machine has the hardware. Embedding stayed in `load.py`, on the machine with the database.

That machine has no GPU. Measured on the 303 remaining documents: four CPU workers spent 35 minutes on the first four filings without committing one, projecting to roughly 88 hours for the corpus. The documents are not outliers — they average 149 pages and 86 tables, against 160 and 116 for the 3M 10-K used as the reference.

`docs/system/batch-runs.md` records the GPU at 2% utilisation and concludes a discrete GPU adds little. That measurement is about docling. Embedding is a few hundred thousand batched transformer forward passes, which is the workload a GPU suits. The conclusion was generalised from one model to a pipeline running two.

Running embedding elsewhere needs the chunk and fact strings, and those come from `load.py`. Three ways to get them onto the GPU machine:

- **Copy the chunking logic into `docling-extract`.** Two copies of the code the embedding cache's md5 is computed over. They drift, and the failure is silent: `.npy` files written, a different key computed, everything re-embedded on CPU with no error.
- **Have `docling-extract` import `rag-lab`.** `docling-extract` is public and `rag-lab` is private; the dependency would not resolve for anyone else.
- **Move the code down into `docling-extract` and have `rag-lab` import it.**

## Decision

Move chunking, fact extraction and embedding into the `docling_extract` package. `rag-lab` depends on it.

The seam is storage. `docling_extract.chunking` turns a `DoclingDocument` into chunks and facts; `docling_extract.embedding` turns strings into vectors cached by content hash. Neither knows a database exists. `load.py` keeps `write()` and `main()` — everything that knows about Postgres — and imports the rest.

The strings these functions return are a wire format. They key the embedding cache and determine every stored vector, so changing them invalidates both.

## Consequences

The GPU machine needs one repository. `./setup.sh --embed` converts nothing and embeds what is already converted, reusing the CUDA detection the conversion path already had.

`rag-lab` gains a git dependency on a public repository. This is the first time the lab depends on something it does not contain.

Decisions the ADRs describe are now implemented in a public tool. The reasoning stays here; `docling_extract/chunking.py` points back at ADRs 0001, 0002 and 0005 rather than restating them. Anyone reading the tool alone gets the mechanism without the argument.

`search.py` no longer keeps its own copy of the model name and query prefix. Indexing with one model and querying with another returns plausible nonsense rather than an error, and two constants in two files is how that happens.

`pyproject.toml` replaces `requirements.txt` as the single pinned list, with `docling` behind a `convert` extra — chunking and embedding work from JSON and do not need it, which is what lets the two halves run on different machines.

Verified before committing: chunk keys, fact keys, counts and sampled strings are byte-identical across 13 documents spanning 10-K, 10-Q, 8-K and earnings, from 7 to 675 chunks. `load.py` reuses `.npy` files written by the pre-move code, and `embed.py` writes files `load.py` then reuses. Byte-identity was the acceptance criterion: a drift would invalidate every cached embedding and put a chunking confound across the 63 documents already loaded.

## Correction, 2026-08-28

"That machine has no GPU" is wrong. It has Intel Lunar Lake integrated
graphics, with Vulkan available and a render node present. The claim came from
`nvidia-smi` not being installed, which only rules out NVIDIA.

The decision stands, and so does the measurement behind it: Ollama's build
ships CUDA and CPU backends only, with no Vulkan or SYCL, so nothing in this
pipeline could have used that iGPU whatever the premise said. An integrated GPU
sharing LPDDR5X with the CPU would in any case be bandwidth-bound and worth
perhaps 2-3x, not the order of magnitude the discrete card gave.

Recorded rather than edited away, because the reasoning is what a later reader
needs to judge, and a premise checked with the wrong tool is exactly the kind of
mistake worth leaving visible.
