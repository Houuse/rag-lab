# AGENTS.md

## What is this

A lab for building question-answering over SEC filings, held to the standards
of a production tool rather than a demo. It answers two different kinds of
question — exact numeric lookups ("3M's FY2018 capital expenditure") and
narrative ones ("why did operating cash flow fall") — from a corpus of PDF
filings. The quality bar is an exact figure with a page citation or an explicit
refusal. FinanceBench's 150 labelled questions are the regression suite, not
the product.

## Stack

| Piece | Choice |
|---|---|
| Language | Python 3.12, pinned — torch publishes no CUDA wheels for 3.14 |
| PDF extraction | Docling (layout model + TableFormer) |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` via sentence-transformers, 768 dimensions |
| Data | PostgreSQL + pgvector, in Podman against a named volume |
| DB driver | psycopg 3 |
| Package manager | uv |

There is no dependency manifest. The environment exists only in `ingest/.venv`
and cannot be reproduced from the repo.

## Directory index

| Path | What's there |
|---|---|
| `ingest/` | The pipeline. `convert.py` PDF→cached JSON, `probe.py` extraction diagnostics, `load.py` chunks+facts+embeddings into Postgres, `search.py` vector search, `batch.py` many documents |
| `ingest/cache/` | Conversion and embedding output, plus per-batch resume files. Gitignored, over 1 GB |
| `db/` | `run.sh` starts the pgvector container against a named volume; `schema.sql` is idempotent and applied on every start |
| `financebench/` | The corpus and the 150 questions. A separate clone of `patronus-ai/financebench`, gitignored |
| `AI-Labs/` | Earlier C# scaffolding, unused by the pipeline. Separate repo, gitignored |
| `docs/` | Durable project context. Sub-folder layout below shows where each kind of doc goes. |

```
docs/
├── system/         ← what the code does today (updated as code changes)
├── architecture/   ← what the system must do (updated when rules change)
├── adr/            ← architecture decisions (immutable once shipped)
├── reference/      ← long-form rationale (append-only)
└── working-notes/  ← active research, in motion until promoted
```

Only `system/` and `adr/` exist today. `docs/` also holds `roadmap.md`,
`eval-plan.md` and `when-rag-fits.md` at its root, from before that layout.

## Commands

Run from `ingest/` unless stated. Every command needs the venv's interpreter —
there is no `python` on PATH and the dependencies live only in `ingest/.venv`.

| What | Command |
|---|---|
| Start the database | `./db/run.sh` from the repo root. Must be running before `load.py` or `search.py`. `--recreate` destroys the volume |
| Convert one PDF | `.venv/bin/python convert.py 3M_2018_10K` |
| Check extraction | `.venv/bin/python probe.py <stem> --evidence <financebench_id>` |
| Load into Postgres | `.venv/bin/python load.py <stem>` |
| Search | `.venv/bin/python search.py "question"`, or `--facts` for facts |
| Many documents | `.venv/bin/python batch.py --corpus 150 --workers 4 --threads 2` |

`RAGLAB_DSN` overrides the connection string. It defaults to
`postgresql://raglab:raglab@localhost:5433/raglab` — port 5433, not 5432,
because 5432 is usually already taken and that failure surfaces much later as
a hung connection.

See `docs/system/batch-runs.md` for why the worker and thread numbers are what
they are.

## Gotchas

- **No dependency manifest.** Recreating the environment means reading the
  imports. `docling-extract` has pins if you need a reference.
- **`financebench/` must be cloned separately** —
  `git clone https://github.com/patronus-ai/financebench` alongside this repo.
  Nothing in `ingest/` works without it.
- **`probe.py` must not be renamed to `inspect.py`.** A file named
  `inspect.py` in the script directory shadows the standard library module and
  breaks Docling's imports. The obvious name is the broken one.
- **Several failures here are silent** — text truncated without error, degraded
  retrieval from a wrong embedding prefix, values stored without their scale.
  `docs/system/ingestion.md` lists the confirmed ones. Assume a new
  integration point fails quietly until checked.
