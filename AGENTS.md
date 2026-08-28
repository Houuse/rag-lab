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
| Chunking, facts, embedding | `docling-extract`, a separate public repo installed as a package (ADR 0006) |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` via sentence-transformers, 768 dimensions |
| Data | PostgreSQL + pgvector, in Podman against a named volume |
| DB driver | psycopg 3 |
| Package manager | uv |

There is still no dependency manifest for the lab itself; the environment
exists only in `ingest/.venv`. What *is* pinned is everything chunking and
embedding touch, in `docling-extract`'s `pyproject.toml` — those pins are load
bearing, not hygiene. See the gotcha below.

## Directory index

| Path | What's there |
|---|---|
| `ingest/` | The pipeline. `convert.py` PDF→cached JSON, `probe.py` extraction diagnostics, `load.py` writes chunks+facts+embeddings into Postgres, `search.py` vector search, `batch.py` many documents. Chunking, fact extraction and embedding themselves live in `docling-extract` (ADR 0006) |
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
| Install/refresh the chunking package | `uv pip install --python .venv/bin/python -e ../docling-extract` (or `git+https://github.com/Houuse/docling-extract`) |
| Convert one PDF | `.venv/bin/python convert.py 3M_2018_10K` |
| Check extraction | `.venv/bin/python probe.py <stem> --evidence <financebench_id>` |
| Load into Postgres | `.venv/bin/python load.py <stem>` |
| Search | `.venv/bin/python search.py "question"`, or `--facts` for facts |
| Many documents | `.venv/bin/python batch.py --corpus 150 --workers 4 --threads 2` |
| Score retrieval | `.venv/bin/python eval.py` — all 150 questions, three conditions, under a minute. `--limit 0` classifies only |
| Ask a question | `.venv/bin/python ask.py "What was 3M's FY2018 capital expenditure?"` |
| Ask several | `.venv/bin/python ask.py --chat` |

## Asking it questions

The comfortable way is a real chat UI. `ingest/serve.py` exposes the pipeline as
an OpenAI-compatible API, so any chat client can front it:

```bash
cd ingest && .venv/bin/python serve.py          # http://localhost:8642/v1

pip install open-webui
OPENAI_API_BASE_URL=http://localhost:8642/v1 \
OPENAI_API_KEY=none ENABLE_OLLAMA_API=False open-webui serve
```

Then pick the `rag-lab` model. LibreChat, LM Studio and curl work the same way.

Point the UI at **serve.py, not at Ollama**. Ollama cannot reach the database,
so a UI talking straight to it gets a model answering SEC questions from
memory. Open WebUI's own RAG is deliberately unused for the same reason: it
would re-chunk the PDFs naively and discard the fact-per-table-cell store, the
page-accurate extraction and the citation checks.

Every reply carries a footer saying what was searched, and any grounding
failure appears in the answer itself rather than in a log nobody reads.

### From the terminal instead


Two things must be running: the database (`./db/run.sh`) and a local model
(`ollama serve` plus `ollama pull qwen2.5:7b-instruct`, which is the default).
Nothing leaves the machine and there is no per-question cost.

```bash
cd ingest
.venv/bin/python ask.py "What is the FY2018 capital expenditure for 3M?"
.venv/bin/python ask.py --chat          # follow-ups keep company and year
```

About 80 seconds per answer on CPU with the 7B. A larger model is slower
without being better at this: the task is to copy the right figure out of a
supplied list and cite it, not to reason.

| Flag | |
|---|---|
| `--chat` | session mode. `/context` toggles what the model was shown, `/reset` clears carried scope, `/quit` |
| `--show-context` | print the exact prompt, to see whether a wrong answer was the model's fault or retrieval's |
| `--no-generate` | route and retrieve only, no model. The fast way to check what would be found |
| `--facts N` `--chunks N` | how much is retrieved, default 15 and 3 |
| `--model` | any Ollama model |

### Reading the output

The two `stderr` lines before the answer are the audit trail:

```
route        numeric  company=3M  fy=2018     <- what it decided to search
retrieved    8 facts, 1 passages              <- how much it found
```

`company=-` means the question named no company it recognised, so the search
was corpus-wide — the commonest reason for a bad answer.

Lines beginning `!!` are grounding failures, and they are the point:

| | |
|---|---|
| `cited F123, which was not in the context` | invented a citation |
| `figure X is in no supplied fact` | computed or recalled a number instead of copying one |
| `no citations` | answered without evidence |

An answer with no `!!` lines had every figure and citation traced back to a
filed value. `INSUFFICIENT EVIDENCE` is a success, not a failure — the bar is
an exact figure with a page, or an explicit refusal.

None of this judges whether the answer is *correct*. It checks the answer
stayed inside its evidence. Correctness across all 150 questions is not
measured yet; `eval.py` scores retrieval only.

`RAGLAB_DSN` overrides the connection string. It defaults to
`postgresql://raglab:raglab@localhost:5433/raglab` — port 5433, not 5432,
because 5432 is usually already taken and that failure surfaces much later as
a hung connection.

See `docs/system/batch-runs.md` for why the worker and thread numbers are what
they are.

## Gotchas

- **No dependency manifest for the lab.** Recreating `ingest/.venv` means
  reading the imports. Everything on the chunking and embedding path is pinned
  in `docling-extract/pyproject.toml`, and those pins matter: the embedding
  cache is keyed by an md5 of the chunk and fact strings, and chunk boundaries
  come from `docling_core`'s HybridChunker and the HuggingFace tokenizer. A
  different version of either produces different text, a different key, a cache
  that silently misses, and vectors incomparable with the ones already stored.
- **Embedding can run on another machine.** It is the expensive half and the
  half a GPU helps with — the 2% GPU figure in `docs/system/batch-runs.md` is
  about conversion, not the pipeline. On the machine with the hardware:
  `./setup.sh --embed` in `docling-extract`. Copy the `*.emb.*.npy` files next
  to the JSON here and `load.py` skips encoding entirely.
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
