# rag-lab

Question answering over SEC filings, held to the standard of a production tool
rather than a demo.

It answers two kinds of question — exact numeric lookups ("3M's FY2018 capital
expenditure") and narrative ones ("why did operating cash flow fall") — from a
corpus of PDF filings. The bar is **an exact figure with a page citation, or an
explicit refusal**. A confidently wrong number is worse than no number, because
someone acts on it.

Everything runs locally. No API keys, no per-question cost, nothing leaves the
machine.

```
$ ask.py "What was 3M's FY2018 capital expenditure?"
route        numeric  company=3M  fy=2018
retrieved    10 facts, 3 passages
3M's FY2018 capital expenditure was $1,577 million [F13028], page 126.
```

## How it answers

Three things enforce the quality bar, and none of them is the prompt:

1. **Every figure the model may use is a table cell.** At ingest, each cell
   becomes a `facts` row — row label, column label, value, page. The model
   selects a figure; it never computes one.
2. **Arithmetic happens in Python or not at all.** Derived metrics are computed
   before the prompt exists, and the model is handed the result with its
   inputs.
3. **Grounding is checked after generation, by code.** Every citation must
   exist in the context; every figure must appear in a supplied fact. Failures
   are reported in the answer, not a log.

`INSUFFICIENT EVIDENCE` is a success, not a failure.

None of this judges whether the answer is *correct* — only that it stayed
inside its evidence. See `docs/system/sequence.md` for the full path a question
takes.

## Setup

| Prerequisite | |
|---|---|
| Python 3.12 | pinned — torch publishes no CUDA wheels for 3.14 |
| Podman | runs the pgvector container |
| [uv](https://docs.astral.sh/uv/) | optional; `pip install -r` works, `uv` is faster |
| [llama.cpp](https://llama.app/docs/installation) | the model server, installed below |
| [Open WebUI](https://docs.openwebui.com/) | optional, only for the chat UI |

```bash
git clone https://github.com/Houuse/rag-lab && cd rag-lab/ingest

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python git+https://github.com/Houuse/docling-extract
```

**Fetch the extracted corpus** rather than converting 366 PDFs yourself, which
is the expensive half of the pipeline and takes days:

```bash
./fetch-cache.sh          # 5.2 GB
```

That pulls [`Houusee/rag-lab-artifacts`](https://huggingface.co/datasets/Houusee/rag-lab-artifacts)
— 732 Docling JSONs, 778 precomputed embedding files, 366 markdown renderings.
`load.py` skips encoding entirely when the `.npy` files are present.

**Start the database and load it:**

```bash
../db/run.sh                              # pgvector in Podman, port 5433
.venv/bin/python load.py 3M_2018_10K      # one filing
.venv/bin/python batch.py --corpus 150    # or the whole corpus
```

**Start a model server.** llama.cpp is the default because it supports more
GPUs — Ollama's build ships CUDA and CPU backends only, while llama.cpp adds
Vulkan, Metal and ROCm, so AMD and Intel cards get used instead of falling back
to the CPU.

Install [llama.cpp](https://llama.app/docs/installation) — the one-liner
detects your hardware and fetches the matching prebuilt binary:

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Then this pulls the model and serves it on `:8080`, which is where `ask.py`
looks by default:

```bash
llama serve -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M -ngl 99 -c 8192
```

Ollama works as a fallback on the CPU:

```bash
ollama serve && ollama pull qwen2.5:7b-instruct
.venv/bin/python ask.py "..." --backend ollama --host http://localhost:11434
```

**Ask:**

```bash
.venv/bin/python ask.py "What was 3M's FY2018 capital expenditure?"
.venv/bin/python ask.py --chat            # follow-ups keep company and year
```

### A chat UI instead

`serve.py` exposes the pipeline as an OpenAI-compatible API, so any chat client
can front it:

```bash
.venv/bin/python serve.py                 # http://localhost:8642/v1
```

Any OpenAI-compatible client works — LibreChat, LM Studio, curl.
[Open WebUI](https://docs.openwebui.com/) installs with pip, into its own
environment rather than this one:

```bash
uv venv --python 3.12 ~/openwebui && ~/openwebui/bin/pip install open-webui

OPENAI_API_BASE_URL=http://localhost:8642/v1 \
OPENAI_API_KEY=none ENABLE_OLLAMA_API=False \
    ~/openwebui/bin/open-webui serve        # http://localhost:3000
```

Then pick the `rag-lab` model. Point the UI at **serve.py, not at Ollama** — a
UI talking straight to a model gets SEC questions answered from memory, and
Open WebUI's own RAG would re-chunk the PDFs naively and discard the
per-table-cell facts, the page-accurate extraction and the grounding checks.

### Only if you are re-converting or evaluating

The PDFs and the labelled questions live in a separate repo:

```bash
git clone https://github.com/patronus-ai/financebench   # alongside this repo
```

`convert.py` and `batch.py` need the PDFs; `eval.py`, `answer_eval.py` and
`probe.py --evidence` need the questions. With the cache fetched, `load.py` and
`ask.py` need neither.

## Where it stands

[FinanceBench](https://github.com/patronus-ai/financebench)'s 150 labelled
questions are the regression suite, not the product. Each carries a golden
answer, the evidence strings it should come from, and a written justification
— which is what makes it possible to tell a retrieval failure from a
generation failure. Also on HuggingFace as
[`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench),
and described in [the paper](https://arxiv.org/abs/2311.11944).

`eval.py` scores retrieval; `answer_eval.py` scores answers end to end.

Retrieval is deliberately incomplete — see `docs/roadmap.md`. Vector search
over chunks and facts plus a metadata pre-filter are built; hybrid rank fusion
exists behind `--retrieval hybrid` but is off by default until it is measured,
and there is no reranker. Jargon the filings never print is a known gap: "capex"
ranks poorly because the line item reads "Purchases of property, plant and
equipment".

## Documentation

| | |
|---|---|
| `AGENTS.md` | commands, gotchas, and the silent failures |
| `docs/system/` | what the code does today |
| `docs/adr/` | why it is built this way |
| `docs/roadmap.md` | what is built and what is not |
| `docs/when-rag-fits.md` | when a report beats RAG in finance |

## Stack

Python 3.12 · Docling (layout + TableFormer) · `nomic-embed-text-v1.5`, 768
dimensions · PostgreSQL + pgvector · psycopg 3 · llama.cpp or Ollama

## License

MIT — see `LICENSE`, and it covers this code and the derived artifacts only.
The filings themselves are public SEC documents. The corpus selection, the
questions and the golden answers come from
[FinanceBench](https://github.com/patronus-ai/financebench) under its own
terms, and are not redistributed here.
