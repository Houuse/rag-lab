# How it works — demo walkthrough

The path from a typed question to an answer, in order.

## Ingestion (done ahead of time, not per-question)

1. **Extract** — Docling turns each PDF into prose chunks and per-table-cell
   facts (e.g. `row_label="Purchases of PP&E", column_label="2018", value=1577`).
2. **Embed** — each chunk's text and each fact's rendered sentence is run
   through `nomic-embed-text-v1.5`, producing a 768-number vector.
3. **Store** — text, metadata (page, doc, year), and vector all land in
   Postgres (`chunks` and `facts` tables, `pgvector` extension).

## Per question (what happens when you hit enter)

1. **Route** — parse the question for company, fiscal year, and whether it's
   a numeric lookup or a narrative question. No model call yet.
2. **Expand** — short follow-ups ("and 2019?") get the previous question's
   words prepended, so they're searchable on their own.
3. **Embed the question** — same embedding model as ingestion turns the
   question into its own 768-number vector.
4. **Retrieve** — that vector is compared against every stored vector in
   Postgres via `ORDER BY embedding <=> $query LIMIT N` (cosine distance,
   sped up by an HNSW index). Back comes the N closest facts/chunks, as text.
5. **Assemble the prompt** — the question plus the retrieved facts/chunks
   (their original text, not the vectors) get glued into one prompt string.
6. **Generate** — the prompt is sent to the LLM server (llama.cpp on the
   iGPU, or Ollama as a fallback) over a plain chat-completion HTTP call.
   The model never sees a vector or touches the database — just text in,
   text out.
7. **Verify grounding** — every citation and figure in the answer is checked
   against what was actually retrieved: does the cited fact ID exist in the
   context, does the number appear in a supplied fact? This is deterministic
   code, not a second model. Flags fabricated citations or computed/recalled
   numbers.
8. **Answer** — printed (or returned to the chat UI) with its citations, or
   `INSUFFICIENT EVIDENCE` if nothing grounded supports an answer.

## Who's who

| Piece | Role |
|---|---|
| Postgres + pgvector | Stores text + vectors, does the similarity search |
| `nomic-embed-text-v1.5` | Turns text into vectors — for storage *and* for the question, so they're comparable |
| `ask.py` / `serve.py` | The orchestrator — routes, embeds, retrieves, builds the prompt, calls the LLM, checks grounding |
| llama.cpp | Generates text from a prompt. Knows nothing about vectors, routing, or the database |
| Open WebUI (or any chat client) | Just a text box — talks to `serve.py` as if it were "a model" |

## The one-sentence version

**Retrieval** (Postgres + embeddings) finds the right evidence;
**generation** (llama.cpp) turns it into a sentence; **grounding checks**
make sure the sentence didn't say anything the evidence doesn't support.
