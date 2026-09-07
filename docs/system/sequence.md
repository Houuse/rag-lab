# Sequence — a question through the pipeline

Traced from the code, not the intent: `ask.py:answer_once` is the spine, and
every arrow below is a call you can find in `ask.py`, `search.py` or
`metrics.py`. `serve.py` runs the same spine for the chat UI.

The two things a demo audience usually asks about are marked: **the model is
never given a vector** (step 6 sends text), and **the model never does
arithmetic** (step 4 computes in code, step 9 checks it didn't).

## Per question

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant A as ask.py / serve.py
    participant M as metrics.py
    participant E as nomic-embed-text-v1.5
    participant P as Postgres + pgvector
    participant L as llama.cpp (Qwen2.5-7B)

    U->>A: "What was 3M's FY2018 capital expenditure?"

    Note over A: route() — no model call
    A->>A: resolve_scope() → company=3M, fy=2018<br/>NUMERIC_RE → kind=numeric
    A->>A: expand() — prepend previous question if this is a follow-up
    A->>A: expand_terms() — append the filing's own words for jargon<br/>(the model still sees the question as asked)

    Note over A,M: arithmetic happens here or not at all (ADR 0001)
    A->>M: detect(question) → e.g. "gross margin"
    M->>P: SELECT the input facts
    P-->>M: numerator, denominator
    M-->>A: Computed(result, inputs) — the model is handed<br/>the answer and its inputs — it never divides

    Note over A: BUDGET by route decides k before anything is fetched —<br/>numeric 10 facts/3 chunks, narrative 4 facts/10 chunks

    A->>E: embed_query("search_query: " + question)
    E-->>A: 768-dim vector
    Note right of E: the whole question is embedded, company<br/>and year included — nothing is parsed out

    A->>P: facts:  ORDER BY embedding <=> $vec LIMIT k_facts
    P-->>A: k nearest facts — fact_text, signed, scale, page
    A->>P: chunks: ORDER BY embedding <=> $vec LIMIT k_chunks
    P-->>A: k nearest passages — text, heading_trail, page_start
    Note right of P: --retrieval hybrid adds a second, sequential<br/>ts_rank(tsv) query per table, each fetched to<br/>depth = max(2k, 20) and fused by RRF 1/(60+rank).<br/>Off by default until measured end to end.

    Note over A,P: company/fiscal_year are a WHERE on the joined<br/>documents row — they narrow what is eligible,<br/>they never rank. Distance alone selects.
    Note right of P: with a filter: MATERIALIZED CTE, exact scan.<br/>HNSW filters after finding neighbours, so a<br/>selective filter can return 0 rows and no error.

    alt nothing matched under the filters
        A->>P: retry without the fiscal-year filter
        Note right of A: a filter matching nothing is worse than no filter
    end
    A->>A: render() — question + retrieved TEXT into one prompt

    A->>L: POST /v1/chat/completions (plain text, temperature 0)
    Note right of L: llama.cpp sees no vector and cannot<br/>reach the database. Text in, text out.
    L-->>A: "…$1,577 million [F13028], page 126."

    Note over A: verify() — deterministic code, not a second model
    A->>A: every [Fnnn] cited must be in the context
    A->>A: every figure must appear in a supplied fact
    A-->>U: answer + citations, or INSUFFICIENT EVIDENCE<br/>plus any !! grounding failure
```

## Ingestion, once per filing

```mermaid
sequenceDiagram
    autonumber
    participant B as batch.py
    participant C as convert.py
    participant D as Docling (layout + TableFormer)
    participant X as docling-extract
    participant E as nomic-embed-text-v1.5
    participant P as Postgres + pgvector

    B->>C: 3M_2018_10K.pdf
    C->>D: convert
    D-->>C: layout + tables
    C->>C: cache JSON (conversion is the expensive, repeatable half)
    B->>X: chunk + extract facts
    Note right of X: one fact per table cell —<br/>row_label, column_label, value, page
    X->>E: encode chunk text and rendered fact sentences
    E-->>X: 768-dim vectors (cached, keyed by md5 of the text)
    B->>P: load.py — chunks, facts, embeddings, HNSW + GIN indexes
```

## Why the pieces are split this way

| Question | Answer |
|---|---|
| Does the model search the database? | No. It never sees a vector and holds no connection. `ask.py` retrieves, then hands it text. |
| Why not let the UI talk to Ollama directly? | It would answer SEC questions from memory. The UI talks to `serve.py`, which owns retrieval. |
| Why not Open WebUI's own RAG? | It would re-chunk the PDFs naively and discard the per-table-cell facts, the page numbers and the grounding checks. |
| Who does the arithmetic? | `metrics.py`, in code, before the prompt is built. The model selects a figure; it never computes one. |
| What stops a hallucinated number? | `verify()` — a figure not present in any supplied fact is reported as `!!`. It checks grounding, not correctness. |

## Rendering these

`docs/system/img/` holds PNGs of both diagrams, for slides. To regenerate
after editing the Mermaid above:

```bash
npx -y @mermaid-js/mermaid-cli -i diagram.mmd -o out.png -w 1800 -b white \
    -p pconf.json        # pconf.json: {"args":["--no-sandbox","--disable-setuid-sandbox"]}
```

The `-p` flag is not optional on this machine: Chromium's sandbox needs
unprivileged user namespaces, which AppArmor denies here, and without it
mermaid-cli fails with `No usable sandbox!`.

Mermaid treats `;` as a statement separator, so a semicolon inside message
text is a parse error — and the reported line number is the message's, which
makes it look like the arrow is malformed. Use a comma or a dash.
