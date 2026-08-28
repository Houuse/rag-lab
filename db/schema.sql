-- rag-lab schema. Idempotent: run.sh applies it on every start.
--
-- Two representations of every table live here (ADR 0001) and are joined by
-- facts.chunk_id, so a fact always knows which chunk it came from and the two
-- cannot silently diverge.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy row-label matching ("capex")

-- One row per filing.
CREATE TABLE IF NOT EXISTS documents (
    doc_id       bigserial PRIMARY KEY,
    doc_name     text NOT NULL UNIQUE,          -- e.g. 3M_2018_10K
    company      text NOT NULL,
    cik          text,                          -- for the XBRL path later
    doc_type     text NOT NULL,                 -- 10k | 10q | 8k | earnings
    fiscal_year  int,
    -- period_end is the real period identity: a column headed "2016" means
    -- calendar year-end for 3M but May 2016 for Nike. Extracted from the
    -- statement's own "Years ended ..." line, not from the column header.
    period_end   date,
    filing_date  date,
    source_path  text NOT NULL,
    pdf_pages    int,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

-- Retrievable units: prose from HybridChunker, plus one serialized chunk per
-- table (ADR 0002, ADR 0003).
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      bigserial PRIMARY KEY,
    doc_id        bigint NOT NULL REFERENCES documents ON DELETE CASCADE,
    kind          text NOT NULL CHECK (kind IN ('prose', 'table')),
    item_section  text,                         -- Item 1A, Item 7, Item 8, ...
    heading_trail text[],                       -- from the chunker
    page_start    int,
    page_end      int,
    table_ordinal int,                          -- doc.tables index, when kind='table'
    text          text NOT NULL,
    -- 768 dims, nomic-embed-text-v1.5. Text is stored WITHOUT the
    -- "search_document: " prefix; the prefix is applied at embed time only.
    embedding     vector(768),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

-- Dedupe on a hash of the text, not the text: a btree index row cannot exceed
-- ~2704 bytes and chunks here reach 28 KB.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_dedupe
    ON chunks (doc_id, kind, table_ordinal, page_start, md5(text));

-- One row per numeric table cell.
CREATE TABLE IF NOT EXISTS facts (
    fact_id       bigserial PRIMARY KEY,
    doc_id        bigint NOT NULL REFERENCES documents ON DELETE CASCADE,
    chunk_id      bigint REFERENCES chunks ON DELETE SET NULL,
    table_ordinal int NOT NULL,
    row_label     text NOT NULL,                -- "Purchases of property, plant and equipment (PP&E)"
    column_label  text NOT NULL,                -- "2018"
    period_end    date,                         -- resolved, when known
    -- Sign convention is deliberately not baked in (ADR 0001 leaves it open).
    -- value_raw keeps what the filing printed; value is the magnitude;
    -- parenthesized records accounting negation; signed applies it.
    value_raw     text NOT NULL,                -- "(1,577)"
    value         numeric NOT NULL,             -- 1577
    parenthesized boolean NOT NULL DEFAULT false,
    signed        numeric GENERATED ALWAYS AS
                      (CASE WHEN parenthesized THEN -value ELSE value END) STORED,
    scale         text,                         -- millions | thousands | billions | null
    unit          text,                         -- USD | shares | percent | null
    page          int NOT NULL,
    -- A fact is its own retrievable unit (ADR 0005). fact_text is the string
    -- that was embedded, kept so a result is explicable and so a change to the
    -- rendering is detectable.
    fact_text     text,
    embedding     vector(768),
    UNIQUE (doc_id, table_ordinal, row_label, column_label)
);

-- Approximate nearest neighbour, cosine. Building this after a bulk load is
-- faster than maintaining it during one; drop and recreate for large ingests.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin      ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc          ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_item_section ON chunks (item_section);

CREATE INDEX IF NOT EXISTS facts_embedding_hnsw
    ON facts USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS facts_lookup    ON facts (doc_id, column_label);
CREATE INDEX IF NOT EXISTS facts_row_trgm  ON facts USING gin (row_label gin_trgm_ops);
-- Lexical search over fact strings. Functional rather than a generated tsv
-- column: adding a column rewrites the table, an index does not, which
-- matters when 387k rows are already loaded and being queried.
CREATE INDEX IF NOT EXISTS facts_text_gin  ON facts USING gin (to_tsvector('english', fact_text));
CREATE INDEX IF NOT EXISTS facts_period    ON facts (period_end);

-- Convenience view: a fact with everything needed to cite it.
CREATE OR REPLACE VIEW facts_cited AS
SELECT f.fact_id,
       d.company,
       d.doc_name,
       d.doc_type,
       d.period_end AS doc_period_end,
       f.row_label,
       f.column_label,
       f.value_raw,
       f.signed,
       f.scale,
       f.unit,
       f.page
FROM facts f
JOIN documents d USING (doc_id);
