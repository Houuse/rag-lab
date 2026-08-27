"""Load a cached DoclingDocument into Postgres: documents, chunks, facts.

    # step 3 alone — no DB, no embeddings
    python load.py 3M_2018_10K.accurate --facts-only --table 40

    # everything, printed, still no DB
    python load.py 3M_2018_10K.accurate --dry-run

    # for real
    python load.py 3M_2018_10K.accurate

Implements ADRs 0001-0004: every table becomes both a chunk and a set of fact
rows, prose is chunked by HybridChunker, embeddings are nomic-768, and both
representations are written in one transaction so they cannot drift.

Chunking, fact extraction and embedding now live in the `docling-extract`
package (ADR 0006) so they can run on a machine with a GPU and no database.
What remains here is everything that knows about Postgres.
"""

import argparse
import os
import sys
from pathlib import Path

from docling_extract.chunking import (
    Chunk,
    DocMeta,
    Fact,
    doc_meta,
    fact_text,
    item_sections,
    period_end,
    prose_chunks,
    scrub,
    table_chunks,
    table_facts,
)
from docling_extract.embedding import embed

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

DSN = os.environ.get(
    "RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab"
)



def write(meta: DocMeta, chunks: list[Chunk], facts: list[Fact], vecs) -> None:
    import psycopg

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (doc_name, company, doc_type, fiscal_year, period_end,
                 source_path, pdf_pages)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (doc_name) DO UPDATE SET ingested_at = now()
            RETURNING doc_id
            """,
            (
                meta.doc_name,
                meta.company,
                meta.doc_type,
                meta.fiscal_year,
                meta.period_end,
                meta.source_path,
                meta.pdf_pages,
            ),
        )
        doc_id = cur.fetchone()[0]

        # Re-loading a document replaces its rows rather than duplicating them.
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM facts  WHERE doc_id = %s", (doc_id,))

        table_chunk_id: dict[int, int] = {}
        for ch, vec in zip(chunks, vecs, strict=True):
            cur.execute(
                """
                INSERT INTO chunks
                    (doc_id, kind, item_section, heading_trail, page_start,
                     page_end, table_ordinal, text, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING chunk_id
                """,
                (
                    doc_id,
                    ch.kind,
                    ch.item_section,
                    ch.heading_trail or None,
                    ch.page_start,
                    ch.page_end,
                    ch.table_ordinal,
                    ch.text,
                    "[" + ",".join(f"{x:.6f}" for x in vec) + "]",
                ),
            )
            cid = cur.fetchone()[0]
            if ch.kind == "table" and ch.table_ordinal is not None:
                table_chunk_id[ch.table_ordinal] = cid

        # The fact string needs the heading trail of its table's chunk, which is
        # only known now, so facts are embedded here rather than alongside
        # chunks.
        trail_of = {
            c.table_ordinal: (c.heading_trail[-1] if c.heading_trail else None)
            for c in chunks
            if c.kind == "table"
        }
        fact_texts = [
            fact_text(meta.company, trail_of.get(f.table_ordinal), f) for f in facts
        ]
        fact_vecs = embed(fact_texts, f"{meta.doc_name}.facts", CACHE) if facts else []

        for f, ftext, fvec in zip(facts, fact_texts, fact_vecs, strict=True):
            cur.execute(
                """
                INSERT INTO facts
                    (doc_id, chunk_id, table_ordinal, row_label, column_label,
                     value_raw, value, parenthesized, scale, unit, page,
                     fact_text, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (doc_id, table_ordinal, row_label, column_label)
                DO NOTHING
                """,
                (
                    doc_id,
                    table_chunk_id.get(f.table_ordinal),
                    f.table_ordinal,
                    f.row_label,
                    f.column_label,
                    f.value_raw,
                    f.value,
                    f.parenthesized,
                    f.scale,
                    f.unit,
                    f.page,
                    ftext,
                    "[" + ",".join(f"{x:.6f}" for x in fvec) + "]",
                ),
            )
        conn.commit()
    print(f"doc_id={doc_id}  {len(chunks)} chunks  {len(facts)} facts")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="cache stem, e.g. 3M_2018_10K.accurate")
    ap.add_argument("--facts-only", action="store_true")
    ap.add_argument("--table", type=int, help="with --facts-only: just this table")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    from docling_core.types.doc.document import DoclingDocument

    path = CACHE / f"{args.stem}.json"
    if not path.exists():
        sys.exit(f"no cached document at {path}")
    doc = DoclingDocument.load_from_json(path)

    if args.facts_only:
        idxs = [args.table] if args.table is not None else range(len(doc.tables))
        for i in idxs:
            fs = table_facts(doc, i)
            if not fs:
                continue
            print(f"\n--- table {i}  page {fs[0].page}  scale={fs[0].scale}")
            for f in fs:
                sign = "-" if f.parenthesized else " "
                print(
                    f"  {f.column_label:>10}  {sign}{f.value:>14,.2f}  "
                    f"{f.value_raw:>12}  {f.row_label[:58]}"
                )
            print(f"  ({len(fs)} facts)")
        return

    meta = doc_meta(args.stem, doc)
    meta.period_end = period_end(doc, meta.fiscal_year)
    meta.source_path = str(path)
    sections = item_sections(doc)

    chunks = prose_chunks(doc, sections) + table_chunks(doc, sections)
    facts = [f for i in range(len(doc.tables)) for f in table_facts(doc, i)]

    # Scrub every string bound for a text column, before embedding, so the
    # vector and the stored text come from the same characters.
    for c in chunks:
        c.text = scrub(c.text)
        c.item_section = scrub(c.item_section) or None
        c.heading_trail = [scrub(h) for h in c.heading_trail]
    for f in facts:
        f.row_label = scrub(f.row_label)
        f.column_label = scrub(f.column_label)
        f.value_raw = scrub(f.value_raw)
    chunks = [c for c in chunks if c.text.strip()]

    print(f"{meta.doc_name}: {meta.company} / {meta.doc_type} / FY{meta.fiscal_year}")
    print(f"period_end   {meta.period_end}")
    print(f"item sections {sorted(set(sections.values()))[:12]}")
    print(f"chunks       {len(chunks)}  ({sum(c.kind == 'prose' for c in chunks)} prose,"
          f" {sum(c.kind == 'table' for c in chunks)} table)")
    print(f"facts        {len(facts)}")

    if args.dry_run:
        for c in chunks[:3]:
            print(f"\n[{c.kind} p{c.page_start} {c.item_section}] {c.text[:160]!r}")
        return

    vecs = embed([c.text for c in chunks], args.stem, CACHE)
    write(meta, chunks, facts, vecs)


if __name__ == "__main__":
    main()
