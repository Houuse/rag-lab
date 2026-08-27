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
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM = 768
EMBED_MAX_TOKENS = 8192
DOC_PREFIX = "search_document: "  # ADR 0003: asymmetric, mandatory, silent if wrong

DSN = os.environ.get(
    "RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab"
)

ITEM_RE = re.compile(r"^\s*item\s+(\d+[A-Z]?)\b[.:]?\s*(.*)", re.I)
# Filings write this many ways: "(Millions)", "(Dollars in millions, except per
# share amount)", "(In thousands)". Match the word, not a bracketed exact form.
SCALE_RE = re.compile(r"\b(millions|thousands|billions)\b", re.I)
PERIOD_RE = re.compile(
    r"(?:year|years|quarter|quarters|period|periods)\s+ended\s+"
    r"([A-Z][a-z]+)\s+(\d{1,2})?",
    re.I,
)
MONTHS = {
    m: i
    for i, m in enumerate(
        "january february march april may june july august september "
        "october november december".split(),
        1,
    )
}


# --------------------------------------------------------------------------
# 1. document metadata
# --------------------------------------------------------------------------

DOC_TYPE_RE = re.compile(r"(10K|10Q|8K|EARNINGS)", re.I)


@dataclass
class DocMeta:
    doc_name: str
    company: str
    doc_type: str
    fiscal_year: int | None
    period_end: str | None = None
    pdf_pages: int | None = None
    source_path: str = ""


def doc_meta(stem: str, doc) -> DocMeta:
    """Parse FinanceBench's naming convention: COMPANY_YEAR[Qn]_TYPE[_extra]."""
    name = stem.split(".")[0]
    parts = name.split("_")
    year = next((int(m.group()) for p in parts if (m := re.fullmatch(r"(19|20)\d{2}", p))), None)
    if year is None:
        m = re.search(r"(19|20)\d{2}", name)
        year = int(m.group()) if m else None
    tm = DOC_TYPE_RE.search(name)
    doc_type = tm.group(1).lower() if tm else "unknown"
    company_parts = []
    for p in parts:
        if re.match(r"^(19|20)\d{2}", p) or DOC_TYPE_RE.fullmatch(p):
            break
        company_parts.append(p)
    return DocMeta(
        doc_name=name,
        company=" ".join(company_parts) or name,
        doc_type=doc_type,
        fiscal_year=year,
        pdf_pages=len(doc.pages) or None,
    )


def period_end(doc, fiscal_year: int | None) -> str | None:
    """Find the statements' own 'Years ended <Month> <day>' line.

    The period identity lives here, not in a table's column header: a column
    headed 2016 is calendar year-end for 3M and May 2016 for Nike.
    """
    if fiscal_year is None:
        return None
    for tx in doc.texts:
        m = PERIOD_RE.search(tx.text or "")
        if not m:
            continue
        month = MONTHS.get(m.group(1).lower())
        if not month:
            continue
        day = int(m.group(2)) if m.group(2) else None
        if day is None:
            # No day given; fall back to the month's last day.
            import calendar

            day = calendar.monthrange(fiscal_year, month)[1]
        return f"{fiscal_year:04d}-{month:02d}-{day:02d}"
    return None


# --------------------------------------------------------------------------
# 2. Item sections — derived, not a Docling field
# --------------------------------------------------------------------------


def item_sections(doc) -> dict[int, str]:
    """page_no -> the Item section in force on that page.

    Docling has no notion of SEC Item structure, so scan text items in reading
    order for 'Item 7'-style headings and carry the last one seen forward.
    """
    current: str | None = None
    out: dict[int, str] = {}
    for tx in doc.texts:
        pg = tx.prov[0].page_no if tx.prov else None
        text = (tx.text or "").strip()
        m = ITEM_RE.match(text)
        # Require it to look like a heading, not a cross-reference buried in a
        # sentence ("as described in Item 7 below").
        if m and len(text) < 120:
            current = f"Item {m.group(1).upper()}"
        if pg is not None and current and pg not in out:
            out[pg] = current
    return out


# --------------------------------------------------------------------------
# 3. facts — one row per numeric table cell
# --------------------------------------------------------------------------

NUMERIC_RE = re.compile(r"^\(?\s*-?\$?\s*([\d,]+(?:\.\d+)?)\s*\)?%?$")


def scrub(text: str | None) -> str:
    """Remove characters PostgreSQL text columns cannot store.

    Some PDFs yield NUL bytes in their text layer, which psycopg rejects
    outright. Applied before embedding as well as before insert, so the vector
    and the stored string are computed from identical text.
    """
    if not text:
        return ""
    return text.replace("\x00", "")

# Row labels that are really header/units rows, not line items.
HEADER_LABEL_RE = re.compile(
    r"^\(?\s*(millions|thousands|billions|dollars|in\s|except\s|amounts\s)", re.I
)


@dataclass
class Fact:
    table_ordinal: int
    row_label: str
    column_label: str
    value_raw: str
    value: float
    parenthesized: bool
    scale: str | None
    unit: str | None
    page: int


def _cell_grid(tbl) -> dict[tuple[int, int], str]:
    return {
        (c.start_row_offset_idx, c.start_col_offset_idx): (c.text or "").strip()
        for c in tbl.data.table_cells
    }


def _header_map(tbl, grid: dict[tuple[int, int], str]) -> dict[int, str]:
    """col index -> column label, from the cells Docling flagged as headers."""
    rows = [c.start_row_offset_idx for c in tbl.data.table_cells if c.column_header]
    hdr_row = min(rows) if rows else 0
    labels: dict[int, str] = {}
    for col in range(tbl.data.num_cols):
        txt = grid.get((hdr_row, col), "")
        if txt:
            labels[col] = txt
    return labels


def table_facts(doc, idx: int) -> list[Fact]:
    tbl = doc.tables[idx]
    page = tbl.prov[0].page_no if tbl.prov else 0
    grid = _cell_grid(tbl)
    headers = _header_map(tbl, grid)

    # Table-level scale: usually the corner cell ("(Millions)", "(Dollars in
    # millions, except per share amount)"), sometimes the header row.
    corner = grid.get((0, 0), "")
    m = SCALE_RE.search(corner) or SCALE_RE.search(" ".join(headers.values()))
    table_scale = m.group(1).lower() if m else None

    # Fallback: text on the page immediately around the table. Restricted to
    # short items so a sentence like "millions of customers" in body prose
    # cannot supply a scale.
    if table_scale is None:
        for tx in doc.texts:
            if tx.prov and tx.prov[0].page_no == page and len(tx.text or "") < 120:
                pm = SCALE_RE.search(tx.text or "")
                if pm:
                    table_scale = pm.group(1).lower()
                    break

    facts: list[Fact] = []
    for (row, col), text in sorted(grid.items()):
        if col == 0 or row == 0 or not text:
            continue  # col 0 is the row label; row 0 is the header
        nm = NUMERIC_RE.match(text)
        if not nm:
            continue  # blanks, "$", footnote markers, prose cells
        label = grid.get((row, 0), "").strip()
        column_label = headers.get(col, "")
        if not label or not column_label:
            continue
        # A row whose label is a units marker is a second header row, not data.
        # Tables with multi-row headers otherwise leak "(Millions) · Capital
        # Expenditures · 2018" as a fact whose value is the year.
        if HEADER_LABEL_RE.match(label):
            continue
        # Row-level scale wins: segment tables put it in the row label
        # ("Sales (millions)") rather than once for the whole table.
        rm = SCALE_RE.search(label)
        scale = rm.group(1).lower() if rm else table_scale

        facts.append(
            Fact(
                table_ordinal=idx,
                row_label=label,
                column_label=column_label,
                value_raw=text,
                value=float(nm.group(1).replace(",", "")),
                # Accounting notation: (1,577) is an outflow. Recorded, not
                # resolved — the schema's generated `signed` column applies it.
                parenthesized=text.strip().startswith("("),
                scale=scale,
                unit="percent" if text.rstrip().endswith("%") else None,
                page=page,
            )
        )
    return facts


# --------------------------------------------------------------------------
# 4. chunks
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    kind: str
    text: str
    item_section: str | None
    heading_trail: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    table_ordinal: int | None = None


def prose_chunks(doc, sections: dict[int, str]) -> list[Chunk]:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    # ADR 0003: the token budget must come from the embedding model's own
    # tokenizer. HybridChunker needs Docling's wrapper — handed a raw HF
    # tokenizer it does not take max_tokens from it, and oversized chunks are
    # then silently truncated at embed time with only a warning.
    hf = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    tok = HuggingFaceTokenizer(tokenizer=hf, max_tokens=EMBED_MAX_TOKENS)
    chunker = HybridChunker(tokenizer=tok)

    out: list[Chunk] = []
    for ch in chunker.chunk(dl_doc=doc):
        pages = [
            p.page_no
            for it in ch.meta.doc_items
            for p in (it.prov or [])
        ]
        if not (ch.text or "").strip():
            continue
        lo = min(pages) if pages else None
        out.append(
            Chunk(
                kind="prose",
                text=ch.text,
                item_section=sections.get(lo) if lo else None,
                heading_trail=list(ch.meta.headings or []),
                page_start=lo,
                page_end=max(pages) if pages else None,
            )
        )
    return out


TRAIL_DEPTH = 3


def table_heading_trails(doc) -> dict[int, list[str]]:
    """table index -> the section headings printed above it, in reading order.

    A segment table's identity is not in its cells: page 33's table says
    "Sales (millions)" and nothing about "Industrial Business", which is a
    heading above it. Without this, five segments' figures are
    indistinguishable.

    Docling reports every section_header at level 1, so there is no hierarchy
    to walk. The last few headers in reading order are used instead: the
    nearest one alone is often useless ("Year 2018 results:").
    """
    ordinal = {id(t): i for i, t in enumerate(doc.tables)}
    recent: list[str] = []
    trails: dict[int, list[str]] = {}
    for item, _ in doc.iterate_items():
        label = str(getattr(item, "label", ""))
        text = (getattr(item, "text", "") or "").strip()
        # Docling's heading classification is inconsistent: "Consumer Business
        # (14.6% of consolidated sales):" comes back as `text` while the four
        # identically-formatted sibling segments come back as `section_header`.
        # Treat a short colon-terminated line as a heading too.
        looks_like_heading = (
            "section_header" in label
            or "title" in label
            or ("text" in label and text.endswith(":") and len(text) < 100)
        )
        if looks_like_heading and text:
            recent.append(text)
            del recent[:-TRAIL_DEPTH]
        elif "table" in label and id(item) in ordinal:
            trails[ordinal[id(item)]] = list(recent)
    return trails


def table_chunks(doc, sections: dict[int, str]) -> list[Chunk]:
    trails = table_heading_trails(doc)
    out: list[Chunk] = []
    for i, tbl in enumerate(doc.tables):
        page = tbl.prov[0].page_no if tbl.prov else None
        md = tbl.export_to_markdown(doc=doc)
        if not md.strip():
            continue
        out.append(
            Chunk(
                kind="table",
                text=md,
                item_section=sections.get(page) if page else None,
                heading_trail=trails.get(i, []),
                page_start=page,
                page_end=page,
                table_ordinal=i,
            )
        )
    return out


# --------------------------------------------------------------------------
# 5. embeddings
# --------------------------------------------------------------------------


def embed(texts: list[str], stem: str, batch: int = 8):
    """Embed, caching to disk keyed by the text content.

    Embedding is the expensive step (minutes on CPU). A failure anywhere after
    it — a bad insert, a dropped connection — must not cost it again.
    """
    import hashlib

    import numpy as np

    key = hashlib.md5(
        ("\x00".join(texts)).encode(), usedforsecurity=False
    ).hexdigest()[:16]
    cached = CACHE / f"{stem}.emb.{key}.npy"
    if cached.exists():
        vecs = np.load(cached)
        print(f"reusing cached embeddings {cached.name}  {vecs.shape}")
        return vecs

    vecs = _encode(texts, batch)
    np.save(cached, vecs)
    print(f"cached embeddings → {cached.name}")
    return vecs


def _encode(texts: list[str], batch: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    # SentenceTransformer carries its own sequence cap, independent of the
    # tokenizer's model_max_length. If it is lower than the chunker's budget,
    # long chunks are truncated here with no error at all.
    if model.max_seq_length < EMBED_MAX_TOKENS:
        print(
            f"raising max_seq_length {model.max_seq_length} -> {EMBED_MAX_TOKENS}"
        )
        model.max_seq_length = EMBED_MAX_TOKENS

    # Sort by length so long chunks batch with long chunks. Mixed batches pad
    # every short chunk up to the longest one, wasting time and memory.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs_sorted = model.encode(
        [DOC_PREFIX + texts[i] for i in order],
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    import numpy as np

    vecs = np.empty_like(vecs_sorted)
    vecs[order] = vecs_sorted
    if vecs.shape[1] != EMBED_DIM:
        sys.exit(f"expected {EMBED_DIM} dims, model gave {vecs.shape[1]}")
    return vecs


# --------------------------------------------------------------------------
# 6. write
# --------------------------------------------------------------------------


def fact_text(company: str, context: str | None, f: Fact) -> str:
    """One fact as a short retrievable string (ADR 0005).

    Everything distinguishing this fact from its neighbours goes in: company,
    which statement it came from, the line item, the period, the value. Inside
    a whole-table chunk the line item is a fortieth of the text; here it is
    most of it, which is what moved capex from rank 10 to rank 1.
    """
    parts = [company]
    if context:
        parts.append(context.rstrip(":"))
    parts += [
        f.row_label,
        f.column_label,
        f"{f.value_raw} {f.scale or ''}".strip(),
    ]
    return " · ".join(p for p in parts if p)


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
        fact_vecs = embed(fact_texts, f"{meta.doc_name}.facts") if facts else []

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

    vecs = embed([c.text for c in chunks], args.stem)
    write(meta, chunks, facts, vecs)


if __name__ == "__main__":
    main()
