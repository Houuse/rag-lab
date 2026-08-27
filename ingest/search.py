"""Vector search over chunks.

    python search.py "why did operating cash flow fall in 2018"
    python search.py "capital expenditures" -k 10 --kind table
    python search.py "supply chain risk" --company 3M --item "Item 1A"

The query prefix is "search_query: " while indexing used "search_document: "
(ADR 0003). Getting that wrong does not error — it silently degrades every
result — so the prefix lives here, not in a caller's hands.
"""

import argparse
import os
import textwrap

DSN = os.environ.get("RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab")

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
QUERY_PREFIX = "search_query: "

_model = None


def embed_query(text: str) -> list[float]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        _model.max_seq_length = 8192
    vec = _model.encode(
        [QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
    )[0]
    return vec.tolist()


def search(
    query: str,
    k: int = 5,
    kind: str | None = None,
    company: str | None = None,
    item: str | None = None,
):
    import psycopg

    vec = "[" + ",".join(f"{x:.6f}" for x in embed_query(query)) + "]"

    where = ["c.embedding IS NOT NULL"]
    params: list[object] = [vec]
    if kind:
        where.append("c.kind = %s")
        params.append(kind)
    if company:
        where.append("d.company ILIKE %s")
        params.append(f"%{company}%")
    if item:
        where.append("c.item_section = %s")
        params.append(item)
    params.append(k)

    sql = f"""
        SELECT c.chunk_id,
               1 - (c.embedding <=> %s::vector) AS score,
               c.kind, c.page_start, c.item_section,
               c.heading_trail[array_length(c.heading_trail, 1)] AS context,
               d.doc_name, c.text
        FROM chunks c JOIN documents d USING (doc_id)
        WHERE {" AND ".join(where)}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    # The vector appears twice: once for the reported score, once for the
    # ordering that the HNSW index can actually use.
    params.insert(-1, vec)

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_facts(query: str, k: int = 5, company: str | None = None):
    """Nearest-neighbour over facts, each its own retrievable unit (ADR 0005)."""
    import psycopg

    vec = "[" + ",".join(f"{x:.6f}" for x in embed_query(query)) + "]"
    where = ["f.embedding IS NOT NULL"]
    params: list[object] = [vec, vec]
    if company:
        where.append("d.company ILIKE %s")
        params.append(f"%{company}%")
    params.append(k)

    sql = f"""
        SELECT 1 - (f.embedding <=> %s::vector) AS score,
               f.fact_text, f.signed, f.scale, f.page, d.doc_name
        FROM facts f JOIN documents d USING (doc_id)
        WHERE {" AND ".join(where)}
        ORDER BY f.embedding <=> %s::vector
        LIMIT %s
    """
    params[0], params[1] = vec, vec
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--kind", choices=["prose", "table"])
    ap.add_argument("--company")
    ap.add_argument("--item", help="e.g. 'Item 7'")
    ap.add_argument("--full", action="store_true", help="print whole chunk text")
    ap.add_argument("--facts", action="store_true", help="search facts, not chunks")
    args = ap.parse_args()

    if args.facts:
        for score, ftext, signed, scale, page, doc in search_facts(
            args.query, args.k, args.company
        ):
            print(f"{score:.3f}  p{page:<4} {signed:>14,}  {scale or '':<9} {ftext}")
        return

    rows = search(args.query, args.k, args.kind, args.company, args.item)
    if not rows:
        print("no results")
        return

    for cid, score, kind, page, item, context, doc, text in rows:
        head = f"{score:.3f}  [{kind}] {doc} p{page}  chunk {cid}"
        if item:
            head += f"  {item}"
        print(head)
        if context:
            print(f"        ↳ {context[:90]}")
        body = text if args.full else text[:400]
        for line in textwrap.wrap(body.replace("\n", " "), 96)[: 200 if args.full else 4]:
            print(f"        {line}")
        print()


if __name__ == "__main__":
    main()
