"""Vector search over chunks.

    python search.py "why did operating cash flow fall in 2018"
    python search.py "capital expenditures" -k 10 --kind table
    python search.py "supply chain risk" --company 3M --item "Item 1A"

The query prefix is "search_query: " while indexing used "search_document: "
(ADR 0003). Getting that wrong does not error — it silently degrades every
result — so the prefix lives here, not in a caller's hands.
"""

import argparse
import functools
import os
import re
import textwrap

DSN = os.environ.get("RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab")

# Imported, not restated: querying with a different model than indexing used
# returns plausible nonsense rather than an error.
from docling_extract.embedding import EMBED_MAX_TOKENS, EMBED_MODEL, QUERY_PREFIX

_model = None


def _norm(s: str) -> str:
    """Uppercase alphanumerics only. 'Johnson & Johnson' and 'JOHNSON JOHNSON'
    are the same company; so are 'Foot Locker' and 'FOOTLOCKER'."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


@functools.lru_cache(maxsize=1)
def known_companies() -> list[tuple[str, str]]:
    """(normalised, stored) for every company in the corpus, longest first.

    Longest first so ACTIVISIONBLIZZARD is tried before AES: a short name that
    is a substring of a longer one would otherwise capture the wrong filings.
    """
    import psycopg

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT company FROM documents WHERE company IS NOT NULL")
        rows = [r[0] for r in cur.fetchall()]
    return sorted(((_norm(c), c) for c in rows), key=lambda t: -len(t[0]))


YEAR_RE = re.compile(r"\bFY\s?(\d{4})\b|\b(20\d{2})\b", re.I)


def resolve_scope(question: str) -> tuple[str | None, int | None]:
    """Read a company and a fiscal year out of the question itself.

    The embedding cannot do this. Measured on our own model, adding a year to a
    query moves the similarity by 0.014 — noise — so a corpus holding 3M's
    2015 through 2022 filings ranks them interchangeably. The period has to be
    a predicate, not something hoped for from the vector.

    Deliberately parses the question text and never FinanceBench's `company` or
    `doc_name` fields. Those are labels the benchmark supplies and a real user
    does not; using them at query time would be measuring a system that cannot
    exist. They are for scoring only.

    Returns (company, fiscal_year), either possibly None. None means no filter
    on that axis — never a guess. Narrowing to the wrong company is worse than
    not narrowing, because the answer then becomes unreachable at any k.
    """
    q = _norm(question)
    company = next((stored for norm, stored in known_companies() if norm and norm in q), None)

    # "FY2018" beats a bare "2018": a question naming both usually means FY for
    # the period it wants and mentions the other in passing.
    years = YEAR_RE.findall(question)
    fy = next((int(a) for a, b in years if a), None)
    if fy is None:
        fy = next((int(b) for a, b in years if b), None)
    return company, fy


def _exact_sql(inner: str, source: str, where: list[str], outer: str) -> str:
    """Vector search restricted by metadata, done exactly rather than by index.

    An HNSW scan finds its nearest candidates *first* and applies the WHERE
    afterwards, so a selective filter can return fewer than k rows — or none
    at all — with no error. Measured here: filtering facts to one company
    pulled 45 candidates out of the index, none of them matching, and returned
    zero rows against 15,113 qualifying facts.

    Whether that happens is the planner's choice. Filtering to a single
    document happened to produce an exact scan and a correct answer; filtering
    by company did not. Correct-by-planner-choice is not correct, and the
    eval's oracle condition rests on this, so materialise the filtered set and
    search it exhaustively. Exact search over one filing's few thousand rows is
    cheap; over the whole corpus it would not be, which is why the unfiltered
    path still goes through the index.

    `inner` selects the columns to keep plus the embedding aliased to `emb`;
    `outer` names them back out, including the score expression, so the caller
    controls column order and the embedding never reaches the result.
    """
    return f"""
        WITH candidates AS MATERIALIZED (
            SELECT {inner}
            FROM {source}
            WHERE {" AND ".join(where)}
        )
        SELECT {outer}
        FROM candidates
        ORDER BY emb <=> %s::vector
        LIMIT %s
    """


@functools.lru_cache(maxsize=1024)
def embed_query(text: str) -> list[float]:
    """Encode a query. Memoised: the eval harness asks the same question under
    both the oracle and full-corpus conditions, and encoding it twice is pure
    waste. Callers must not mutate the returned list."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        _model.max_seq_length = EMBED_MAX_TOKENS
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
    doc_name: str | None = None,
    fiscal_year: int | None = None,
):
    import psycopg

    vec = "[" + ",".join(f"{x:.6f}" for x in embed_query(query)) + "]"

    where = ["c.embedding IS NOT NULL"]
    params: list[object] = []
    if kind:
        where.append("c.kind = %s")
        params.append(kind)
    if company:
        where.append("d.company ILIKE %s")
        params.append(f"%{company}%")
    if item:
        where.append("c.item_section = %s")
        params.append(item)
    if doc_name:
        # Exact, unlike --company's ILIKE: the eval's oracle condition means
        # "this filing and no other", and 3M_2018_10K must not admit
        # 3M_2018_10K_something.
        where.append("d.doc_name = %s")
        params.append(doc_name)
    if fiscal_year:
        where.append("d.fiscal_year = %s")
        params.append(fiscal_year)

    source = "chunks c JOIN documents d USING (doc_id)"
    if params:  # a metadata filter is in play — see _exact_sql
        sql = _exact_sql(
            inner="""c.chunk_id, c.kind, c.page_start, c.item_section,
                     c.heading_trail[array_length(c.heading_trail, 1)] AS context,
                     d.doc_name, c.text, c.embedding AS emb""",
            source=source,
            where=where,
            outer="""chunk_id, 1 - (emb <=> %s::vector) AS score,
                     kind, page_start, item_section, context, doc_name, text""",
        )
        params += [vec, vec, k]
    else:
        sql = f"""
            SELECT c.chunk_id,
                   1 - (c.embedding <=> %s::vector) AS score,
                   c.kind, c.page_start, c.item_section,
                   c.heading_trail[array_length(c.heading_trail, 1)] AS context,
                   d.doc_name, c.text
            FROM {source}
            WHERE {" AND ".join(where)}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        # The vector appears twice: once for the reported score, once for the
        # ordering that the HNSW index can actually use.
        params = [vec, vec, k]

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_facts(
    query: str,
    k: int = 5,
    company: str | None = None,
    doc_name: str | None = None,
    fiscal_year: int | None = None,
):
    """Nearest-neighbour over facts, each its own retrievable unit (ADR 0005)."""
    import psycopg

    vec = "[" + ",".join(f"{x:.6f}" for x in embed_query(query)) + "]"
    where = ["f.embedding IS NOT NULL"]
    params: list[object] = []
    if company:
        where.append("d.company ILIKE %s")
        params.append(f"%{company}%")
    if doc_name:
        # Exact, unlike --company's ILIKE: the eval's oracle condition means
        # "this filing and no other", and 3M_2018_10K must not admit
        # 3M_2018_10K_something.
        where.append("d.doc_name = %s")
        params.append(doc_name)
    if fiscal_year:
        where.append("d.fiscal_year = %s")
        params.append(fiscal_year)

    source = "facts f JOIN documents d USING (doc_id)"
    if params:  # a metadata filter is in play — see _exact_sql
        sql = _exact_sql(
            inner="f.fact_id, f.fact_text, f.signed, f.scale, f.page, d.doc_name, f.embedding AS emb",
            source=source,
            where=where,
            outer="""1 - (emb <=> %s::vector) AS score,
                     fact_text, signed, scale, page, doc_name, fact_id""",
        )
        params += [vec, vec, k]
    else:
        sql = f"""
            SELECT 1 - (f.embedding <=> %s::vector) AS score,
                   f.fact_text, f.signed, f.scale, f.page, d.doc_name, f.fact_id
            FROM {source}
            WHERE {" AND ".join(where)}
            ORDER BY f.embedding <=> %s::vector
            LIMIT %s
        """
        params = [vec, vec, k]

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
    ap.add_argument("--doc", help="restrict to one filing, e.g. 3M_2018_10K")
    ap.add_argument("--full", action="store_true", help="print whole chunk text")
    ap.add_argument("--facts", action="store_true", help="search facts, not chunks")
    args = ap.parse_args()

    if args.facts:
        for score, ftext, signed, scale, page, doc, fid in search_facts(
            args.query, args.k, args.company, args.doc
        ):
            print(f"{score:.3f}  p{page:<4} {signed:>14,}  {scale or '':<9} {ftext}")
        return

    rows = search(args.query, args.k, args.kind, args.company, args.item, args.doc)
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
