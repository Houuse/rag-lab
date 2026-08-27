"""Experiment: does making each fact its own retrievable unit beat whole-table
chunks?

Baseline to beat: searching whole-table chunks for "3M capital expenditures
2018" put the correct page (60) at rank 10, with the top 10 spanning only
0.650-0.620 — no real discrimination.

Nothing is written to the database. If this works, it justifies a schema
change; if not, we learned it cheaply.
"""

import os

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer

DSN = os.environ.get("RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab")
MODEL = "nomic-ai/nomic-embed-text-v1.5"

QUERIES = [
    ("3M capital expenditures 2018", 1577.0),
    ("3M capital expenditures 2016", 1420.0),
    ("3M net income 2018", 5363.0),
    ("3M net sales 2018", 32765.0),
    ("capex", 1577.0),
    ("money spent on property and equipment 2018", 1577.0),
]


def fact_text(company, context, row_label, column_label, value_raw, scale) -> str:
    """One fact as a short retrievable string.

    Everything that distinguishes this fact from its neighbours is in here:
    the company, which statement it came from, the line item, the period, and
    the value. In a whole-table chunk the line item is 1/40th of the text; here
    it is most of it.
    """
    parts = [company]
    if context:
        parts.append(context.rstrip(":"))
    parts += [row_label, column_label, f"{value_raw} {scale or ''}".strip()]
    return " · ".join(p for p in parts if p)


def main() -> None:
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            """
            SELECT f.fact_id, d.company,
                   c.heading_trail[array_length(c.heading_trail,1)] AS context,
                   f.row_label, f.column_label, f.value_raw, f.scale,
                   f.value, f.page
            FROM facts f
            JOIN documents d USING (doc_id)
            LEFT JOIN chunks c ON c.chunk_id = f.chunk_id
            """
        ).fetchall()

    texts = [fact_text(*r[1:7]) for r in rows]
    print(f"{len(texts)} facts")
    print("example:", texts[0])
    print()

    model = SentenceTransformer(MODEL, trust_remote_code=True)
    F = model.encode(
        ["search_document: " + t for t in texts],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    for q, want in QUERIES:
        v = model.encode(
            ["search_query: " + q], normalize_embeddings=True, show_progress_bar=False
        )[0]
        order = np.argsort(-(F @ v))
        rank = next(
            (
                i
                for i, j in enumerate(order, 1)
                if abs(float(rows[j][7]) - want) < 0.01
            ),
            None,
        )
        print(f"\n{q!r}   target {want:,.0f}  ->  rank {rank}")
        for i in order[:3]:
            print(f"   {F[i] @ v:.3f}  p{rows[i][8]:<4} {texts[i][:88]}")


if __name__ == "__main__":
    main()
