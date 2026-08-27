"""Inspect a cached DoclingDocument. Fast, re-runnable, never touches a PDF.

Usage:
    python probe.py 3M_2018_10K.accurate                 # structural overview
    python probe.py 3M_2018_10K.accurate --find "Cash Flow"
    python probe.py 3M_2018_10K.accurate --table 12      # dump one table
    python probe.py 3M_2018_10K.accurate --evidence financebench_id_03029

The four questions this is built to answer:
  1. did a financial statement fragment across a page break?
  2. did the scale marker ("(Millions)") survive, and is it inside the table?
  3. are year columns aligned to the right values?
  4. does the extracted page agree with FinanceBench's human transcription?
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
FB_DATA = HERE.parent / "financebench" / "data" / "financebench_open_source.jsonl"

SCALE_RE = re.compile(r"\((?:in\s+)?(millions|thousands|billions)\)", re.I)


def load(stem: str):
    from docling_core.types.doc.document import DoclingDocument

    path = CACHE / f"{stem}.json" if not stem.endswith(".json") else Path(stem)
    if not path.exists():
        sys.exit(f"No cached document at {path}. Run convert.py first.")
    return DoclingDocument.load_from_json(path)


def page_of(item) -> int | None:
    return item.prov[0].page_no if getattr(item, "prov", None) else None


def vertical_extent(item, doc):
    """Return (top, bottom) as fractions of page height, 0.0 = top of page.

    Docling bboxes carry their own coordinate origin, so normalise rather than
    assuming one. Returns (None, None) when there is no usable provenance.
    """
    if not getattr(item, "prov", None):
        return None, None
    prov = item.prov[0]
    page = doc.pages.get(prov.page_no)
    if page is None or not page.size.height:
        return None, None
    h = page.size.height
    bbox = prov.bbox
    if str(getattr(bbox, "coord_origin", "")).upper().endswith("BOTTOMLEFT"):
        return (h - bbox.t) / h, (h - bbox.b) / h
    return bbox.t / h, bbox.b / h


def overview(doc) -> None:
    print(f"pages   {len(doc.pages)}")
    print(f"tables  {len(doc.tables)}")
    print(f"texts   {len(doc.texts)}\n")

    by_page: dict[int, list[int]] = {}
    for i, tbl in enumerate(doc.tables):
        by_page.setdefault(page_of(tbl), []).append(i)

    print("page-break split candidates")
    print("  Docling assembles tables strictly per page, so a statement running")
    print("  across a break arrives as fragments. A continuation fragment has no")
    print("  header row of its own — that is a far better signal than geometry,")
    print("  which needs a per-document threshold to mean anything.\n")

    def header_cells(tbl) -> int:
        return sum(1 for c in tbl.data.table_cells if c.column_header)

    headless = [i for i, t in enumerate(doc.tables) if header_cells(t) == 0]
    print(f"  tables with no header row at all: {len(headless)} of {len(doc.tables)}")
    if headless:
        print(f"    {headless[:20]}")

    found = 0
    for j in headless:
        frag = doc.tables[j]
        pg = page_of(frag)
        for i, tbl in enumerate(doc.tables):
            if (
                page_of(tbl) == (pg or 0) - 1
                and tbl.data.num_cols == frag.data.num_cols
                and header_cells(tbl) > 0
            ):
                print(
                    f"  table {i:>3} (p{page_of(tbl)}, "
                    f"{tbl.data.num_rows}x{tbl.data.num_cols})  ->  "
                    f"table {j:>3} (p{pg}, {frag.data.num_rows}x{frag.data.num_cols})"
                )
                found += 1
    if not found:
        print("  no continuation fragments found")

    # Show the geometry range so a null result above is checkable rather than
    # merely asserted: if nothing comes near the page bottom, no threshold on
    # vertical position could have fired.
    bottoms = [b for b in (vertical_extent(t, doc)[1] for t in doc.tables) if b]
    if bottoms:
        print(
            f"  (lowest any table reaches on its page: {max(bottoms):.0%} "
            f"of page height)"
        )

    print("\ntables carrying a scale marker inside their own cells")
    inside = [i for i, t in enumerate(doc.tables) if SCALE_RE.search(cells_text(t))]
    print(f"  {len(inside)} of {len(doc.tables)}: {inside[:20]}")

    orphan = [
        i
        for i, t in enumerate(doc.tables)
        if not SCALE_RE.search(cells_text(t))
        and any(
            SCALE_RE.search(tx.text or "")
            for tx in doc.texts
            if page_of(tx) == page_of(t)
        )
    ]
    print(f"  scale marker on the page but OUTSIDE the table: {orphan[:20]}")
    print("  ^ these need a separate scale-hunting pass at ingestion")


def cells_text(tbl) -> str:
    return " ".join(c.text or "" for c in tbl.data.table_cells)


def find(doc, needle: str) -> None:
    low = needle.lower()
    print(f"text items matching {needle!r}:")
    for tx in doc.texts:
        if low in (tx.text or "").lower():
            print(f"  p{page_of(tx):>4}  [{tx.label}]  {(tx.text or '')[:110]}")
    print(f"\ntables whose cells mention {needle!r}:")
    for i, tbl in enumerate(doc.tables):
        if low in cells_text(tbl).lower():
            top, bottom = vertical_extent(tbl, doc)
            span = f"{top:.2f}-{bottom:.2f}" if top is not None else "?"
            print(
                f"  table {i:>3}  p{page_of(tbl):>4}  "
                f"{tbl.data.num_rows}x{tbl.data.num_cols}  vspan {span}"
            )


def dump_table(doc, idx: int) -> None:
    tbl = doc.tables[idx]
    top, bottom = vertical_extent(tbl, doc)
    print(f"table {idx}  page {page_of(tbl)}  {tbl.data.num_rows}x{tbl.data.num_cols}")
    if top is not None:
        print(f"vertical span on page: {top:.2f} -> {bottom:.2f}  (0.0 = page top)")
    print()
    try:
        df = tbl.export_to_dataframe(doc=doc)
        with_opts = df.to_string(max_colwidth=28)
        print(with_opts)
    except Exception as exc:  # noqa: BLE001
        print(f"(dataframe export failed: {exc}) — falling back to raw cells")
        for c in tbl.data.table_cells:
            print(f"  r{c.start_row_offset_idx} c{c.start_col_offset_idx}: {c.text!r}")

    csv_path = CACHE / f"table_{idx}.csv"
    try:
        tbl.export_to_dataframe(doc=doc).to_csv(csv_path, index=False)
        print(f"\nwrote {csv_path}")
    except Exception:  # noqa: BLE001
        pass


NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def as_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def numbers(text: str) -> set[str]:
    """Distinctive numeric tokens. Short ones ('1', '31') match everywhere."""
    return {n for n in NUM_RE.findall(text) if len(n.replace(",", "")) >= 3}


def page_blobs(doc) -> dict[int, str]:
    """One normalised text blob per page: all text items plus all table cells.

    Docling fragments a page into many items, so any matching done against
    individual items fails on phrases that straddle an item boundary. Match
    against the page instead.
    """
    blobs: dict[int, list[str]] = {}
    for tx in doc.texts:
        pg = page_of(tx)
        if pg is not None:
            blobs.setdefault(pg, []).append(tx.text or "")
    for tbl in doc.tables:
        pg = page_of(tbl)
        if pg is not None:
            blobs.setdefault(pg, []).append(cells_text(tbl))
    return {pg: re.sub(r"\s+", " ", " ".join(parts)) for pg, parts in blobs.items()}


def evidence(doc, fb_id: str) -> None:
    """Compare extraction against FinanceBench's human transcription.

    The evidence page is located by content overlap rather than by the stated
    page number, because SEC filings offset printed page labels from physical
    page order — and because a wrong page would make the recovery score below
    meaningless in a way that looks like an extraction failure.
    """
    rec = None
    for line in FB_DATA.open(encoding="utf-8"):
        r = json.loads(line)
        if r["financebench_id"] == fb_id:
            rec = r
            break
    if rec is None:
        sys.exit(f"{fb_id} not found in {FB_DATA}")

    print(f"question: {rec['question']}")
    print(f"answer:   {rec['answer']}")

    blobs = page_blobs(doc)

    for ev in rec["evidence"]:
        label = ev["evidence_page_num"]
        full = ev["evidence_text_full_page"]
        want = numbers(full)
        print(f"\n--- evidence claims page {label} of {ev['doc_name']}")
        print(f"    transcription has {len(want)} distinctive numbers")

        if not want:
            print("    (no numbers to match on — skipping)")
            continue

        scored = sorted(
            (
                (len(want & numbers(blob)) / len(want), pg)
                for pg, blob in blobs.items()
            ),
            reverse=True,
        )
        print("    best matching pages by number overlap:")
        for frac, pg in scored[:3]:
            print(f"      p{pg:<4} {frac:6.1%}")

        best_frac, best_pg = scored[0]
        if best_frac < 0.30:
            print("    >>> no page matches well — extraction or alignment is wrong")
            continue

        if str(best_pg) != str(label):
            print(f"    >>> OFFSET: printed label {label} -> physical page {best_pg}")
        else:
            print(f"    printed label and physical page agree ({best_pg})")

        missing = want - numbers(blobs[best_pg])
        print(
            f"    recovered {len(want) - len(missing)}/{len(want)} "
            f"of the transcription's numbers on p{best_pg}"
        )
        if missing:
            print(f"      missing: {sorted(missing)[:15]}")

        # Does the figure the question actually asks for survive? Compare as
        # numbers, not strings: the answer says "$1577.00", the filing prints
        # "1,577".
        for tok in numbers(rec["answer"]):
            target = as_float(tok)
            if target is None:
                continue
            where = [
                pg
                for pg, b in blobs.items()
                if any(as_float(n) == target for n in numbers(b))
            ]
            mark = "found" if best_pg in where else "NOT on the evidence page"
            print(f"    answer value {target:g}: {mark}  (pages {where[:6]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="cache stem, e.g. 3M_2018_10K.accurate")
    ap.add_argument("--find", help="locate text/tables containing this string")
    ap.add_argument("--table", type=int, help="dump one table by index")
    ap.add_argument("--evidence", help="compare against a financebench_id")
    args = ap.parse_args()

    doc = load(args.stem)

    if args.find:
        find(doc, args.find)
    elif args.table is not None:
        dump_table(doc, args.table)
    elif args.evidence:
        evidence(doc, args.evidence)
    else:
        overview(doc)


if __name__ == "__main__":
    main()
