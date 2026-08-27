"""Convert one PDF with Docling in ACCURATE mode and cache the result.

Converts in page batches, so that:
  * progress is real (pages actually finished), not a clock-based guess
  * peak memory is bounded by the batch, not the document
  * a crash costs one batch — rerun and it resumes from the parts on disk

Usage:
    python convert.py 3M_2018_10K
    python convert.py 3M_2018_10K --batch 8
    python convert.py 3M_2018_10K --no-resume       # ignore parts, start over
    python convert.py 3M_2018_10K --mode fast --ocr

Conversion is the slow step, so it is isolated here. Everything you want to
look at afterwards goes in probe.py, which reads the cache and never
re-parses a PDF.
"""

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "financebench" / "pdfs"
CACHE = Path(__file__).resolve().parent / "cache"


def resolve_pdf(name: str) -> Path:
    p = Path(name)
    if p.suffix.lower() == ".pdf" and p.exists():
        return p
    candidate = PDF_DIR / f"{name}.pdf"
    if candidate.exists():
        return candidate
    sys.exit(f"No such PDF: {name!r} (looked in {PDF_DIR})")


def page_count(pdf: Path) -> int:
    import pypdfium2

    doc = pypdfium2.PdfDocument(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def build_converter(mode: str, threads: int, ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # SEC filings are born-digital: the text layer is already in the file, and
    # measurement showed OCR to be a no-op here. Off by default; --ocr for a
    # document that turns out to be scanned.
    opts = PdfPipelineOptions(do_table_structure=True, do_ocr=ocr)
    opts.table_structure_options.mode = (
        TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
    )
    # Match predicted table structure back to the PDF's real text cells, so the
    # digits we report come from the file rather than the model's reading of it.
    opts.table_structure_options.do_cell_matching = True

    # Best-effort thread pinning; the option has moved between versions, so a
    # failure here is not worth aborting a long run over.
    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )

        opts.accelerator_options = AcceleratorOptions(
            num_threads=threads, device=AcceleratorDevice.CPU
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (accelerator options unavailable: {exc})", file=sys.stderr)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


class Heartbeat:
    """Proof of life during a blocking convert() call.

    Reports elapsed time only. It deliberately does not estimate a percentage:
    within a batch there is no way to know how far along we are, and a made-up
    number invites trust it has not earned. Real progress is reported between
    batches, in pages actually finished.
    """

    def __init__(self, every: float = 15.0) -> None:
        self._every = every
        self._stop = threading.Event()

    def __enter__(self) -> "Heartbeat":
        started = time.perf_counter()

        def run() -> None:
            while not self._stop.wait(self._every):
                mins, secs = divmod(int(time.perf_counter() - started), 60)
                print(f"      ... still working  {mins}m{secs:02d}s", flush=True)

        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()


def batches(total: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size - 1, total)) for s in range(1, total + 1, size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="doc name (e.g. 3M_2018_10K) or a path to a .pdf")
    ap.add_argument("--mode", choices=["accurate", "fast"], default="accurate")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=12, help="pages per batch")
    ap.add_argument("--ocr", action="store_true", help="enable OCR (scanned PDFs only)")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing parts")
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()

    from docling_core.types.doc import ImageRefMode
    from docling_core.types.doc.document import DoclingDocument

    pdf = resolve_pdf(args.pdf)
    stem = f"{pdf.stem}.{args.mode}" + (".ocr" if args.ocr else "")
    args.out.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out / "parts" / stem
    parts_dir.mkdir(parents=True, exist_ok=True)

    total = page_count(pdf)
    plan = batches(total, args.batch)

    print(f"{pdf.name}  {total} pages  mode={args.mode}  ocr={args.ocr}")
    print(f"{len(plan)} batches of {args.batch} pages, {args.threads} threads")
    print(f"parts → {parts_dir}\n")

    converter = build_converter(args.mode, args.threads, args.ocr)

    done_pages = 0
    t_start = time.perf_counter()

    for i, (lo, hi) in enumerate(plan, 1):
        part = parts_dir / f"p{lo:04d}-{hi:04d}.json"
        span = hi - lo + 1
        head = f"[{i}/{len(plan)}] pages {lo}-{hi}"

        if part.exists() and not args.no_resume:
            done_pages += span
            print(f"{head}  cached, skipping")
            continue

        t0 = time.perf_counter()
        with Heartbeat():
            result = converter.convert(str(pdf), page_range=(lo, hi))
        took = time.perf_counter() - t0

        result.document.save_as_json(part, image_mode=ImageRefMode.PLACEHOLDER)

        done_pages += span
        pct = 100 * done_pages / total
        rate = took / span
        remaining = (total - done_pages) * (time.perf_counter() - t_start) / done_pages
        print(
            f"{head}  {took:5.1f}s  ({rate:.1f}s/page)   "
            f"{done_pages}/{total} pages done, {pct:.0f}%   "
            f"~{remaining / 60:.0f} min left"
        )

        del result
        gc.collect()

    print("\nmerging parts…")
    docs = [
        DoclingDocument.load_from_json(p) for p in sorted(parts_dir.glob("p*.json"))
    ]
    merged = DoclingDocument.concatenate(docs)

    json_path = args.out / f"{stem}.json"
    md_path = args.out / f"{stem}.md"
    merged.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER)
    md_path.write_text(merged.export_to_markdown(), encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    pages_seen = sorted(merged.pages)
    meta = {
        "pdf": str(pdf),
        "mode": args.mode,
        "ocr": args.ocr,
        "pdf_pages": total,
        "batch_size": args.batch,
        "batches": len(plan),
        "seconds": round(elapsed, 1),
        "seconds_per_page": round(elapsed / total, 2),
        "merged_pages": len(merged.pages),
        "page_range_seen": [pages_seen[0], pages_seen[-1]] if pages_seen else None,
        "tables": len(merged.tables),
        "texts": len(merged.texts),
    }
    (args.out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))

    print()
    for k, v in meta.items():
        print(f"  {k:18} {v}")

    # Page numbering across a batched convert is the one thing that could
    # silently corrupt every downstream page reference. Say so loudly.
    if pages_seen and (pages_seen[0], pages_seen[-1]) != (1, total):
        print(
            f"\n  !! merged pages run {pages_seen[0]}..{pages_seen[-1]}, "
            f"expected 1..{total} — page numbers need remapping before use"
        )

    print(f"\ncached → {json_path}")
    print(f"next:  .venv/bin/python probe.py {stem} --evidence financebench_id_03029")


if __name__ == "__main__":
    main()
