"""Convert and load many filings. Resumable, one document at a time.

    python batch.py                  # the 84 documents the questions reference
    python batch.py --all            # all 368 PDFs
    python batch.py --limit 5        # first 5 of the worklist
    python batch.py --dry-run        # print the plan, do nothing
    python batch.py --convert-only   # skip the load step
    python batch.py --corpus 150     # referenced + same-company distractors
    python batch.py --workers 4 --threads 2

Each document runs in its own subprocess. That costs ~15s of model loading per
step, and buys isolation: a PDF that crashes Docling or gets OOM-killed takes
down one document, not the run. Both steps are individually resumable —
convert skips cached page batches, load skips documents already in the
database — so re-running after any failure picks up where it stopped.

Sequential by default. The work is CPU-bound neural inference, not I/O: one
document uses about 5 of 8 cores, so parallelism helps only until the cores are
full. Measured: 6 workers at 8 threads each was *slower* than sequential —
48 threads on 8 cores thrashes. Keep workers x threads near the core count.
Memory is not the binding constraint; 6 workers peaked well inside 30GB.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PDF_DIR = HERE.parent / "financebench" / "pdfs"
FB_DATA = HERE.parent / "financebench" / "data" / "financebench_open_source.jsonl"
LOG = HERE / "batch.log"

DSN = os.environ.get("RAGLAB_DSN", "postgresql://raglab:raglab@localhost:5433/raglab")
PY = str(HERE / ".venv" / "bin" / "python")
MODE = "accurate"


def question_docs() -> dict[str, set[str]]:
    """financebench_id -> every document that question needs."""
    need: dict[str, set[str]] = {}
    for line in FB_DATA.open(encoding="utf-8"):
        rec = json.loads(line)
        docs = {rec["doc_name"]} | {
            ev["doc_name"] for ev in rec.get("evidence", []) if ev.get("doc_name")
        }
        need[rec["financebench_id"]] = docs
    return need


def referenced_docs() -> list[str]:
    return sorted({d for s in question_docs().values() for d in s})


def by_coverage() -> list[str]:
    """Documents ordered so each one unlocks as many questions as possible.

    Question coverage is sharply front-loaded: 20 of 84 documents fully cover
    74 of the 150 questions. Ingesting alphabetically wastes that.
    """
    need = question_docs()
    remaining = dict(need)
    order: list[str] = []
    pool = {d for s in need.values() for d in s}
    while pool:
        gain = {
            d: sum(1 for s in remaining.values() if d in s and len(s - set(order)) == 1)
            for d in pool
        }
        # Prefer documents that complete a question outright; fall back to
        # whichever appears in the most still-unanswered questions.
        best = max(
            pool,
            key=lambda d: (
                gain[d],
                sum(1 for s in remaining.values() if d in s),
                d,
            ),
        )
        order.append(best)
        pool.discard(best)
        remaining = {
            q: s for q, s in remaining.items() if not s <= set(order)
        }
    return order


def company_of(doc_name: str) -> str:
    """Leading company token(s) of a FinanceBench filename."""
    parts = []
    for p in doc_name.split("_"):
        if re.match(r"^(19|20)\d{2}", p) or re.fullmatch(
            r"(10K|10Q|8K|EARNINGS)", p, re.I
        ):
            break
        parts.append(p)
    return "_".join(parts) or doc_name


def with_distractors(total: int) -> list[str]:
    """The referenced documents, then the most confusable fillers.

    A distractor only earns its place if retrieval could mistake it for the
    answer. Another filing by the *same company* — 3M's 2016 10-K beside its
    2018 one — differs mainly in its numbers, which is exactly the
    discrimination being tested. An unrelated company tests nothing, so those
    come last.
    """
    referenced = by_coverage()
    have = set(referenced)
    companies = {company_of(d) for d in referenced}

    others = [p.stem for p in PDF_DIR.glob("*.pdf") if p.stem not in have]
    same_company = sorted(d for d in others if company_of(d) in companies)
    rest = sorted(d for d in others if company_of(d) not in companies)

    return (referenced + same_company + rest)[:total]


def already_loaded() -> set[str]:
    try:
        import psycopg

        with psycopg.connect(DSN) as conn:
            return {
                r[0]
                for r in conn.execute(
                    """
                    SELECT d.doc_name FROM documents d
                    WHERE EXISTS (SELECT 1 FROM facts f WHERE f.doc_id = d.doc_id
                                  AND f.embedding IS NOT NULL)
                    """
                ).fetchall()
            }
    except Exception as exc:  # noqa: BLE001
        print(f"warning: cannot read database ({exc}); nothing treated as loaded")
        return set()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(step: str, args: list[str], name: str) -> tuple[bool, str]:
    """Run one subprocess. Returns (ok, last line of output)."""
    proc = subprocess.run(  # noqa: S603
        [PY, *args],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        # A subprocess killed by the OOM killer exits with a signal and no
        # traceback, which is worth naming explicitly.
        if proc.returncode < 0:
            detail = f"killed by signal {-proc.returncode} (likely OOM)"
        log(f"  FAIL {step} {name}: {detail}")
        return False, detail
    out = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return True, out[-1] if out else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every PDF, not just referenced")
    ap.add_argument(
        "--corpus",
        type=int,
        metavar="N",
        help="the referenced documents plus same-company distractors, N total",
    )
    ap.add_argument(
        "--alpha",
        action="store_true",
        help="alphabetical order instead of by question coverage",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="documents in parallel; each wants several GB of RAM",
    )
    ap.add_argument("--batch", type=int, default=12, help="pages per convert batch")
    ap.add_argument(
        "--threads",
        type=int,
        default=8,
        help="torch threads per convert process; keep workers x threads near "
        "the core count, since the work is CPU-bound",
    )
    args = ap.parse_args()

    if args.all:
        wanted = sorted(p.stem for p in PDF_DIR.glob("*.pdf"))
    elif args.corpus:
        wanted = with_distractors(args.corpus)
    elif args.alpha:
        wanted = referenced_docs()
    else:
        wanted = by_coverage()

    missing = [n for n in wanted if not (PDF_DIR / f"{n}.pdf").exists()]
    wanted = [n for n in wanted if (PDF_DIR / f"{n}.pdf").exists()]

    loaded = already_loaded()
    todo = [n for n in wanted if n not in loaded]
    if args.limit:
        todo = todo[: args.limit]

    print(f"worklist        {len(wanted)} documents")
    print(f"already loaded  {len(loaded & set(wanted))}")
    print(f"to process      {len(todo)}")
    if missing:
        print(f"referenced but no PDF present ({len(missing)}): {missing[:5]}")
    cached = sum(1 for n in todo if (CACHE / f"{n}.{MODE}.json").exists())
    print(f"already converted (load only)   {cached}")
    print(f"estimated       ~{(len(todo) - cached) * 5 + len(todo) * 1.5:.0f} min")

    if args.dry_run or not todo:
        for n in todo[:20]:
            print(f"  {n}")
        if len(todo) > 20:
            print(f"  ... {len(todo) - 20} more")
        return

    log(f"=== batch start: {len(todo)} documents, {args.workers} worker(s)")
    t0 = time.perf_counter()
    failures: list[tuple[str, str, str]] = []
    done = 0
    lock = threading.Lock()

    def process(name: str) -> None:
        nonlocal done
        stem = f"{name}.{MODE}"

        if not (CACHE / f"{stem}.json").exists():
            ok, detail = run("convert", ["convert.py", name, "--mode", MODE,
                                         "--batch", str(args.batch),
                                         "--threads", str(args.threads)], name)
            if not ok:
                with lock:
                    failures.append((name, "convert", detail))
                return

        if not args.convert_only:
            ok, detail = run("load", ["load.py", stem], name)
            if not ok:
                with lock:
                    failures.append((name, "load", detail))
                return
        else:
            detail = "converted"

        with lock:
            done += 1
            elapsed = time.perf_counter() - t0
            eta = (elapsed / done * (len(todo) - done)) / 60
            log(f"[{done}/{len(todo)}] {name}  {detail}  (~{eta:.0f} min left)")

    if args.workers > 1:
        # Neither CPU nor GPU saturates on a single document — the bottleneck is
        # single-threaded PDF parsing — so parallel documents scale well. The
        # ceiling is memory: convert peaks at 2-5GB and load holds an embedding
        # model, so each worker wants several GB.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(process, todo))
    else:
        for name in todo:
            process(name)

    mins = (time.perf_counter() - t0) / 60
    log(f"=== done: {done}/{len(todo)} in {mins:.0f} min, {len(failures)} failed")
    for name, step, detail in failures:
        log(f"    {step:8} {name}: {detail}")
    if failures:
        print("\nRe-run the same command to retry the failures; "
              "successful documents are skipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
