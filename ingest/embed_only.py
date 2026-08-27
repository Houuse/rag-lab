"""Produce the embedding caches for cached documents, without a database.

The expensive half of `load.py` is embedding, and it is the half a GPU is
actually good at — unlike conversion, which `docs/system/batch-runs.md`
measured at 2% GPU utilisation. This script runs that half on whatever machine
has the hardware and leaves `.npy` files behind; `load.py` on the database
machine then finds them cached and only writes rows.

    python embed_only.py --all --cache ../../docling-extract/out

Copy the resulting `*.emb.*.npy` back into `ingest/cache/` and run the normal
batch. Nothing else changes.

Correctness rests on one thing: the cache key is an md5 of the exact strings
being embedded, so this must build them with `load.py`'s own functions rather
than a second copy of the logic. It imports `load` and calls `load.embed`,
overriding only *how* the encoding happens, never *what* is encoded. If the
two ever disagree the hash changes, the cache silently misses, and the whole
run is wasted with no error — so there is deliberately no reimplementation
here to drift.
"""

import argparse
import sys
import time
from pathlib import Path

import load

HERE = Path(__file__).resolve().parent


def build_texts(stem: str):
    """The chunk and fact strings for one document, exactly as load.py makes them.

    Mirrors the order of operations in `load.main` and `load.write`: scrub
    before embedding so the vector and the stored text come from the same
    characters, and build fact strings from the *scrubbed* heading trail,
    because that is when `load.write` builds them.
    """
    from docling_core.types.doc.document import DoclingDocument

    path = load.CACHE / f"{stem}.json"
    if not path.exists():
        return None, f"no cached document at {path}"

    doc = DoclingDocument.load_from_json(path)
    meta = load.doc_meta(stem, doc)
    sections = load.item_sections(doc)

    chunks = load.prose_chunks(doc, sections) + load.table_chunks(doc, sections)
    facts = [f for i in range(len(doc.tables)) for f in load.table_facts(doc, i)]

    for c in chunks:
        c.text = load.scrub(c.text)
        c.item_section = load.scrub(c.item_section) or None
        c.heading_trail = [load.scrub(h) for h in c.heading_trail]
    for f in facts:
        f.row_label = load.scrub(f.row_label)
        f.column_label = load.scrub(f.column_label)
        f.value_raw = load.scrub(f.value_raw)
    chunks = [c for c in chunks if c.text.strip()]

    trail_of = {
        c.table_ordinal: (c.heading_trail[-1] if c.heading_trail else None)
        for c in chunks
        if c.kind == "table"
    }
    fact_texts = [
        load.fact_text(meta.company, trail_of.get(f.table_ordinal), f) for f in facts
    ]
    return (meta, [c.text for c in chunks], fact_texts), None


def install_encoder(device: str | None, batch: int) -> str:
    """Replace load._encode with one that loads the model once and picks a device.

    load.py constructs a SentenceTransformer on every call, which is fine for
    one document and wasteful for several hundred. The vectors are unaffected:
    same model, same prefix, same normalisation, same sequence cap.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit(
            "--device cuda but torch reports no CUDA. On Python 3.14 there is no "
            "CUDA wheel and the resolver installs a CPU-only build; see "
            "docs/system/batch-runs.md."
        )

    model = SentenceTransformer(load.EMBED_MODEL, trust_remote_code=True, device=device)
    # The same silent truncation load.py guards against: SentenceTransformer's
    # own cap is independent of the tokenizer's and is lower than the chunker's
    # budget, so long chunks would be cut with no error at all.
    if model.max_seq_length < load.EMBED_MAX_TOKENS:
        print(f"raising max_seq_length {model.max_seq_length} -> {load.EMBED_MAX_TOKENS}")
        model.max_seq_length = load.EMBED_MAX_TOKENS

    def _encode(texts: list[str], _batch: int):
        import numpy as np

        # Sort by length so long chunks batch with long chunks; mixed batches
        # pad every short chunk up to the longest one.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs_sorted = model.encode(
            [load.DOC_PREFIX + texts[i] for i in order],
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs = np.empty_like(vecs_sorted)
        vecs[order] = vecs_sorted
        if vecs.shape[1] != load.EMBED_DIM:
            sys.exit(f"expected {load.EMBED_DIM} dims, model gave {vecs.shape[1]}")
        return vecs

    load._encode = _encode
    return device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", help="cache stems, e.g. 3M_2018_10K.accurate")
    ap.add_argument("--all", action="store_true", help="every *.json in --cache")
    ap.add_argument(
        "--cache",
        type=Path,
        default=HERE / "cache",
        help="directory holding <stem>.json; also where .npy files are written",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=64,
        help="encode batch size; load.py uses 8 for CPU, a GPU wants far more",
    )
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()

    cache = args.cache.resolve()
    if not cache.is_dir():
        sys.exit(f"no such cache directory: {cache}")
    load.CACHE = cache

    if args.all:
        stems = sorted(
            p.name[: -len(".json")]
            for p in cache.glob("*.json")
            if not p.name.endswith(".meta.json")
        )
    else:
        stems = args.stems
    if not stems:
        sys.exit("nothing to do: pass stems or --all")

    device = install_encoder(args.device, args.batch)
    print(f"device {device}  batch {args.batch}  cache {cache}")
    print(f"{len(stems)} document(s)\n")

    t0 = time.perf_counter()
    done = 0
    failures: list[tuple[str, str]] = []

    for i, stem in enumerate(stems, 1):
        try:
            built, err = build_texts(stem)
            if err:
                failures.append((stem, err))
                print(f"[{i}/{len(stems)}] SKIP {stem}: {err}")
                continue
            meta, chunk_texts, fact_texts = built

            if chunk_texts:
                load.embed(chunk_texts, stem, args.batch)
            if fact_texts:
                load.embed(fact_texts, f"{meta.doc_name}.facts", args.batch)

            done += 1
            eta = (time.perf_counter() - t0) / done * (len(stems) - done) / 60
            print(
                f"[{i}/{len(stems)}] {stem}  "
                f"{len(chunk_texts)} chunks  {len(fact_texts)} facts  "
                f"(~{eta:.0f} min left)"
            )
        except Exception as exc:  # one bad document must not end the run
            failures.append((stem, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(stems)}] FAIL {stem}: {type(exc).__name__}: {exc}")

    mins = (time.perf_counter() - t0) / 60
    print(f"\n=== done: {done}/{len(stems)} in {mins:.0f} min, {len(failures)} failed")
    for stem, detail in failures:
        print(f"    {stem}: {detail}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
