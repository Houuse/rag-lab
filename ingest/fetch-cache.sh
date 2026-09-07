#!/usr/bin/env bash
# Pull the conversion and embedding cache from HuggingFace into ingest/cache/.
#
# The alternative is converting 366 PDFs again, which is the expensive half of
# the pipeline and takes days. The artifacts are 5.2 GB: 732 Docling JSONs,
# 778 .npy embedding files, 366 markdown dumps.
#
# Embeddings are keyed by an md5 of the chunk and fact strings, so this cache is
# only valid against the versions pinned in docling-extract's pyproject.toml. A
# different docling_core or tokenizer produces different chunk boundaries, a
# different key, and a silent cache miss.
set -euo pipefail

REPO="${RAGLAB_ARTIFACTS_REPO:-Houusee/rag-lab-artifacts}"
DEST="$(dirname "$0")/cache"
HF="$(dirname "$0")/.venv/bin/hf"

[ -x "$HF" ] || { echo "no hf CLI at $HF — see AGENTS.md, the venv is not reproducible" >&2; exit 1; }

echo "fetching $REPO into $DEST (5.2 GB)"
"$HF" download "$REPO" \
    --repo-type=dataset \
    --local-dir "$DEST"

echo "done. $(find "$DEST" -maxdepth 1 -type f | wc -l) files in $DEST"
