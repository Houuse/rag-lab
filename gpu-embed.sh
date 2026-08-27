#!/usr/bin/env bash
# Run the embedding half of ingestion on a machine with a GPU.
#
#   ./gpu-embed.sh                          embed every document in --cache
#   ./gpu-embed.sh --cache ../docling-extract/out
#   ./gpu-embed.sh --setup-only             set up and verify, embed nothing
#   ./gpu-embed.sh --batch 128              bigger batches if VRAM allows
#   ./gpu-embed.sh --cpu                    force CPU (mostly for testing)
#
# Conversion is the half a GPU turns out not to help with — measured at 2%
# utilisation in docs/system/batch-runs.md. Embedding is the half it should:
# a few hundred thousand batched transformer forward passes. This script does
# only that, needs no database, and leaves `.npy` files behind for `load.py`
# to find.
#
# Structure and the CUDA-torch ordering follow docling-extract/setup.sh, which
# already worked these out on this hardware.

set -euo pipefail

CACHE=""              # default resolved below, after we know where we are
PYVER="3.12"          # torch CUDA wheels lag new Python releases; 3.14 has none
DEVICE="auto"         # auto | cuda | cpu
BATCH=64
SETUP_ONLY=0
FORCED_GPU=0

while [ $# -gt 0 ]; do
  case "$1" in
    --cache)       CACHE="$2"; shift 2 ;;
    --python)      PYVER="$2"; shift 2 ;;
    --batch)       BATCH="$2"; shift 2 ;;
    --cpu)         DEVICE="cpu"; shift ;;
    --gpu)         DEVICE="cuda"; FORCED_GPU=1; shift ;;
    --setup-only)  SETUP_ONLY=1; shift ;;
    -h|--help)     sed -n '2,17p' "$0"; exit 0 ;;
    *)             echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/ingest"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# The documents most likely came from docling-extract on this same machine, so
# default to its output directory when it is there and ours is empty.
if [ -z "$CACHE" ]; then
  if [ -n "$(find cache -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]; then
    CACHE="cache"
  elif [ -d ../../docling-extract/out ]; then
    CACHE="../../docling-extract/out"
  elif [ -d ../docling-extract/out ]; then
    CACHE="../docling-extract/out"
  else
    die "no documents found. Pass --cache <dir> holding <stem>.accurate.json files."
  fi
fi

[ -d "$CACHE" ] || die "no such directory: $CACHE"
COUNT="$(find "$CACHE" -maxdepth 1 -name '*.json' ! -name '*.meta.json' | wc -l | tr -d ' ')"
[ "$COUNT" -gt 0 ] || die "no document JSON in $CACHE"

# ---------------------------------------------------------------- 1. uv
say "1/5  uv"
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --quiet uv 2>/dev/null \
    || python3 -m pip install --quiet uv \
    || die "could not install uv. Install Python 3.12+ and pip, then rerun."
fi
uv --version

# ------------------------------------------------------- 2. venv, pinned
say "2/5  virtualenv on Python $PYVER"

find_py() {
  if   [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe
  elif [ -x .venv/Scripts/python ];     then echo .venv/Scripts/python
  elif [ -x .venv/bin/python ];         then echo .venv/bin/python
  fi
}

PY="$(find_py)"
if [ -n "$PY" ] && [ "$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = "$PYVER" ]; then
  echo "reusing existing .venv"
else
  if [ -d .venv ]; then
    echo "existing .venv is not on Python $PYVER — replacing it"
    uv venv --python "$PYVER" --clear --quiet
  else
    uv venv --python "$PYVER" --quiet
  fi
  PY="$(find_py)"
fi

[ -n "$PY" ] || die "no interpreter in .venv — check the uv output above"
"$PY" --version
ACTUAL="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$ACTUAL" = "$PYVER" ] || die "venv is on Python $ACTUAL, wanted $PYVER"

# ------------------------------------------------------ 3. the embed deps
say "3/5  embedding dependencies"
uv pip install --python "$PY" --quiet -r requirements-embed.txt
"$PY" - <<'EOF'
from importlib.metadata import version
for p in ("docling-core", "transformers", "sentence-transformers"):
    print(f"  {p} {version(p)}")
EOF

# --------------------------------------------------------------- 4. torch
# After the pinned installs, never before: resolving them can replace a CUDA
# wheel with the CPU-only one, and that failure only shows up at runtime as
# "Torch not compiled with CUDA enabled".
if [ "$DEVICE" = "auto" ]; then
  say "4/5  detecting GPU"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "no nvidia-smi -> using CPU"
    DEVICE="cpu"
  elif ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi present but failing (driver problem?) -> using CPU"
    DEVICE="cpu"
  else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    DEVICE="cuda"
  fi
fi

if [ "$DEVICE" = "cuda" ]; then
  say "4/5  CUDA torch"
  CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  [ -n "$CUDA_VER" ] || die "could not read a CUDA version from nvidia-smi"
  echo "driver supports CUDA $CUDA_VER"
  echo -n "torch as installed: "
  "$PY" -c 'import torch; print(torch.__version__, "| cuda:", torch.cuda.is_available())'

  if "$PY" -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "already CUDA-capable, leaving it alone"
  else
    MAJOR="${CUDA_VER%%.*}"; MINOR="${CUDA_VER#*.}"
    if   [ "$MAJOR" -ge 13 ]; then IDX=cu128
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 8 ]; then IDX=cu128
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 4 ]; then IDX=cu124
    elif [ "$MAJOR" -eq 12 ]; then IDX=cu121
    else IDX=cu118
    fi
    echo "CUDA unavailable; reinstalling torch from index $IDX"
    uv pip install --python "$PY" --force-reinstall --quiet \
       --index-url "https://download.pytorch.org/whl/$IDX" torch \
       || die "no $IDX torch wheel for Python $PYVER. Try --python 3.11, or --cpu."
  fi

  say "5/5  verify GPU"
  if "$PY" - <<'EOF'
import sys, torch
v, ok = torch.__version__, torch.cuda.is_available()
print("torch:", v)
print("cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    print("torch cannot see the GPU.", file=sys.stderr)
    if v.endswith("+cpu"):
        print("A CPU-only wheel is installed: this Python version has no CUDA"
              " build. Try --python 3.11.", file=sys.stderr)
sys.exit(0 if ok else 1)
EOF
  then
    :
  elif [ "$FORCED_GPU" = "1" ]; then
    die "GPU was requested with --gpu but is not usable (see above)"
  else
    printf '\n!! GPU unusable — falling back to CPU. That is the thing this script\n'
    printf '!! exists to avoid; on CPU you may as well run the normal batch.\n'
    printf '!! Ctrl-C now if you would rather fix the GPU first.\n\n'
    DEVICE="cpu"
  fi
else
  say "4/5  torch (CPU)"
  "$PY" -c 'import torch; print("torch:", torch.__version__)'
  say "5/5  verify — skipped, running on CPU"
fi

# ---------------------------------------------------------------- 6. run
if [ "$SETUP_ONLY" = "1" ]; then
  say "setup complete"
  echo "embed with:  $PY embed_only.py --all --cache \"$CACHE\" --device $DEVICE --batch $BATCH"
  exit 0
fi

say "embedding $COUNT documents from $CACHE"
# Resumable at document granularity: `load.embed` skips any document whose
# .npy already exists, so rerunning after an interruption is free.
"$PY" embed_only.py --all --cache "$CACHE" --device "$DEVICE" --batch "$BATCH"

NPY="$(find "$CACHE" -maxdepth 1 -name '*.emb.*.npy' | wc -l | tr -d ' ')"
say "done: $NPY .npy files in $CACHE"
echo
echo "package just the embeddings (they are small next to the JSON):"
echo "  tar -czf embeddings.tar.gz -C \"$CACHE\" \$(cd \"$CACHE\" && ls *.emb.*.npy)"
echo
echo "then on the database machine, unpack into ingest/cache/ and run:"
echo "  .venv/bin/python batch.py --all --workers 4 --threads 2"
