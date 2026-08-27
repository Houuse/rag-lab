#!/usr/bin/env bash
# Start (or restart) the rag-lab Postgres container.
#
#   ./run.sh            start, creating the volume and applying schema if new
#   ./run.sh --recreate destroy the container AND the volume, then start clean
#
# Data lives in the named volume raglab_pgdata, so `podman rm` on the container
# does not lose anything. Only --recreate does.

set -euo pipefail

NAME=raglab-pg
VOLUME=raglab_pgdata
IMAGE=docker.io/pgvector/pgvector:pg17
PORT=${PGPORT:-5433}
DB=raglab
USER=raglab
PASS=${PGPASSWORD:-raglab}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--recreate" ]]; then
  echo "removing container and volume"
  podman rm -f "$NAME" 2>/dev/null || true
  podman volume rm "$VOLUME" 2>/dev/null || true
fi

# Port 5433 by default: 5432 is often already taken by a host Postgres or
# another container, and the failure looks like a hung connection later.
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  if ! podman ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "port ${PORT} is in use by something that is not ${NAME}" >&2
    echo "set PGPORT to a free port and rerun" >&2
    exit 1
  fi
fi

if podman container exists "$NAME"; then
  podman start "$NAME" >/dev/null
  echo "started existing container ${NAME}"
else
  podman volume create "$VOLUME" >/dev/null 2>&1 || true
  podman run -d \
    --name "$NAME" \
    -e POSTGRES_DB="$DB" \
    -e POSTGRES_USER="$USER" \
    -e POSTGRES_PASSWORD="$PASS" \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v "${VOLUME}:/var/lib/postgresql/data" \
    -p "${PORT}:5432" \
    --health-cmd "pg_isready -U ${USER} -d ${DB}" \
    --health-interval 2s \
    --health-retries 30 \
    "$IMAGE" >/dev/null
  echo "created container ${NAME} on port ${PORT}"
fi

printf 'waiting for postgres'
for _ in $(seq 60); do
  if podman exec "$NAME" pg_isready -U "$USER" -d "$DB" >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  printf '.'
  sleep 1
done

if ! podman exec "$NAME" pg_isready -U "$USER" -d "$DB" >/dev/null 2>&1; then
  echo " timed out" >&2
  echo "logs:" >&2
  podman logs --tail 30 "$NAME" >&2
  exit 1
fi

echo "applying schema (idempotent)"
podman exec -i "$NAME" psql -v ON_ERROR_STOP=1 -U "$USER" -d "$DB" \
  < "${HERE}/schema.sql"

cat <<EOF

ready.

  DSN   postgresql://${USER}:${PASS}@localhost:${PORT}/${DB}
  psql  podman exec -it ${NAME} psql -U ${USER} -d ${DB}
  stop  podman stop ${NAME}          (data survives)
  wipe  ./run.sh --recreate          (data does not)
EOF
