#!/usr/bin/env bash
# Download a flywheel run's TensorBoard files from S3 and launch TensorBoard.
#
# Usage:
#   ./bash/tb_from_s3.sh <run-name>
#
# Example:
#   ./bash/tb_from_s3.sh ablation-dagger-intervention-ratio-20260718-033707
#
# TensorBoard is started on the downloaded runs directory.
#
# Requirements: uv, AWS credentials, S3_BUCKET.
set -euo pipefail

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run-name>"
    exit 1
fi

RUN_NAME="$1"
RUNS_DIR="runs/flywheel/${RUN_NAME}"

uv run python scripts/s3_backup.py download --components runs "${RUN_NAME}"

find_free_port() {
    local port=6006
    if command -v ss &>/dev/null; then
        while ss -tlnp 2>/dev/null | grep -qP ":$port\b"; do
            port=$((port + 1))
        done
    elif command -v lsof &>/dev/null; then
        while lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null; do
            port=$((port + 1))
        done
    fi
    echo "$port"
}

PORT=$(find_free_port)

echo "Starting TensorBoard for ${RUNS_DIR} on port ${PORT} ..."
uv run tensorboard --logdir "${RUNS_DIR}" --port "${PORT}"
