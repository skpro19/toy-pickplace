#!/usr/bin/env bash
# Download all checkpoints and results from an S3 backup session, then
# re-evaluate each flywheel run's best.pt checkpoint using final_score.py.
#
# Usage:
#   ./bash/score_from_s3.sh <S3_SESSION_PREFIX> [--workers <N>]
#
# Example:
#   ./bash/score_from_s3.sh ablation-dagger-intervention-ratio-20260718-033707
#   ./bash/score_from_s3.sh ablation-dagger-intervention-ratio-20260718-033707 --workers 6
#
# This will:
#   1. Download checkpoints/ and results/ for the given session from S3.
#   2. For every run folder under results/flywheel/<prefix>/, run
#      final_score.py --run-name <prefix>/<run>.
#
# Requirements: uv, AWS credentials, S3_BUCKET, and EGL-capable MuJoCo rendering.
# Defaults to MUJOCO_GL=egl; set MUJOCO_GL to override it.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <S3_SESSION_PREFIX> [--workers <N>]"
    exit 1
fi

PREFIX="$1"
shift

export MUJOCO_GL="${MUJOCO_GL:-egl}"

WORKERS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --workers)
            WORKERS_ARGS=(--workers "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "=== MuJoCo renderer: $MUJOCO_GL ==="
echo "=== Downloading checkpoints and results ==="
uv run python scripts/s3_backup.py download --components checkpoints,results "$PREFIX"

for run_dir in "results/flywheel/$PREFIX"/*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    full_run_name="$PREFIX/$run_name"
    echo ""
    echo "=== Evaluating $full_run_name ==="
    uv run python scripts/final_score.py --run-name "$full_run_name" "${WORKERS_ARGS[@]}"
done

echo ""
echo "=== Done ==="
