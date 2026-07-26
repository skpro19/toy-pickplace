#!/usr/bin/env bash
# Re-evaluate each flywheel run's best.pt checkpoint from an S3 backup session.
# Completed evaluation artifacts are uploaded back to the same S3 session.
#
# Usage:
#   ./bash/score_from_s3.sh <S3_SESSION_PREFIX> [--workers <N>]
#
# Example:
#   ./bash/score_from_s3.sh ablation-dagger-intervention-ratio-20260718-033707
#   ./bash/score_from_s3.sh ablation-dagger-intervention-ratio-20260718-033707 --workers 6
#
# This will:
#   1. Reuse locally complete evaluation artifacts, if available; otherwise,
#      download checkpoints/ and results/ for the given session from S3.
#   2. Skip runs whose complete artifacts are already in S3.
#   3. Evaluate the remaining runs and upload their artifacts to S3.
#
# Requirements: uv, AWS credentials, S3_BUCKET, and EGL-capable MuJoCo rendering.
# Defaults to MUJOCO_GL=egl; set MUJOCO_GL to override it.
set -euo pipefail

# Auto-load project configuration so S3 settings are available without manual exports.
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

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

LOCAL_RESULTS_ROOT="results/flywheel/$PREFIX"
ARTIFACTS=(
    final_scores.json
    final_scores_comparison.png
)

local_artifacts_complete() {
    local run_dir="$1"
    local artifact
    for artifact in "${ARTIFACTS[@]}"; do
        [ -f "$run_dir/$artifact" ] || return 1
    done
}

local_session_complete() {
    local run_dir
    local found_run=false

    [ -d "$LOCAL_RESULTS_ROOT" ] || return 1
    for run_dir in "$LOCAL_RESULTS_ROOT"/*/; do
        [ -d "$run_dir" ] || continue
        found_run=true
        local_artifacts_complete "$run_dir" || return 1
    done
    "$found_run"
}

remote_artifacts_complete() {
    local full_run_name="$1"
    uv run python scripts/s3_backup.py has-files \
        --components results \
        "$full_run_name" \
        "${ARTIFACTS[@]}" \
        >/dev/null 2>&1
}

# A completed local session contains every artifact needed for upload. Keeping it
# intact avoids overwriting newer local scores with the version stored in S3.
if local_session_complete; then
    echo "=== Reusing locally completed evaluation artifacts ==="
else
    echo "=== Downloading checkpoints and results ==="
    uv run python scripts/s3_backup.py download --components checkpoints,results "$PREFIX"
fi

for run_dir in "$LOCAL_RESULTS_ROOT"/*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    full_run_name="$PREFIX/$run_name"

    # Both files mark a successful, fully persisted final evaluation.
    if remote_artifacts_complete "$full_run_name"; then
        echo "=== Skipping $full_run_name: artifacts already uploaded ==="
        continue
    fi

    if local_artifacts_complete "$run_dir"; then
        echo "=== Uploading existing artifacts for $full_run_name ==="
    else
        echo ""
        echo "=== Evaluating $full_run_name ==="
        uv run python scripts/final_score.py --run-name "$full_run_name" "${WORKERS_ARGS[@]}"
    fi

    echo ""
    echo "=== Uploading artifacts for $full_run_name ==="
    aws s3 sync "results/flywheel/$PREFIX/$run_name/" "s3://$S3_BUCKET/$S3_PREFIX/$PREFIX/results/$run_name/"
done

echo ""
echo "=== Done ==="
