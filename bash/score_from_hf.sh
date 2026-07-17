#!/usr/bin/env bash
# Download all checkpoints and results from a HuggingFace backup session, then
# re-evaluate each flywheel run's best.pt checkpoint using final_score.py.
#
# Usage:
#   ./bash/score_from_hf.sh <HF_SESSION_PREFIX>
#
# Example:
#   ./bash/score_from_hf.sh ablation-dagger-intervention-ratio-20260718-033707
#
# This will:
#   1. Download checkpoints/ and results/ for the given session from HF Hub.
#   2. For every run folder under results/flywheel/<prefix>/, run
#      final_score.py --run-name <prefix>/<run>.
#
# Requirements: uv, huggingface_hub, and HF_TOKEN (for private repos).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <HF_SESSION_PREFIX>"
    exit 1
fi

PREFIX="$1"

echo "=== Downloading checkpoints and results ==="
uv run python scripts/hf_backup.py download --components checkpoints,results "$PREFIX"

for run_dir in "results/flywheel/$PREFIX"/*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    full_run_name="$PREFIX/$run_name"
    echo ""
    echo "=== Evaluating $full_run_name ==="
    uv run python scripts/final_score.py --run-name "$full_run_name"
done

echo ""
echo "=== Done ==="
