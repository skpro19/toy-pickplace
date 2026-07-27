#!/usr/bin/env bash
# Download a flywheel run's checkpoints from S3 into the local project.
#
# Usage:
#   ./bash/checkpoints_from_s3.sh <run-name>
#
# Example:
#   ./bash/checkpoints_from_s3.sh vision_mlp-flywheel-num_expert_episodes=200_20260727-045818
#
# Checkpoints are placed at checkpoints/flywheel/<run-name>/ matching the S3 layout.
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

uv run python scripts/s3_backup.py download --components checkpoints "$1"
