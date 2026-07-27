#!/usr/bin/env bash
# Download a flywheel run's dagger data from S3 into the local project.
#
# Usage:
#   ./bash/dagger_data_from_s3.sh <run-name>
#
# Example:
#   ./bash/dagger_data_from_s3.sh vision_mlp-flywheel-num_expert_episodes=200_20260727-045818
#
# Dagger data is placed at data/flywheel/<run-name>/ matching the S3 layout.
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

uv run python scripts/s3_backup.py download --components dagger "$1"
