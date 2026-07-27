#!/usr/bin/env bash
# Download a flywheel run's results directory from S3 into the local project.
#
# Usage:
#   ./bash/results_from_s3.sh <run-name>
#
# Example:
#   ./bash/results_from_s3.sh ablation-dagger-intervention-ratio-20260718-033707
#
# Results are placed at results/flywheel/<run-name>/ matching the S3 layout.
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

uv run python scripts/s3_backup.py download --components results "$1"
