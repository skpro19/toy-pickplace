#!/usr/bin/env bash
# Download a flywheel run's checkpoints from S3 into the local project.
#
# Usage:
#   ./bash/ckpt_from_s3.sh <run-name>
#
# Example:
#   ./bash/ckpt_from_s3.sh 2026-07-31_00-08-26_1785436706135705407_BASEv2_recency-0p8_seed5693
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
