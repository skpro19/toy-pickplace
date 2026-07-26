#!/bin/bash
set -o pipefail
_R=__RUN_NAME__
_C=__CONTROL_DIR__

# wait for any artifact
for i in $(seq 1 60); do
  for d in checkpoints/flywheel/${_R} runs/flywheel/${_R} \
           data/flywheel/${_R} results/flywheel/${_R}; do
    if test -d "/workspace/toy-pickplace/$d" && \
       find "/workspace/toy-pickplace/$d" -type f 2>/dev/null | \
       head -1 | grep -q .; then break 2; fi
  done; sleep 5
done

while true; do
  touch "${_C}/state/backup-cycle-started"
  cd /workspace/toy-pickplace
  /root/.local/bin/uv run --env-file "${_C}/s3-env.env" \
    python scripts/s3_backup.py upload \
    --prefix "${_R}" --components checkpoints,runs,results,dagger "${_R}"
  if [ "$?" -eq 0 ]; then
    touch "${_C}/state/backup-last-succeeded"
  else
    echo "failed" > "${_C}/state/backup-failed.tmp"
    mv "${_C}/state/backup-failed.tmp" "${_C}/state/backup-failed"
    exit 1
  fi
  sleep 120
done
