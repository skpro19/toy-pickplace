#!/bin/bash
set -o pipefail
_R=__RUN_NAME__
_A=__ARCH__
_C=__CONTROL_DIR__

# wait for any artifact
for i in $(seq 1 60); do
  for d in checkpoints/flywheel/${_A}/${_R} runs/flywheel/${_A}/${_R} \
           data/flywheel/${_A}/${_R} results/flywheel/${_A}/${_R}; do
    if test -d "/workspace/toy-pickplace/$d" && \
       find "/workspace/toy-pickplace/$d" -type f 2>/dev/null | \
       head -1 | grep -q .; then break 2; fi
  done; sleep 5
done

while true; do
  cycle_id=$(date +%s%N)

  echo "$cycle_id" > "${_C}/state/backup-cycle-started.tmp"
  mv "${_C}/state/backup-cycle-started.tmp" "${_C}/state/backup-cycle-started"

  final_token=""
  if [ -f "${_C}/state/backup-final-requested" ]; then
    final_token=$(cat "${_C}/state/backup-final-requested" 2>/dev/null || true)
  fi

  retry_delay=30
  attempt=0
  cd /workspace/toy-pickplace
  while [ "$attempt" -lt 3 ]; do
    /root/.local/bin/uv run --env-file "${_C}/s3-env.env" \
      python scripts/s3_backup.py --arch "${_A}" upload \
      --components checkpoints,runs,results,dagger "${_R}"
    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
      echo "$cycle_id" > "${_C}/state/backup-last-succeeded.tmp"
      mv "${_C}/state/backup-last-succeeded.tmp" "${_C}/state/backup-last-succeeded"
      if [ -n "$final_token" ]; then
        echo "$final_token" > "${_C}/state/backup-final-succeeded.tmp"
        mv "${_C}/state/backup-final-succeeded.tmp" "${_C}/state/backup-final-succeeded"
      fi
      break
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -lt 3 ]; then
      echo "backup attempt ${attempt} failed (exit=${exit_code}), retrying in ${retry_delay}s"
      sleep "${retry_delay}"
      retry_delay=$((retry_delay * 2))
    fi
  done
  if [ "$attempt" -ge 3 ]; then
    printf '{"timestamp":"%s","exit_code":%d,"attempts":%d}\n' \
      "$(date -Iseconds)" "$exit_code" "3" \
      > "${_C}/state/backup-failed.tmp"
    mv "${_C}/state/backup-failed.tmp" "${_C}/state/backup-failed"
    exit 1
  fi
  attempt=0
  sleep 120
done
