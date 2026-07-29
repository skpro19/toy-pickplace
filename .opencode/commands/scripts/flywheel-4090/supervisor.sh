#!/bin/bash
# Durable local supervisor for one flywheel run.
# Sources ./.env for VAST_API_KEY and AWS credentials (tracing disabled).
# Arguments: INSTANCE_ID HOST PORT RUN_NAME CONTROL_DIR LOCAL_SSH_SESSION LOCAL_TB_SESSION ARCH

set -o pipefail

INSTANCE_ID="$1"; HOST="$2"; PORT="$3"
RUN_NAME="$4"; CONTROL_DIR="$5"
LOCAL_SSH_SESSION="$6"; LOCAL_TB_SESSION="$7"
ARCH="$8"
SUPERVISOR_SESSION="flywheel-supervisor-$6"
SUPERVISOR_SESSION="${SUPERVISOR_SESSION#vast-ssh-}"
SUPERVISOR_SESSION="flywheel-supervisor-${SUPERVISOR_SESSION}"

STATUS_FILE="/tmp/toy-pickplace-flywheel-${RUN_NAME}.status"

write_status() {
  local outcome="$1"
  local now state_val
  now=$(date -Iseconds)
  state_val="${STATE:-starting}"
  cat > "$STATUS_FILE" << EOF
{"outcome":"${outcome}","timestamp":"${now}","run_name":"${RUN_NAME}","instance_id":"${INSTANCE_ID}","last_state":"${state_val}"}
EOF
}

write_status "supervisor-starting"

set -a; . ./.env; set +a

SSH_CMD="ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$HOST"

cleanup() {
  local outcome="$1"
  echo "=== Cleanup: $outcome ==="
  capture_local_diagnostics || true
  echo "Instance destruction in 60s — SSH: ssh -p $PORT root@$HOST"
  $SSH_CMD "printf '%s\n' 'pending-destruction at $(date -Iseconds)' > '${CONTROL_DIR}/state/pending-destruction'" 2>/dev/null || true
  sleep 60
  test -n "$INSTANCE_ID" || {
    echo "ERROR: INSTANCE_ID is empty — cannot destroy"
    write_status "cleanup-aborted-empty-id"
    return 1
  }
  vastai destroy instance -y "$INSTANCE_ID" 2>&1 || echo "destroy may have already completed"
  for i in $(seq 1 30); do
    instances=$(vastai show instances --raw 2>/dev/null || echo "[]")
    count=""
    count=$(echo "$instances" | INSTANCE_ID="$INSTANCE_ID" uv run python -c '
import json, os, sys
data = json.load(sys.stdin)
if isinstance(data, dict):
    data = data.get("instances", [data])
target = os.environ.get("INSTANCE_ID", "")
if not target:
    raise SystemExit(2)
ids = [item.get("id") for item in data if str(item.get("id")) == target]
print(len(ids))
' 2>/dev/null) || { echo "ERROR: verification poll failed for $INSTANCE_ID"; count=; }
    if [ "$count" = "0" ]; then
      echo "Instance $INSTANCE_ID destroyed and removed"
      break
    fi
    if [ -z "$count" ]; then
      echo "Skipping one verification poll due to transient error; will retry"
    fi
    sleep 10
  done
  if [ "$count" != "0" ]; then
    echo "WARNING: Instance $INSTANCE_ID still present after 30 polls, retrying destroy"
    vastai destroy instance -y "$INSTANCE_ID" 2>&1 || true
  fi
  tmux kill-session -t "$LOCAL_SSH_SESSION" 2>/dev/null || true
  tmux kill-session -t "$LOCAL_TB_SESSION" 2>/dev/null || true
  write_status "$outcome"
  echo "Cleanup complete: $outcome"
}

upload_diagnostics() {
  local deadline=$(( $(date +%s) + 300 ))

  . "${CONTROL_DIR}/s3-env.env" 2>/dev/null || {
    mapfile -t AWS < <(uv run python -c "
import os, boto3
s = boto3.Session(profile_name=os.environ.get('AWS_PROFILE', ''))
c = s.get_credentials().get_frozen_credentials()
print(c.access_key); print(c.secret_key); print(c.token or ''); print(s.region_name or '')
" 2>/dev/null) || return 1
    AWS_ACCESS_KEY_ID="${AWS[0]}"; AWS_SECRET_ACCESS_KEY="${AWS[1]}"
    AWS_SESSION_TOKEN="${AWS[2]}"; AWS_REGION="${AWS[3]}"
  }

  prefix=".flywheel/${RUN_NAME}"

  for dir in logs state; do
    local_dir="${CONTROL_DIR}/${dir}"
    if [ -d "$local_dir" ]; then
      find "$local_dir" -type f | while read f; do
        [ $(date +%s) -gt $deadline ] && break
        rel="${f#${CONTROL_DIR}/${dir}/}"
        key="${prefix}/${dir}/${rel}"
        uv run python -c "
import boto3, os
c = boto3.client('s3', region_name='${AWS_REGION:-ap-south-1}')
c.upload_file('$f', os.environ['S3_BUCKET'], '$key')
" 2>/dev/null || true
      done
    fi
  done
}

capture_local_diagnostics() {
  local local_dir="/tmp/toy-pickplace-flywheel-${RUN_NAME}"
  mkdir -p "$local_dir"
  {
    echo "=== Capture $(date -Iseconds) ==="
    echo "=== Run: ${RUN_NAME} ==="
    echo "=== Instance: ${INSTANCE_ID} ==="
    echo "=== Host: ${HOST}:${PORT} ==="
    echo "=== State: ${STATE} ==="
    echo ""
  } > "$local_dir/meta.txt"

  $SSH_CMD "cat '${CONTROL_DIR}/state/'* 2>/dev/null; echo '---'; ls -la '${CONTROL_DIR}/state/'" \
    > "$local_dir/state.txt" 2>/dev/null || true

  for session in flywheel-run ckpt-bkp tensorboard; do
    $SSH_CMD "tmux capture-pane -t '$session' -p -S -300 2>/dev/null" \
      > "$local_dir/${session}.log" 2>/dev/null || true
  done

  $SSH_CMD "tail -500 '${CONTROL_DIR}/logs/${RUN_NAME}.log' 2>/dev/null" \
    > "$local_dir/training-tail.log" 2>/dev/null || true

  echo "Local diagnostics saved to $local_dir"
}

echo "Supervisor started: instance=$INSTANCE_ID host=$HOST port=$PORT run=$RUN_NAME"

$SSH_CMD "echo connected" >/dev/null 2>&1 || {
  echo "ERROR: Cannot connect to instance on startup"
  cleanup "supervisor-startup-failed"; exit 1
}

STATE="running"
HELDOUT_LAUNCHED=false
SSH_FAILURES=0

while true; do
  case "$STATE" in
    running)
      completed=$($SSH_CMD "test -f '${CONTROL_DIR}/state/completed' && cat '${CONTROL_DIR}/state/completed' || echo not_found" 2>/dev/null)
      failed=$($SSH_CMD "test -f '${CONTROL_DIR}/state/failed' && cat '${CONTROL_DIR}/state/failed' || echo not_found" 2>/dev/null)
      backup_failed=$($SSH_CMD "test -f '${CONTROL_DIR}/state/backup-failed' && echo yes || echo no" 2>/dev/null)
      ckpt_alive=$($SSH_CMD "tmux has-session -t ckpt-bkp 2>/dev/null && echo yes || echo no" 2>/dev/null)

      if [ -n "$failed" ] && [ "$failed" != "not_found" ]; then
        echo "Training failed: $failed"
        upload_diagnostics
        cleanup "training-failed-${failed#failed }"
        break
      fi
      if [ -n "$completed" ] && [ "$completed" != "not_found" ]; then
        echo "Training completed: $completed"
        REQUEST_TOKEN=$(date +%s%N)
        $SSH_CMD "echo '$REQUEST_TOKEN' > '${CONTROL_DIR}/state/backup-final-requested.tmp' && mv '${CONTROL_DIR}/state/backup-final-requested.tmp' '${CONTROL_DIR}/state/backup-final-requested'" 2>/dev/null || true
        echo "Backup final token requested: $REQUEST_TOKEN"
        STATE="completed-wait-backup"
      elif [ "$backup_failed" = "yes" ] || [ "$ckpt_alive" = "no" ]; then
        echo "WARNING: backup unhealthy, continuing to monitor training"
        BACKUP_HEALTHY=false
      fi
      ;;

    completed-wait-backup)
      completed_exists=$($SSH_CMD "test -f '${CONTROL_DIR}/state/completed' && echo yes || echo no" 2>/dev/null)
      if [ "$completed_exists" != "yes" ]; then
        echo "WARNING: completed marker disappeared, falling back to running state"
        STATE="running"
        continue
      fi
      final_succeeded=$($SSH_CMD "test -f '${CONTROL_DIR}/state/backup-final-succeeded' && cat '${CONTROL_DIR}/state/backup-final-succeeded' || echo not_found" 2>/dev/null)
      launch_eval=false

      if [ -n "$final_succeeded" ] && [ "$final_succeeded" != "not_found" ] && [ "$final_succeeded" = "$REQUEST_TOKEN" ]; then
        echo "Post-completion backup acknowledged (token ${final_succeeded:0:10}...). Launching held-out eval."
        launch_eval=true
      else
        ckpt_alive=$($SSH_CMD "tmux has-session -t ckpt-bkp 2>/dev/null && echo yes || echo no" 2>/dev/null)
        backup_failed=$($SSH_CMD "test -f '${CONTROL_DIR}/state/backup-failed' && echo yes || echo no" 2>/dev/null)
        if [ "$ckpt_alive" = "no" ] || [ "$backup_failed" = "yes" ]; then
          echo "WARNING: backup is dead, launching held-out eval from local data"
          upload_diagnostics
          launch_eval=true
        fi
      fi

      if [ "$launch_eval" = true ]; then
        $SSH_CMD "tmux kill-session -t ckpt-bkp 2>/dev/null; echo ckpt-bkp-stopped" >/dev/null 2>&1
        . "${CONTROL_DIR}/s3-env.env" 2>/dev/null || true
        $SSH_CMD "tmux new-session -d -s heldout-eval 'cd /workspace/toy-pickplace && export MUJOCO_GL=egl && exec bash ${CONTROL_DIR}/heldout-eval.sh'" 2>&1
        HELDOUT_LAUNCHED=true
        STATE="evaluation-running"
        echo "Held-out evaluation launched"
      fi
      ;;

    evaluation-running)
      heldout_completed=$($SSH_CMD "test -f '${CONTROL_DIR}/state/heldout-completed' && echo yes || echo no" 2>/dev/null)
      heldout_failed=$($SSH_CMD "test -f '${CONTROL_DIR}/state/heldout-failed' && cat '${CONTROL_DIR}/state/heldout-failed' || echo not_found" 2>/dev/null)

      if [ "$heldout_completed" = "yes" ]; then
        echo "Held-out evaluation completed successfully"
        . "${CONTROL_DIR}/s3-env.env" 2>/dev/null || true
        verify_ok=true
        for key in "results/flywheel/${ARCH}/${RUN_NAME}/resolved-config.yaml" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/experiment-manifest.json" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/final_scores.json" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/final-placement-score.png" \
                   "results/flywheel/${ARCH}/${RUN_NAME}/final-score-curve.png"; do
          uv run python -c "
import boto3, os
c = boto3.client('s3', region_name='${AWS_REGION:-ap-south-1}')
r = c.head_object(Bucket='${S3_BUCKET}', Key='$key')
print(f'Verified: s3://${S3_BUCKET}/\$key ({r[\"ContentLength\"]} bytes)')
" 2>/dev/null || { verify_ok=false; echo "FAILED to verify: $key"; }
        done
        if [ "$verify_ok" = true ]; then
          cleanup "evaluation-verified"
        else
          cleanup "evaluation-verification-failed"
        fi
        break
      fi
      if [ "$heldout_failed" != "not_found" ]; then
        echo "Held-out evaluation failed: $heldout_failed"
        upload_diagnostics
        cleanup "heldout-failed"
        break
      fi
      ;;
  esac

  ssh_ok=false
  if $SSH_CMD "echo health" >/dev/null 2>&1; then
    ssh_ok=true
    SSH_FAILURES=0
  else
    SSH_FAILURES=$((SSH_FAILURES + 1))
    echo "SSH probe failed ($SSH_FAILURES consecutive)"

    if [ $SSH_FAILURES -ge 3 ]; then
      status=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
        uv run python -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','unknown'))" 2>/dev/null)
      echo "Vast status: $status"
      if [ "$status" = "running" ]; then
        new_url=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
        if [ -n "$new_url" ]; then
          NEW_HOST=$(echo "$new_url" | sed 's|ssh://root@||;s|:.*||')
          NEW_PORT=$(echo "$new_url" | sed 's|.*:||')
          if [ -n "$NEW_HOST" ] && [ -n "$NEW_PORT" ] && [ "$NEW_HOST" != "$HOST" ]; then
            echo "Endpoint changed: $HOST:$PORT -> $NEW_HOST:$NEW_PORT"
            ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p "$NEW_PORT" "root@$NEW_HOST" "echo pinning" >/dev/null 2>&1
            HOST=$NEW_HOST; PORT=$NEW_PORT
            SSH_CMD="ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$HOST"
            SSH_FAILURES=0
            ssh_ok=true
          fi
        fi
      elif echo "$status" | grep -qE "offline|stopped|error|terminated"; then
        echo "Instance terminal: $status"
        upload_diagnostics
        cleanup "instance-terminated-$status"
        break
      fi
    fi

    if [ $SSH_FAILURES -ge 120 ]; then
      echo "ALERT: 120 consecutive SSH failures, continuing at 5-minute interval"
      sleep 270
    fi
  fi

  if [ $((SECONDS % 120)) -lt 30 ]; then
    echo "[$(date +%H:%M:%S)] STATE=$STATE SSH_FAILURES=$SSH_FAILURES"
  fi

  sleep 30
done
