#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash/ablation-status.sh [INSTANCE_ID]"
  echo
  echo "Show pretty-printed status of the currently running ablation grid."
  echo "If no INSTANCE_ID is given, the first running instance is used."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INSTANCE_ID="${1:-}"

# ── Detect instance ──────────────────────────────────────────────────
if [[ -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | jq -r '.[0].id // empty')
  if [[ -z "$INSTANCE_ID" ]]; then
    echo "Error: no running instance found." >&2
    exit 1
  fi
fi

# ── Get SSH details ──────────────────────────────────────────────────
SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(echo "$SSH_URL" | sed 's|ssh://root@||' | sed 's|:[0-9]*$||')
PORT=$(echo "$SSH_URL" | sed 's|.*:||')
SSH_BASE="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p $PORT root@$HOST"

ssh_cmd() {
  $SSH_BASE "$@"
}

# ── Instance info ────────────────────────────────────────────────────
INSTANCE_JSON=$(vastai show instances --raw 2>/dev/null | jq -r ".[] | select(.id == $INSTANCE_ID) // empty")
if [[ -z "$INSTANCE_JSON" ]]; then
  echo "Error: instance $INSTANCE_ID not found." >&2
  exit 1
fi

INSTANCE_LABEL=$(echo "$INSTANCE_JSON" | jq -r '.label // "—"')
INSTANCE_GPU=$(echo "$INSTANCE_JSON" | jq -r '.gpu_name // "—"')
INSTANCE_START_EPOCH=$(echo "$INSTANCE_JSON" | jq -r 'if (.start_date | type) == "number" then .start_date | floor else empty end')
if [[ -n "$INSTANCE_START_EPOCH" && "$INSTANCE_START_EPOCH" -le "$(date +%s)" ]]; then
  INSTANCE_UPTIME_MIN=$(( ($(date +%s) - INSTANCE_START_EPOCH) / 60 ))
else
  INSTANCE_UPTIME_MIN=$(echo "$INSTANCE_JSON" | jq -r '.uptime_mins // 0' | cut -d. -f1)
  INSTANCE_UPTIME_MIN=$(( INSTANCE_UPTIME_MIN < 0 ? 0 : INSTANCE_UPTIME_MIN ))
fi
INSTANCE_AGE_HR=$(echo "$INSTANCE_JSON" | jq -r '.client_age // "—"')
INSTANCE_DPH=$(echo "$INSTANCE_JSON" | jq -r '.dph_total // "—"')
INSTANCE_DPH_FMT=$(printf "%.4f" "$INSTANCE_DPH" 2>/dev/null || echo "$INSTANCE_DPH")
INSTANCE_STATUS=$(echo "$INSTANCE_JSON" | jq -r '.actual_status // "—"')

# ── Find ablation control dir ────────────────────────────────────────
CONTROL_DIR=$(ssh_cmd "ls -d /workspace/toy-pickplace/.ablation/*/ 2>/dev/null | tail -1" 2>/dev/null || true)
if [[ -z "$CONTROL_DIR" ]]; then
  echo "╔══════════════════════════════════════════╗"
  echo "║  No ablation grid found on instance $INSTANCE_ID  ║"
  echo "╚══════════════════════════════════════════╝"
  exit 0
fi
CONTROL_DIR=${CONTROL_DIR%/}

# ── Parse plan ───────────────────────────────────────────────────────
PLAN=$(ssh_cmd "cat '$CONTROL_DIR/plan' 2>/dev/null" 2>/dev/null || true)

if [[ -z "$PLAN" ]]; then
  echo "╔══════════════════════════════════════════╗"
  echo "║  Plan file not found in ablation dir     ║"
  echo "╚══════════════════════════════════════════╝"
  exit 0
fi

TOTAL_CELLS=$(echo "$PLAN" | wc -l)
UNIQUE_BATCHES=$(echo "$PLAN" | cut -d'|' -f1 | sed 's/batch=//' | sort -n | tail -1)
FIRST_BATCH=$(echo "$PLAN" | cut -d'|' -f1 | sed 's/batch=//' | sort -n | head -1)

# Derive param abbreviations from run names (e.g. "abl-it0p1-dir0" → "it, dir")
# Plan rows use: batch|run_name|tmux_session.
FIRST_RUN=$(echo "$PLAN" | head -1 | cut -d'|' -f2)
STRIPPED=$(echo "$FIRST_RUN" | sed 's/^abl-//')
PARAM_KEYS=""
IFS='-' read -ra SEGMENTS <<< "$STRIPPED"
for seg in "${SEGMENTS[@]}"; do
  key=$(echo "$seg" | sed 's/[0-9.].*//')
  if [[ -n "$key" ]]; then
    if [[ -n "$PARAM_KEYS" ]]; then PARAM_KEYS+=", "; fi
    PARAM_KEYS+="$key"
  fi
done

# ── State ────────────────────────────────────────────────────────────
STATE_FILES=$(ssh_cmd "ls '$CONTROL_DIR/state/' 2>/dev/null" 2>/dev/null || true)

BATCHES_DONE=0
BATCHES_RUNNING=0
for b in $(seq "$FIRST_BATCH" "$UNIQUE_BATCHES"); do
  if echo "$STATE_FILES" | grep -q "batch-$b-succeeded"; then
    BATCHES_DONE=$((BATCHES_DONE + 1))
  elif echo "$STATE_FILES" | grep -q "batch-$b-started"; then
    BATCHES_RUNNING=$((BATCHES_RUNNING + 1))
  fi
done
BATCHES_PENDING=$((UNIQUE_BATCHES - BATCHES_DONE - BATCHES_RUNNING))

COMPLETED=$(ssh_cmd "test -f '$CONTROL_DIR/state/completed' && printf yes" 2>/dev/null || true)
FAILED=$(ssh_cmd "cat '$CONTROL_DIR/state/failed' 2>/dev/null" 2>/dev/null || true)

# ── Cell statuses ────────────────────────────────────────────────────
CELL_STATUS_RAW=$(ssh_cmd "for f in '$CONTROL_DIR/status/'*; do echo \"\$(basename \"\$f\"):\$(cat \"\$f\")\"; done" 2>/dev/null || true)

# Parse run names from batch|run_name|tmux_session plan rows.
RUN_NAMES=$(echo "$PLAN" | cut -d'|' -f2)
CELLS_SUCCEEDED=0
CELLS_RUNNING=0
CELLS_FAILED=0
CELLS_WAITING=0

while IFS= read -r run_name; do
  status_line=$(echo "$CELL_STATUS_RAW" | grep "^$run_name:" || true)
  if [[ -z "$status_line" ]]; then
    CELLS_WAITING=$((CELLS_WAITING + 1))
  elif echo "$status_line" | grep -q "succeeded 0"; then
    CELLS_SUCCEEDED=$((CELLS_SUCCEEDED + 1))
  elif echo "$status_line" | grep -q "running"; then
    CELLS_RUNNING=$((CELLS_RUNNING + 1))
  else
    CELLS_FAILED=$((CELLS_FAILED + 1))
  fi
done <<< "$RUN_NAMES"

# ── Round progress for running cells (via TensorBoard) ───────────────
TENSORBOARD_RUNS=$(ssh_cmd "curl -sf http://127.0.0.1:6006/data/runs 2>/dev/null" 2>/dev/null || true)

# Get max rounds from config
DAGGER_ROUNDS=$(ssh_cmd "grep -E '^dagger_rounds:' /workspace/toy-pickplace/configs/flywheel/default.yaml 2>/dev/null | awk '{print \$2}'" 2>/dev/null || echo "10")

# Get running cell names
RUNNING_CELLS=$(echo "$CELL_STATUS_RAW" | grep ":running$" | cut -d: -f1 || true)

declare -A ROUND_PROGRESS
while IFS= read -r cell; do
  if [[ -z "$cell" ]]; then continue; fi
  round_count=$(echo "$TENSORBOARD_RUNS" | tr ',' '\n' | grep -c "\"$cell/round-" || true)
  ROUND_PROGRESS["$cell"]=$round_count
done <<< "$RUNNING_CELLS"

# ── Render ───────────────────────────────────────────────────────────
W1=30
W2=18

hr() { printf '╟%s╢\n' "$(printf '═%.0s' $(seq 1 $((W1 + W2 + 1))))"; }
sep() { printf '║ %-*s %s\n' "$W1" "$1" "$2"; }
subsep() { printf '║   %-*s %s\n' "$((W1 - 3))" "$1" "$2"; }
row3() { printf '║ %-*s │ %-*s │ %-*s ║\n' 13 "$1" 13 "$2" 12 "$3"; }

echo ""
echo "╔$(printf '═%.0s' $(seq 1 $((W1 + W2 + 1))))╗"
echo "║  Ablation Status  $(printf '%*s' $((W1 + W2 - 18)) '')║"
echo "╟$(printf '═%.0s' $(seq 1 $((W1 + W2 + 1))))╢"

# Instance info
sep "Instance" "$INSTANCE_ID"
sep "Label" "$INSTANCE_LABEL"
sep "GPU" "$INSTANCE_GPU"
sep "Status" "$INSTANCE_STATUS"
sep "Uptime" "${INSTANCE_UPTIME_MIN} min"
sep "Cost" "\$$INSTANCE_DPH_FMT / hr"

echo "╟$(printf '─%.0s' $(seq 1 $((W1 + W2 + 1))))╢"

# Grid overview
sep "Parameters" "$PARAM_KEYS"
sep "Total cells" "$TOTAL_CELLS"
sep "Total batches" "$UNIQUE_BATCHES"

echo "╟$(printf '─%.0s' $(seq 1 $((W1 + W2 + 1))))╢"

# Batch progress
sep "Batch progress" ""
subsep "Done" "$BATCHES_DONE / $UNIQUE_BATCHES"
subsep "Running" "$BATCHES_RUNNING / $UNIQUE_BATCHES"
subsep "Pending" "$BATCHES_PENDING / $UNIQUE_BATCHES"

echo "╟$(printf '─%.0s' $(seq 1 $((W1 + W2 + 1))))╢"

# Cell progress
sep "Cell progress" ""
subsep "Succeeded" "$CELLS_SUCCEEDED / $TOTAL_CELLS"
subsep "Running" "$CELLS_RUNNING / $TOTAL_CELLS"
subsep "Pending" "$CELLS_WAITING / $TOTAL_CELLS"
if [[ "$CELLS_FAILED" -gt 0 ]]; then
  subsep "Failed" "$CELLS_FAILED / $TOTAL_CELLS"
fi

# ── Running cells round progress ───────────────────────────────────
if [[ "$CELLS_RUNNING" -gt 0 ]]; then
  echo "╟$(printf '─%.0s' $(seq 1 $((W1 + W2 + 1))))╢"
  sep "Rounds progress" "(max $DAGGER_ROUNDS)"
  while IFS= read -r cell; do
    if [[ -z "$cell" ]]; then continue; fi
    rounds=${ROUND_PROGRESS[$cell]:-0}
    display_rounds=$(( rounds > DAGGER_ROUNDS ? DAGGER_ROUNDS : rounds ))
    filled=$(( display_rounds * 10 / DAGGER_ROUNDS ))
    empty=$((10 - filled))
    bar=""
    for ((i = 0; i < filled; i++)); do bar="${bar}█"; done
    for ((i = 0; i < empty; i++)); do bar="${bar}░"; done
    printf '║ %-16s %3d/%-3d %s║\n' "$cell" "$rounds" "$DAGGER_ROUNDS" "$bar"
  done <<< "$RUNNING_CELLS"
fi

# ── Final state ────────────────────────────────────────────────────
echo "╟$(printf '═%.0s' $(seq 1 $((W1 + W2 + 1))))╢"
if [[ -n "$COMPLETED" ]]; then
  echo "║  ✅  All batches completed!                          ║"
elif [[ -n "$FAILED" ]]; then
  echo "║  ❌  A cell failed: $FAILED  ║"
else
  remaining_cells=$((TOTAL_CELLS - CELLS_SUCCEEDED))
  echo "║  ⏳  $remaining_cells / $TOTAL_CELLS cells remaining                    ║"
fi

echo "╚$(printf '═%.0s' $(seq 1 $((W1 + W2 + 1))))╝"
echo ""
