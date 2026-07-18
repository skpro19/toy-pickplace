#!/usr/bin/env bash
# ssh-forward.sh -- Create local tmux wrappers for a running Vast.ai instance.
#
# Creates two local tmux sessions:
#   vast-ssh       persistent SSH shell to the remote instance
#   tb-ablation    SSH tunnel forwarding remote TensorBoard (6006) to localhost
#
# Usage:
#   bash/ssh-forward.sh                    # auto-detect first running instance
#   bash/ssh-forward.sh <INSTANCE_ID>      # use a specific instance
#   bash/ssh-forward.sh -h | --help        # show this help
#
# The script is safe to re-run: it kills stale local sessions before creating
# new ones. This only affects local tmux sessions -- remote training
# (ablation-controller, cell sessions, TensorBoard, ckpt-bkp) is not touched.
#
# Requirements: vastai CLI, tmux, jq
set -euo pipefail

usage() {
  echo "Usage: bash/ssh-forward.sh [INSTANCE_ID]"
  echo
  echo "Create local tmux wrappers for a running Vast.ai instance:"
  echo "  vast-ssh       persistent SSH session"
  echo "  tb-ablation    SSH tunnel forwarding TensorBoard (6006)"
  echo
  echo "If no INSTANCE_ID is given, the first running instance is used."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INSTANCE_ID="${1:-}"

if [[ -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | jq -r '.[0].id // empty')
  if [[ -z "$INSTANCE_ID" ]]; then
    echo "Error: no running instance found. Provide an instance ID." >&2
    exit 1
  fi
  echo "Using instance: $INSTANCE_ID"
fi

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(echo "$SSH_URL" | sed 's|ssh://root@||' | sed 's|:[0-9]*$||')
PORT=$(echo "$SSH_URL" | sed 's|.*:||')

tmux kill-session -t vast-ssh 2>/dev/null || true
tmux kill-session -t tb-ablation 2>/dev/null || true

tmux new-session -d -s vast-ssh \
  "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p $PORT root@$HOST"

tmux new-session -d -s tb-ablation \
  "ssh -N -L 6006:127.0.0.1:6006 -p $PORT root@$HOST"

if tmux has-session -t vast-ssh 2>/dev/null && tmux has-session -t tb-ablation 2>/dev/null; then
  echo "Wrappers created:"
  echo "  vast-ssh:       tmux attach -t vast-ssh"
  echo "  tb-ablation:    tmux attach -t tb-ablation"
  echo "  TensorBoard:    http://localhost:6006"
else
  echo "Error: failed to create tmux wrappers" >&2
  exit 1
fi
