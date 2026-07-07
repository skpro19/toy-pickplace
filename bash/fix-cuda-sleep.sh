#!/usr/bin/env bash
set -euo pipefail

kill_stale=false

usage() {
  echo "Usage: bash/fix-cuda-sleep.sh [--kill-stale]"
  echo
  echo "Options:"
  echo "  --kill-stale  Stop user-owned processes using nvidia_uvm before retrying."
}

validate_sudo() {
  if [[ -t 0 ]]; then
    sudo -v
  else
    sudo -S -v
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill-stale)
      kill_stale=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

active_uvm_pids() {
  local devices=()
  local output
  local pid

  [[ -e /dev/nvidia-uvm ]] && devices+=(/dev/nvidia-uvm)
  [[ -e /dev/nvidia-uvm-tools ]] && devices+=(/dev/nvidia-uvm-tools)

  if [[ ${#devices[@]} -eq 0 ]]; then
    return
  fi

  output=$(sudo fuser "${devices[@]}" 2>&1 || true)

  for pid in $output; do
    if [[ $pid =~ ^([0-9]+) ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
    fi
  done | sort -u
}

kill_stale_uvm_users() {
  mapfile -t pids < <(active_uvm_pids)

  if [[ ${#pids[@]} -eq 0 ]]; then
    echo "No active NVIDIA UVM users found."
    return
  fi

  echo "Stopping user-owned NVIDIA UVM users: ${pids[*]}"
  ps -o pid,ppid,user,stat,etime,cmd -p "$(IFS=,; echo "${pids[*]}")" || true

  for pid in "${pids[@]}"; do
    if [[ $(ps -o user= -p "$pid" 2>/dev/null | xargs) == "$USER" ]]; then
      kill "$pid" 2>/dev/null || true
    else
      echo "Skipping PID $pid: not owned by $USER."
    fi
  done

  sleep 2
  mapfile -t pids < <(active_uvm_pids)

  for pid in "${pids[@]}"; do
    if [[ $(ps -o user= -p "$pid" 2>/dev/null | xargs) == "$USER" ]]; then
      echo "Force-killing PID $pid."
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

echo "Reloading NVIDIA UVM module..."
validate_sudo

if ! sudo modprobe -r nvidia_uvm; then
  echo "Could not unload nvidia_uvm. It is probably being used by a CUDA process."
  echo "Active NVIDIA device users:"
  sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidiactl || true

  if [[ $kill_stale == false ]]; then
    echo "Stop active CUDA jobs and run this script again."
    echo "Or rerun with --kill-stale to stop user-owned NVIDIA UVM users automatically."
    echo "If CUDA still fails, reboot the laptop."
    exit 1
  fi

  kill_stale_uvm_users

  echo "Retrying NVIDIA UVM unload..."
  if ! sudo modprobe -r nvidia_uvm; then
    echo "Could not unload nvidia_uvm after stopping user-owned processes. Reboot the laptop."
    exit 1
  fi
fi

sudo modprobe nvidia_uvm

echo "NVIDIA UVM module reloaded."

if [[ -x /usr/local/cuda-12.8/extras/demo_suite/deviceQuery ]]; then
  echo "Running CUDA deviceQuery..."
  /usr/local/cuda-12.8/extras/demo_suite/deviceQuery
else
  echo "Skipping deviceQuery: /usr/local/cuda-12.8/extras/demo_suite/deviceQuery not found."
fi

if command -v uv >/dev/null 2>&1 && [[ -f pyproject.toml ]]; then
  echo "Running PyTorch CUDA check..."
  uv run python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
else
  echo "Skipping PyTorch check: uv or pyproject.toml not found."
fi

echo "Done. If CUDA still fails, reboot the laptop."
