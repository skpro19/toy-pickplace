#!/usr/bin/env bash
set -euo pipefail

echo "Reloading NVIDIA UVM module..."
sudo -v

if ! sudo modprobe -r nvidia_uvm; then
  echo "Could not unload nvidia_uvm. It is probably being used by a CUDA process."
  echo "Active NVIDIA device users:"
  sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidiactl || true
  echo "Stop active CUDA jobs and run this script again. If CUDA still fails, reboot the laptop."
  exit 1
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
