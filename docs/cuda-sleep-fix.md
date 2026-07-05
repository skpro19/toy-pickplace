# CUDA Sleep Fix

## Symptom

After resuming the laptop from sleep, CUDA may stop working even though `nvidia-smi` still detects the GPU.

Observed failures:

```text
torch.cuda.is_available() == False
CUDA initialization: CUDA unknown error
```

The lower-level CUDA runtime failed in the same state:

```text
cudaGetDeviceCount returned 999
-> unknown error
Result = FAIL
```

The training command still ran, but silently fell back to CPU:

```bash
uv run scripts/train.py --epochs 100 --npz data/test-new
```

```text
Using device: cpu
```

## Cause

This was not a project dependency or PyTorch package issue. The machine could see the GPU through `nvidia-smi`, but the CUDA runtime could not initialize it after suspend/resume.

Reloading the NVIDIA UVM kernel module fixed the stale CUDA runtime state.

## Fix

Run the helper script from the repository root:

```bash
bash bash/fix-cuda-sleep.sh
```

The script reloads `nvidia_uvm` with `sudo`, then checks CUDA using `deviceQuery` and PyTorch.

The manual equivalent is:

```bash
sudo modprobe -r nvidia_uvm
sudo modprobe nvidia_uvm
```

Then verify:

```bash
/usr/local/cuda-12.8/extras/demo_suite/deviceQuery
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result:

```text
Result = PASS
True
NVIDIA GeForce RTX 3070 Laptop GPU
```

After the fix, training should report:

```text
Using device: cuda
```

If reloading `nvidia_uvm` fails or CUDA still returns error `999`, reboot the laptop.
