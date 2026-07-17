---
description: Flywheel ablation sweep for intervention_threshold on one Vast.ai RTX 4090
agent: build
---

Run an ablation of `intervention_threshold` on a **single** Vast.ai RTX 4090 instance.
All other flywheel params stay at baseline; only `--intervention-threshold` varies per
run. `/flywheel-4090 intervention-threshold` provisions the instance, applies profile
tuning, and launches the **parallel** sweep by default; use the sequential batch only
when the orchestrator falls back.

This doc is the source of truth for run names, sweep order, ablation launch
commands, TensorBoard setup, and the backup prefix.

Planning reference: [`canvas/flywheel-ablation-runner.canvas.tsx`](../../canvas/flywheel-ablation-runner.canvas.tsx)  
Benchmark reference: [`docs/vast-ai/ablation-study.md`](../vast-ai/ablation-study.md)

## Agent instructions

1. Source `.env` and verify `VAST_API_KEY` and `HF_TOKEN` (see flywheel-4090 step 0).
2. Run `/flywheel-4090 intervention-threshold` — it owns provisioning, tuning, and
   default parallel launch.
3. Do **not** commit instance-specific `configs/flywheel/default.yaml` overrides to the
   repo.
4. After launch: local SSH tunnel on port **6006**, HF backup in `ckpt-bkp`, print
   monitor URLs. Do **not** wait for the sweep to finish.
5. Optional offline download after the user asks or the sweep completes (section below).

All modes use **one** instance with four flywheel processes. Do **not** provision one
instance per threshold value.

---

## Parameter under test

| Field | Value |
|---|---|
| CLI flag | `--intervention-threshold` |
| Config key | `intervention_threshold` |
| Default (`configs/flywheel/default.yaml`) | `0.1` |
| Meaning | L2 arm-action threshold; gripper disagreement also triggers expert takeover |

All other params come from `configs/flywheel/default.yaml` plus 4090 instance overrides.

---

## Ablation sweep

Four runs, one threshold each:

| Order | Run name | `--intervention-threshold` | Notes |
|---:|---|---:|---|
| 1 | `abl-intervention_threshold-0.05` | `0.05` | Lowest threshold — most expert interventions |
| 2 | `abl-intervention_threshold-0.1` | `0.1` | Config default |
| 3 | `abl-intervention_threshold-0.2` | `0.2` | |
| 4 | `abl-intervention_threshold-0.3` | `0.3` | |

Each run is a full flywheel (expert collection → DAgger rounds → train/eval).

**TensorBoard metrics to compare:** `eval/placement_success_rate`, `eval/score`,
`train/loss`. Filter scalars by prefix `abl-intervention_threshold-`.

---

## Provision + launch

`/flywheel-4090 intervention-threshold` runs shared provisioning and instance setup
(search → create → SSH → clone → `uv sync` → profile tuning). Substitute `$HOST` and
`$PORT` from `vastai ssh-url "$INSTANCE_ID"`.

### Batch 1 — verify 4090 tuning

```bash
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
  "grep -E '^(workers|batch_size|dataloader_workers):' \
     /workspace/toy-pickplace/configs/flywheel/default.yaml"
```

Expected for **parallel** (default):

```
workers: 6
batch_size: 768
dataloader_workers: 2
```

Expected for **sequential** (fallback):

```
workers: 12
batch_size: 768
dataloader_workers: 0
```

If values differ, apply the profile overrides from `flywheel-4090.md` before continuing.

### Batch 2 — launch parallel sweep + TensorBoard

Four concurrent tmux sessions on one instance — `/flywheel-4090` uses this by default.

```bash
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s ablation-0.05 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.05 \
        --intervention-threshold 0.05' && \
   tmux new-session -d -s ablation-0.1 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.1 \
        --intervention-threshold 0.1' && \
   tmux new-session -d -s ablation-0.2 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.2 \
        --intervention-threshold 0.2' && \
   tmux new-session -d -s ablation-0.3 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.3 \
        --intervention-threshold 0.3' && \
   tmux new-session -d -s tensorboard \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

### Batch 2 — launch sequential sweep + TensorBoard

One tmux session; runs execute **back-to-back**. Use only when `/flywheel-4090`
falls back to sequential.

```bash
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s ablation-sweep \
     'cd /workspace/toy-pickplace && \
      echo Starting abl-intervention_threshold-0.05 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.05 \
        --intervention-threshold 0.05 && \
      echo Starting abl-intervention_threshold-0.1 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.1 \
        --intervention-threshold 0.1 && \
      echo Starting abl-intervention_threshold-0.2 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.2 \
        --intervention-threshold 0.2 && \
      echo Starting abl-intervention_threshold-0.3 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-intervention_threshold-0.3 \
        --intervention-threshold 0.3 && \
      echo Ablation sweep complete' && \
   tmux new-session -d -s tensorboard \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

### Batch 3 — local SSH tunnel (dev machine)

```bash
SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(echo "$SSH_URL" | sed 's/.*@//;s/:.*//')
PORT=$(echo "$SSH_URL" | sed 's/.*://')

tmux new-session -d -s tb-ablation \
  "ssh -N -L 6006:127.0.0.1:6006 -p $PORT root@$HOST"
```

### Batch 4 — HF checkpoint backup

After Batch 2, use the `ckpt-bkp` procedure in
[`flywheel-4090.md`](../../.opencode/commands/flywheel-4090.md). Use a backup
prefix such as `ablation-intervention-threshold-YYYYMMDD-HHMMSS`; do not start
a second `ckpt-bkp` session.

### Verify sessions

```bash
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" "tmux ls"
```

Expected for **parallel**: `ablation-0.05`, `ablation-0.1`, `ablation-0.2`,
`ablation-0.3`, `tensorboard`, `ckpt-bkp`.

Expected for **sequential**: `ablation-sweep`, `tensorboard`, `ckpt-bkp`.

### Monitor

| What | Where |
|---|---|
| TensorBoard (local browser) | http://localhost:6006 |
| Attach to parallel run | `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t ablation-0.05'` (or `0.1`, `0.2`, `0.3`) |
| Attach to sequential sweep | `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t ablation-sweep'` |
| Attach to TensorBoard | `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t tensorboard'` |
| Remote logdir | `/workspace/toy-pickplace/runs/flywheel/` |

All four runs share one logdir and overlay in the TensorBoard scalars tab.

---

## Standalone sweep scripts

Save on the instance, then launch with the matching tmux wrapper.

**Parallel** — `/workspace/toy-pickplace/scripts/ablation-intervention-threshold-parallel.sh`:

```bash
#!/bin/bash
# intervention_threshold ablation — parallel, one instance
set -euo pipefail
cd /workspace/toy-pickplace

UV=/root/.local/bin/uv
CFG=configs/flywheel/default.yaml

launch() {
  local threshold=$1
  tmux new-session -d -s "ablation-${threshold}" \
    "cd /workspace/toy-pickplace && exec $UV run python scripts/flywheel.py \
      --config $CFG --run-name abl-intervention_threshold-${threshold} \
      --intervention-threshold ${threshold}"
}

launch 0.05
launch 0.1
launch 0.2
launch 0.3

echo "Parallel ablation sweep launched."
```

**Sequential** — `/workspace/toy-pickplace/scripts/ablation-intervention-threshold-sequential.sh`:

```bash
#!/bin/bash
# intervention_threshold ablation — sequential, one instance
set -euo pipefail
cd /workspace/toy-pickplace

UV=/root/.local/bin/uv
CFG=configs/flywheel/default.yaml

echo "=== Run 1/4: abl-intervention_threshold-0.05 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-intervention_threshold-0.05 \
  --intervention-threshold 0.05

echo "=== Run 2/4: abl-intervention_threshold-0.1 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-intervention_threshold-0.1 \
  --intervention-threshold 0.1

echo "=== Run 3/4: abl-intervention_threshold-0.2 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-intervention_threshold-0.2 \
  --intervention-threshold 0.2

echo "=== Run 4/4: abl-intervention_threshold-0.3 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-intervention_threshold-0.3 \
  --intervention-threshold 0.3

echo "Ablation sweep complete."
```

Launch wrappers:

```bash
# parallel
tmux new-session -d -s tensorboard \
  'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'
bash /workspace/toy-pickplace/scripts/ablation-intervention-threshold-parallel.sh

# sequential
tmux new-session -d -s ablation-sweep \
  'bash /workspace/toy-pickplace/scripts/ablation-intervention-threshold-sequential.sh'
tmux new-session -d -s tensorboard \
  'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'
```

---

## Optional offline download for unified viewing

Use after at least one run has produced checkpoints, or after the full sweep completes.
Run on the **dev machine** (not the instance).

### List available backup sessions

```bash
set -a; . ./.env; set +a
uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel list
```

### Download all four runs

Replace `SESSION` with the backup prefix from the list (e.g. `ablation-intervention-threshold-20260717-153000`).

```bash
set -a; . ./.env; set +a
mkdir -p runs/flywheel/ablations/intervention_threshold

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-intervention_threshold-0.05 \
  runs/flywheel/ablations/intervention_threshold/abl-intervention_threshold-0.05

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-intervention_threshold-0.1 \
  runs/flywheel/ablations/intervention_threshold/abl-intervention_threshold-0.1

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-intervention_threshold-0.2 \
  runs/flywheel/ablations/intervention_threshold/abl-intervention_threshold-0.2

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-intervention_threshold-0.3 \
  runs/flywheel/ablations/intervention_threshold/abl-intervention_threshold-0.3
```

### Local TensorBoard (no SSH tunnel)

```bash
uv run python -m tensorboard.main \
  --logdir runs/flywheel/ablations/intervention_threshold \
  --host 127.0.0.1 --port 6010
```

Open http://localhost:6010 — all four thresholds overlaid.

### Re-evaluate best checkpoint (optional)

After picking a winner from TensorBoard / `results/flywheel/*/metrics.json`:

```bash
uv run python scripts/final_score.py --run-name abl-intervention_threshold-0.2
```

---

## Cleanup

When the sweep is done and backups verified:

```bash
vastai destroy instance "$INSTANCE_ID"
```

Confirm the instance no longer appears in `vastai show instances`.
