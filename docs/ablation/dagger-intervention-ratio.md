---
description: Flywheel ablation sweep for dagger_intervention_ratio on one Vast.ai RTX 4090
agent: build
---

Run an ablation of `dagger_intervention_ratio` on a **single** Vast.ai RTX 4090 instance.
All other flywheel params stay at baseline; only `--dagger-intervention-ratio` varies per
run. `/flywheel-4090 dagger-intervention-ratio` provisions the instance, applies profile
tuning, and launches the **parallel** sweep by default; use the sequential batch only
when the orchestrator falls back.

This doc is the source of truth for run names, sweep order, ablation launch
commands, TensorBoard setup, and the backup prefix.

Planning reference: [`canvas/flywheel-ablation-runner.canvas.tsx`](../../canvas/flywheel-ablation-runner.canvas.tsx)  
Benchmark reference: [`docs/vast-ai/ablation-study.md`](../vast-ai/ablation-study.md)

## Agent instructions

1. Source `.env` and verify `VAST_API_KEY` and `HF_TOKEN` (see flywheel-4090 step 0).
2. Run `/flywheel-4090 dagger-intervention-ratio` — it owns provisioning, tuning, and
   default parallel launch.
3. Do **not** commit instance-specific `configs/flywheel/default.yaml` overrides to the
   repo.
4. After launch: local SSH tunnel on port **6006**, HF backup in `ckpt-bkp`, print
   monitor URLs. Do **not** wait for the sweep to finish.
5. Optional offline download after the user asks or the sweep completes (section below).

All modes use **one** instance with four flywheel processes. Do **not** provision one
instance per ratio value.

---

## Parameter under test

| Field | Value |
|---|---|
| CLI flag | `--dagger-intervention-ratio` |
| Config key | `dagger_intervention_ratio` |
| Default (`configs/flywheel/default.yaml`) | `0.8` |
| Meaning | Sampling share for expert-executed frames within DAgger data (retrain rounds only) |

All other params come from `configs/flywheel/default.yaml` plus 4090 instance overrides.

---

## Ablation sweep

Four runs, one ratio each:

| Order | Run name | `--dagger-intervention-ratio` | Notes |
|---:|---|---:|---|
| 1 | `abl-dagger_intervention_ratio-0.2` | `0.2` | Lowest ratio — most policy-executed frames in DAgger sampling |
| 2 | `abl-dagger_intervention_ratio-0.5` | `0.5` | Balanced expert vs policy frames |
| 3 | `abl-dagger_intervention_ratio-0.8` | `0.8` | Config default |
| 4 | `abl-dagger_intervention_ratio-1.0` | `1.0` | Expert-executed frames only within DAgger data |

Each run is a full flywheel (expert collection → DAgger rounds → train/eval).

**TensorBoard metrics to compare:** `eval/placement_success_rate`, `eval/score`,
`train/loss`. Filter scalars by prefix `abl-dagger_intervention_ratio-`.

---

## Provision + launch

`/flywheel-4090 dagger-intervention-ratio` runs shared provisioning and instance setup
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
  "tmux new-session -d -s ablation-0.2 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-0.2 \
        --dagger-intervention-ratio 0.2' && \
   tmux new-session -d -s ablation-0.5 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-0.5 \
        --dagger-intervention-ratio 0.5' && \
   tmux new-session -d -s ablation-0.8 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-0.8 \
        --dagger-intervention-ratio 0.8' && \
   tmux new-session -d -s ablation-1.0 \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-1.0 \
        --dagger-intervention-ratio 1.0' && \
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
      echo Starting abl-dagger_intervention_ratio-0.2 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-0.2 \
        --dagger-intervention-ratio 0.2 && \
      echo Starting abl-dagger_intervention_ratio-0.5 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-0.5 \
        --dagger-intervention-ratio 0.5 && \
      echo Starting abl-dagger_intervention_ratio-0.8 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-0.8 \
        --dagger-intervention-ratio 0.8 && \
      echo Starting abl-dagger_intervention_ratio-1.0 && \
      /root/.local/bin/uv run python scripts/flywheel.py \
        --config configs/flywheel/default.yaml \
        --run-name abl-dagger_intervention_ratio-1.0 \
        --dagger-intervention-ratio 1.0 && \
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
prefix such as `ablation-dagger-intervention-ratio-YYYYMMDD-HHMMSS`; do not start
a second `ckpt-bkp` session.

### Verify sessions

```bash
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" "tmux ls"
```

Expected for **parallel**: `ablation-0.2`, `ablation-0.5`, `ablation-0.8`,
`ablation-1.0`, `tensorboard`, `ckpt-bkp`.

Expected for **sequential**: `ablation-sweep`, `tensorboard`, `ckpt-bkp`.

### Monitor

| What | Where |
|---|---|
| TensorBoard (local browser) | http://localhost:6006 |
| Attach to parallel run | `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t ablation-0.2'` (or `0.5`, `0.8`, `1.0`) |
| Attach to sequential sweep | `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t ablation-sweep'` |
| Attach to TensorBoard | `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t tensorboard'` |
| Remote logdir | `/workspace/toy-pickplace/runs/flywheel/` |

All four runs share one logdir and overlay in the TensorBoard scalars tab.

---

## Standalone sweep scripts

Save on the instance, then launch with the matching tmux wrapper.

**Parallel** — `/workspace/toy-pickplace/scripts/ablation-dagger-intervention-ratio-parallel.sh`:

```bash
#!/bin/bash
# dagger_intervention_ratio ablation — parallel, one instance
set -euo pipefail
cd /workspace/toy-pickplace

UV=/root/.local/bin/uv
CFG=configs/flywheel/default.yaml

launch() {
  local ratio=$1
  tmux new-session -d -s "ablation-${ratio}" \
    "cd /workspace/toy-pickplace && exec $UV run python scripts/flywheel.py \
      --config $CFG --run-name abl-dagger_intervention_ratio-${ratio} \
      --dagger-intervention-ratio ${ratio}"
}

launch 0.2
launch 0.5
launch 0.8
launch 1.0

echo "Parallel ablation sweep launched."
```

**Sequential** — `/workspace/toy-pickplace/scripts/ablation-dagger-intervention-ratio-sequential.sh`:

```bash
#!/bin/bash
# dagger_intervention_ratio ablation — sequential, one instance
set -euo pipefail
cd /workspace/toy-pickplace

UV=/root/.local/bin/uv
CFG=configs/flywheel/default.yaml

echo "=== Run 1/4: abl-dagger_intervention_ratio-0.2 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-dagger_intervention_ratio-0.2 \
  --dagger-intervention-ratio 0.2

echo "=== Run 2/4: abl-dagger_intervention_ratio-0.5 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-dagger_intervention_ratio-0.5 \
  --dagger-intervention-ratio 0.5

echo "=== Run 3/4: abl-dagger_intervention_ratio-0.8 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-dagger_intervention_ratio-0.8 \
  --dagger-intervention-ratio 0.8

echo "=== Run 4/4: abl-dagger_intervention_ratio-1.0 ==="
$UV run python scripts/flywheel.py \
  --config "$CFG" \
  --run-name abl-dagger_intervention_ratio-1.0 \
  --dagger-intervention-ratio 1.0

echo "Ablation sweep complete."
```

Launch wrappers:

```bash
# parallel
tmux new-session -d -s tensorboard \
  'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'
bash /workspace/toy-pickplace/scripts/ablation-dagger-intervention-ratio-parallel.sh

# sequential
tmux new-session -d -s ablation-sweep \
  'bash /workspace/toy-pickplace/scripts/ablation-dagger-intervention-ratio-sequential.sh'
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

Replace `SESSION` with the backup prefix from the list (e.g. `ablation-dagger-intervention-ratio-20260718-153000`).

```bash
set -a; . ./.env; set +a
mkdir -p runs/flywheel/ablations/dagger_intervention_ratio

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-dagger_intervention_ratio-0.2 \
  runs/flywheel/ablations/dagger_intervention_ratio/abl-dagger_intervention_ratio-0.2

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-dagger_intervention_ratio-0.5 \
  runs/flywheel/ablations/dagger_intervention_ratio/abl-dagger_intervention_ratio-0.5

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-dagger_intervention_ratio-0.8 \
  runs/flywheel/ablations/dagger_intervention_ratio/abl-dagger_intervention_ratio-0.8

uv run python scripts/hf_backup.py \
  --repo skpro19/toy-pickplace-flywheel download \
  SESSION/abl-dagger_intervention_ratio-1.0 \
  runs/flywheel/ablations/dagger_intervention_ratio/abl-dagger_intervention_ratio-1.0
```

### Local TensorBoard (no SSH tunnel)

```bash
uv run python -m tensorboard.main \
  --logdir runs/flywheel/ablations/dagger_intervention_ratio \
  --host 127.0.0.1 --port 6010
```

Open http://localhost:6010 — all four ratios overlaid.

### Re-evaluate best checkpoint (optional)

After picking a winner from TensorBoard / `results/flywheel/*/metrics.json`:

```bash
uv run python scripts/final_score.py --run-name abl-dagger_intervention_ratio-0.8
```

---

## Cleanup

When the sweep is done and backups verified:

```bash
vastai destroy instance "$INSTANCE_ID"
```

Confirm the instance no longer appears in `vastai show instances`.
