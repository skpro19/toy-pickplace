---
description: Provision a Vast.ai RTX 4090 instance and start the flywheel pipeline
agent: general
---

Provision and set up a Vast.ai instance to run the flywheel training pipeline.

Read `docs/vast-ai/vast-ai-2.md` for the full runbook. Follow the "Provisioning workflow" section step by step.

## Prerequisites

Before starting, ensure these are available on the dev machine:

| Tool / key | Check |
|---|---|
| `vastai` CLI | `which vastai` |
| `VAST_API_KEY` | Set in `.env` file (copy `.env.example` → `.env` if missing) |
| `HF_TOKEN` | Set in `.env` file |
| `tmux` | `which tmux` |

The agent will source `.env` at the start and abort if any key is missing.
Instructions will be printed for missing items.

## Workflow

### 0. Load secrets

Source `.env` and verify both `VAST_API_KEY` and `HF_TOKEN` are non-empty.
If either is missing, print instructions pointing to `.env.example` and stop.

### 1. Search offers

Run the search command from the doc. Parse the output.

Show results as a table with exactly these columns:

| Offer ID | GPU frac | VRAM | Effective vCPUs | CPU GHz | $/hr | Host reliability | Driver | Location |

Recommend the best offer based on this priority:
1. **$/hr** (lowest first)
2. **Effective vCPUs** (>= 32 preferred)
3. **CPU GHz** (higher is better)
4. **Host reliability** (>= 99% preferred)

Show the recommendation with a brief rationale, then **ask the user to confirm** or select a different offer before proceeding.

### 2. User confirms priority list

After the user picks a top offer (or a priority-ordered shortlist of 3-5 offers),
note the ordered list. Do NOT check availability yet — offers disappear within
seconds on RTX 4090.

### 3. Rapid-fire create

Iterate the priority list. For each offer, call `vastai create instance` directly
with `--cancel-unavail`. If the offer is gone, the API returns an error — move
to the next in the list. **Stop at the first successful create.** Do not
continue iterating after a success (use a flag or `break`). Record the returned
instance ID.

```bash
# IMPORTANT: The output is NOT valid JSON — it's a mix of "Started." prefix
# and a Python dict literal. grep for 'new_contract' key as plain text.
CREATED=false
for id in OFFER_1 OFFER_2 OFFER_3; do
  output=$(vastai create instance "$id" --image ... --disk 100 --ssh --direct --label toy-pickplace-flywheel --cancel-unavail 2>&1)
  if echo "$output" | grep -q "new_contract"; then
    INSTANCE_ID=$(echo "$output" | grep -oP "new_contract': \K\d+")
    CREATED=true
    break
  fi
  sleep 1
done
if [ "$CREATED" = false ]; then
  echo "ERROR: Could not create any instance from the priority list"
fi
```

### 4. Post-create duplicate cleanup

Check `vastai show instances` for other instances with the same label
(`toy-pickplace-flywheel`). If more than one exists (e.g. from a previous
attempt that wasn't cleaned up), ask the user which to keep and destroy the
rest, OR keep the one with the best specs and destroy the others
automatically after listing them.

### 5. Poll for running

Poll every 10 seconds until `actual_status` is `"running"`, then get the SSH URL.

### 6. Tune config

Based on the provisioned instance's effective vCPUs, recommend these overrides
for `configs/flywheel/default.yaml`:

| Effective vCPUs | Recommended `workers` | Recommended `dataloader_workers` | `batch_size` (optional) |
|---|---:|---:|---:|
| >= 24 | 12 | 0 | 768 (benchmarked best) |
| 16-23 | 6 | 0 | 768 (benchmarked best) |

Benchmark reference (`docs/vast-ai/vast-ai-1.md`):
- `workers: 12` — 12% faster evaluation, 31% faster DAgger collection vs 6
- `batch_size: 768` — 49% faster training than 200, best placement (36%)
- `dataloader_workers: 0` — dataset is in-memory; IPC overhead not justified

**Ask the user to confirm** the recommendation or adjust before applying it to
`configs/flywheel/default.yaml`.

**Do NOT commit or push the config changes** — they are instance-specific
tuning, not part of the repo. Instead, apply the edits locally and copy the
modified file to the instance via `scp`, or apply the overrides on the instance
via SSH after cloning (e.g. `sed -i 's/workers: 6/workers: 12/' ...`). Verify
the overrides with `grep -E 'workers:|batch_size:|dataloader_workers:'`.

### 7. Setup on the instance

SSH in and run all commands from the doc's "Setup on the instance" section in order.
Use the SSH URL from `vastai ssh-url INSTANCE_ID` (host and port may differ from
the create output). Break the setup into batches to avoid overly long SSH
commands:

```bash
# Batch 1: clone, uv, CUDA verify
ssh -o StrictHostKeyChecking=no -p PORT root@HOST \
  "git clone ... && curl ... && uv sync ... && python -c 'import torch; ...'"

# Batch 2: tmux config, expert data, config overrides
ssh -o StrictHostKeyChecking=no -p PORT root@HOST \
  "touch ~/.no_auto_tmux && ..."

# Batch 3: launch flywheel, tensorboard, ckpt-bkp in tmux
ssh -o StrictHostKeyChecking=no -p PORT root@HOST \
  "tmux new-session -d -s flywheel '...' && tmux new-session -d -s tensorboard '...' && tmux new-session -d -s ckpt-bkp '...'"
```

Verify all three tmux sessions exist: `ssh ... "tmux ls"`

Commands to run on the instance:
- Clone the repo (`dev` branch)
- Install uv
- `uv sync --locked`
- Verify CUDA (`torch.cuda.is_available()`)
- Disable auto-tmux (`touch ~/.no_auto_tmux`)
- Enable tmux mouse
- Generate 100 expert episodes
- Launch flywheel in tmux:`flywheel`  
- Launch TensorBoard in tmux:`tensorboard`  
- Launch HF backup in tmux:`ckpt-bkp`  

**ckpt-bkp critical details** (based on `scripts/hf_backup.py`):
- `--repo` flag must come **before** the `upload` subcommand, not after  
  - Correct: `uv run python scripts/hf_backup.py --repo ORG/REPO upload --prefix PREFIX`
  - Wrong: `uv run python scripts/hf_backup.py upload --repo ORG/REPO --prefix PREFIX`
- Use the full path `/root/.local/bin/uv` inside tmux sessions started via SSH
  (the tmux session doesn't inherit the SSH login PATH)
- Export `HF_TOKEN` inside the tmux command string so the backup script can
  authenticate. Source it from the dev machine's `.env` and pass it through SSH.

Example:
```bash
BACKUP_PREFIX=$(date +%Y%m%d-%H%M%S)
source .env && ssh ... \
  "tmux new-session -d -s ckpt-bkp \"cd /workspace/toy-pickplace && \
    export HF_TOKEN=$HF_TOKEN && \
    while true; do \
      /root/.local/bin/uv run python scripts/hf_backup.py \
        --repo skpro19/toy-pickplace-flywheel upload --prefix \$BACKUP_PREFIX; \
      sleep 120; \
    done\""
```

Verify the backup is working: `ssh ... "tmux capture-pane -t ckpt-bkp -p -S -10"`

### 8. Local tmux wrappers

On the dev machine, parse HOST/PORT from `vastai ssh-url INSTANCE_ID` and create:
- tmux:`vast-ssh` (SSH shell into the instance, use `ServerAliveInterval=30` to prevent idle disconnects)
- tmux:`tb-setup` (TensorBoard port tunnel)

### 9. Local download and replay

After flywheel has produced at least one checkpoint (or after the run completes):

```bash
# List available sessions in the HF repo
HF_TOKEN=<token> uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel list

# Download a specific run from the latest session
uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel download 20260717-153000/run-003

# View TensorBoard locally (no SSH tunnel needed)
tensorboard --logdir runs/flywheel/

# Re-evaluate on held-out seeds
uv run python scripts/final_score.py --run-name run-003
```

## Final output

When complete, print a summary with:
- Instance ID
- SSH URL retrieval command (`vastai ssh-url INSTANCE_ID`)
- Attach commands for all 5 tmux sessions (`vast-ssh`, `tb-setup`, `flywheel`, `tensorboard`, `ckpt-bkp`)
- TensorBoard URL
- Download command (`uv run python scripts/hf_backup.py download ...`)
- Destroy command for cleanup

## Notes

- Do not store API keys, instance API keys, Jupyter tokens, SSH keys, or transient host/port in any file.
- **RTX 4090 offers are extremely volatile** — they appear and disappear within seconds. Do not check availability before creating; just try `--cancel-unavail` and move to the next offer on failure.
- The rapid-fire loop must **stop after the first successful create** to avoid creating multiple instances. Use a flag variable and `break` carefully.
- Do NOT edit `docs/vast-ai/vast-ai-2.md` — that file is a manual run log.
