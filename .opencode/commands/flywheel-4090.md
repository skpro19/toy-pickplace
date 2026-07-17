---
description: Provision a Vast.ai RTX 4090 instance and start the flywheel pipeline
agent: build
---

Provision and set up a Vast.ai instance to run the flywheel training pipeline.

## Workload profile

The optional first command argument selects the workload profile:

| Invocation | Profile | Instance label | Workload session | Local TensorBoard tunnel |
|---|---|---|---|---|
| `/flywheel-4090` or `/flywheel-4090 standard` | Standard flywheel | `toy-pickplace-flywheel` | `flywheel` | `tb-setup` |
| `/flywheel-4090 intervention-threshold` | Intervention-threshold ablation | `toy-pickplace-ablation-intervention-threshold` | `ablation-sweep` | `tb-ablation` |

Requested profile: `$1`

Treat an empty argument as `standard`. Accept only `standard` and
`intervention-threshold`; for any other value, ask the user to choose a
supported profile and stop. Set `INSTANCE_LABEL` to the label in the table
before creating the instance.

For the `intervention-threshold` profile, this command owns shared
provisioning and instance setup. The ablation-specific sweep, TensorBoard
launch, backup prefix, local tunnel, and follow-up commands are defined in
@docs/ablation/intervention-threshold.md.

This command is the source of truth for provisioning control flow, confirmation
gates, failure handling, and setup commands.

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

Run this search command and parse the output:

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 cpu_cores_effective>=24 rentable=true verification=verified' \
  --order dph_total+
```

Show results as a table with exactly these columns:

| Offer ID | GPU frac | VRAM | Effective vCPUs | CPU GHz | $/hr | Host reliability | Driver | Location |

Recommend the best offer using this deterministic ordering:
1. **CPU tier** (>= 32 effective vCPUs first; otherwise 24-31)
2. **$/hr** (lowest first within the CPU tier)
3. **CPU GHz** (higher first when prices tie)
4. **Host reliability** (higher first when the preceding values tie; call out
   reliability below 99%)

Show the recommendation with a brief rationale, then **ask the user to confirm** or select a different offer before proceeding.

### 2. User confirms priority list

After the user picks a top offer (or a priority-ordered shortlist of 3-5 offers),
note the ordered list. Do not run a separate availability check; offers
disappear within seconds on RTX 4090.

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
OFFER_IDS=(OFFER_1 OFFER_2 OFFER_3) # Include every user-confirmed offer (3-5).
for id in "${OFFER_IDS[@]}"; do
  output=$(vastai create instance "$id" \
    --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
    --disk 100 --ssh --direct --label "$INSTANCE_LABEL" \
    --cancel-unavail 2>&1)
  if echo "$output" | grep -q "new_contract"; then
    INSTANCE_ID=$(echo "$output" | grep -oP "new_contract': \K\d+")
    if [ -n "$INSTANCE_ID" ]; then
      CREATED=true
      break
    fi
  fi
  sleep 1
done
if [ "$CREATED" = false ]; then
  echo "ERROR: Could not create any instance from the priority list"
  exit 1
fi
```

If no instance is created or no instance ID can be parsed, stop the workflow.
Do not poll, clean up, or run setup commands with an empty instance ID.

### 4. Post-create duplicate cleanup

Check `vastai show instances` for other instances with the same label
(`$INSTANCE_LABEL`). If more than one exists (e.g. from a previous
attempt that wasn't cleaned up), ask the user which to keep and destroy the
rest, OR keep the one with the best specs and destroy the others
automatically after listing them.

### 5. Poll for running and SSH readiness

Poll every 10 seconds for at most 30 attempts until `actual_status` is
`"running"`, then get the SSH URL. If the instance reports a terminal failure
or is not running after 30 attempts, print its latest status and stop. Do not
continue to SSH setup.

```bash
for i in $(seq 1 30); do
  status=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null)
  echo "Poll $i: status=$status"
  [ "$status" = "running" ] && break
  sleep 10
done
[ "$status" = "running" ] || {
  echo "ERROR: Instance did not reach running state; latest status=$status" >&2
  exit 1
}
SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(echo "$SSH_URL" | sed 's/.*@//;s/:.*//')
PORT=$(echo "$SSH_URL" | sed 's/.*://')
```

`actual_status=running` does not guarantee that the mapped SSH port is ready.
After resolving the SSH URL, probe SSH every 10 seconds for at most 12 attempts.
Use batch mode so a failed key negotiation cannot block for interactive input.

```bash
SSH_READY=false
for i in $(seq 1 12); do
  echo "SSH probe $i"
  if ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
      -o ConnectTimeout=10 -p "$PORT" "root@$HOST" true 2>/dev/null; then
    SSH_READY=true
    break
  fi
  sleep 10
done

[ "$SSH_READY" = true ] || {
  echo "ERROR: Instance is running but SSH did not become ready" >&2
  exit 1
}
```

Do not run clone, configuration, tmux, or backup commands unless the SSH probe
succeeds.

#### SSH failure recovery

If SSH does not become ready within 12 attempts:

1. Print the instance ID, latest `actual_status`, SSH URL, and the failed probe
   count. Do not print API keys or other instance secrets.
2. Ask the user whether to destroy the unusable instance and retry provisioning.
3. Do not destroy the instance without confirmation.
4. If confirmed, destroy that instance and verify it no longer appears in
   `vastai show instances`.
5. Return to **Step 1: Search offers** and obtain a fresh offer snapshot. RTX
   4090 offers from the previous priority list may already be stale, so ask the
   user to confirm the new priority list before creating another instance.
6. If the user declines destruction or retry, stop the workflow. Do not attempt
   setup commands against the failed instance.

Do not change the image, add SSH installation commands, reboot repeatedly, or
otherwise modify the provisioning command as an SSH workaround. A replacement
host using the original create command is the recovery path.

### 6. Tune config

All accepted offers have at least 24 effective vCPUs. Recommend these
overrides for `configs/flywheel/default.yaml`:

| Effective vCPUs | Recommended `workers` | Recommended `dataloader_workers` | `batch_size` |
|---|---:|---:|---:|
| >= 24 | 12 | 0 | 768 (benchmarked best) |

Benchmark reference (`docs/vast-ai/vast-ai-1.md`):
- `workers: 12` — 12% faster evaluation, 31% faster DAgger collection vs 6
- `batch_size: 768` — 49% faster training than 200, best placement (36%)
- `dataloader_workers: 0` — dataset is in-memory; IPC overhead not justified

**Ask the user to confirm** the recommendation or adjust before applying it to
`configs/flywheel/default.yaml`.

**Do NOT edit, commit, push, or copy the local config file** — these are
instance-specific tuning values, not repository changes. Apply the overrides
only on the cloned repository on the instance via SSH after cloning (for
example, `sed -i -E 's/^workers:.*/workers: 12/' ...`). Verify the instance-side
overrides with `grep -E 'workers:|batch_size:|dataloader_workers:'`.

### 7. Setup on the instance

Use the SSH URL from `vastai ssh-url INSTANCE_ID` (host and port may differ from
the create output). Break the setup into batches to avoid overly long SSH
commands. Substitute the user-approved numeric values for all `CONFIRMED_*`
placeholders before execution.

```bash
# Batch 1: clone, uv, CUDA verify
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
  "git clone --branch dev --single-branch \
     https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace && \
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   /root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace && \
   cd /workspace/toy-pickplace && \
   /root/.local/bin/uv run python -c \
     'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'"

# Batch 2: tmux config and confirmed config overrides
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
  "set -e; \
   touch ~/.no_auto_tmux; \
   printf '%s\n' 'set -g mouse on' > ~/.tmux.conf; \
   cd /workspace/toy-pickplace; \
   sed -i -E \
     -e 's/^workers:.*/workers: CONFIRMED_WORKERS/' \
     -e 's/^batch_size:.*/batch_size: CONFIRMED_BATCH_SIZE/' \
     -e 's/^dataloader_workers:.*/dataloader_workers: CONFIRMED_DATALOADER_WORKERS/' \
     configs/flywheel/default.yaml; \
   grep -E '^(workers|batch_size|dataloader_workers):' \
     configs/flywheel/default.yaml"

# Batch 3: standard profile only -- launch flywheel and TensorBoard
ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s flywheel \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py --config configs/flywheel/default.yaml' && \
   tmux new-session -d -s tensorboard \
      'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

Run Batch 3 only for the `standard` profile. For `intervention-threshold`, do
not start the `flywheel` session or this TensorBoard session; after Batches 1
and 2, complete its Batches 1-4 in
@docs/ablation/intervention-threshold.md.
Then return to Step 8 and create `vast-ssh` only.

**ckpt-bkp critical details** (based on `scripts/hf_backup.py`):
- `--repo` flag must come **before** the `upload` subcommand, not after  
  - Correct: `uv run python scripts/hf_backup.py --repo ORG/REPO upload --prefix PREFIX`
  - Wrong: `uv run python scripts/hf_backup.py upload --repo ORG/REPO --prefix PREFIX`
- Use the full path `/root/.local/bin/uv` inside tmux sessions started via SSH
  (the tmux session doesn't inherit the SSH login PATH)
- Transfer `HF_TOKEN` over SSH standard input. Never interpolate its value into
  the SSH command string, command arguments, or output. Set it in the remote
  tmux server environment so the backup session inherits it.

For the `standard` profile, start this backup session after Batch 3 with the
timestamp prefix shown below. For `intervention-threshold`, start it only after
the ablation sweep and TensorBoard sessions have been launched, using the
ablation-specific prefix specified in the ablation document. Do not start a
second `ckpt-bkp` session.

Example:
```bash
set -a
. ./.env
set +a
BACKUP_PREFIX=$(date +%Y%m%d-%H%M%S)
printf '%s\n' "$HF_TOKEN" | \
  ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" "
  IFS= read -r HF_TOKEN
  [ -n \"\$HF_TOKEN\" ] || { echo 'ERROR: HF_TOKEN transfer failed' >&2; exit 1; }
  tmux set-environment -g HF_TOKEN \"\$HF_TOKEN\"
  tmux new-session -d -s ckpt-bkp \
    'cd /workspace/toy-pickplace && while true; do \
      /root/.local/bin/uv run python scripts/hf_backup.py \
        --repo skpro19/toy-pickplace-flywheel upload --prefix $BACKUP_PREFIX; \
      sleep 120; \
    done'
"
```

Do not enable shell tracing while handling secrets. Verify that neither the
token nor the contents of `.env` appear in command output.

Verify the backup is working:
`ssh -p "$PORT" "root@$HOST" "tmux capture-pane -t ckpt-bkp -p -S -10"`.
Verify the profile-specific instance-side sessions exist with:
`ssh -p "$PORT" "root@$HOST" "tmux ls"`.

### 8. Local tmux wrappers

On the dev machine, parse HOST/PORT from `vastai ssh-url INSTANCE_ID` and create
`vast-ssh` (SSH shell into the instance, using `ServerAliveInterval=30` to
prevent idle disconnects). For the `standard` profile, also create `tb-setup`.
For `intervention-threshold`, create `tb-ablation` from the ablation document.

```bash
SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
HOST=$(echo "$SSH_URL" | sed 's/.*@//;s/:.*//')
PORT=$(echo "$SSH_URL" | sed 's/.*://')

tmux new-session -d -s vast-ssh \
  "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p $PORT root@$HOST"
tmux new-session -d -s tb-setup \
  "ssh -N -L 6006:127.0.0.1:6006 -p $PORT root@$HOST"
```

Run the `tb-setup` command only for the `standard` profile.

### 9. Local download and replay (follow-up)

Do not wait for training or a checkpoint before completing the `standard`
profile. Print these as follow-up commands after flywheel has produced at least
one checkpoint (or after the run completes). For `intervention-threshold`, use
the offline download and replay instructions in the ablation document.

```bash
# List available sessions in the HF repo
set -a; . ./.env; set +a
uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel list

# Download a specific run from the latest session
uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel download 20260717-153000/run-003

# View TensorBoard locally (no SSH tunnel needed)
tensorboard --logdir runs/flywheel/

# Re-evaluate on held-out seeds
uv run python scripts/final_score.py --run-name run-003
```

## Final output

Provisioning is complete after the profile-specific instance-side sessions and
local tmux wrappers have been verified. Then print a summary with:
- Instance ID
- SSH URL retrieval command (`vastai ssh-url INSTANCE_ID`)
- Local attach commands for `vast-ssh` and the profile-specific TensorBoard
  tunnel (`tb-setup` or `tb-ablation`)
- SSH commands that attach directly to the remote workload session (`flywheel`
  or `ablation-sweep`), `tensorboard`, and `ckpt-bkp` (for example,
  `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t flywheel'`)
- TensorBoard URL
- Download command (`uv run python scripts/hf_backup.py --repo REPO download ...`)
- Destroy command for cleanup

## Notes

- Do not store API keys, instance API keys, Jupyter tokens, SSH keys, or transient host/port in any file.
- **RTX 4090 offers are extremely volatile** — they appear and disappear within seconds. Do not check availability before creating; just try `--cancel-unavail` and move to the next offer on failure.
- The rapid-fire loop must **stop after the first successful create** to avoid creating multiple instances. Use a flag variable and `break` carefully.
