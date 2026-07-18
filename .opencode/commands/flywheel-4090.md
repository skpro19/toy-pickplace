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
| `/flywheel-4090 intervention-threshold` | Intervention-threshold ablation (parallel default) | `toy-pickplace-ablation-intervention-threshold` | `ablation-0.05` … `ablation-0.3` | `tb-ablation` |
| `/flywheel-4090 dagger-intervention-ratio` | Dagger-intervention-ratio ablation (parallel default) | `toy-pickplace-ablation-dagger-intervention-ratio` | `ablation-0.2` … `ablation-1.0` | `tb-ablation` |

Requested profile: `$1`

Treat an empty argument as `standard`. Accept only `standard`,
`intervention-threshold`, and `dagger-intervention-ratio`; for any other value,
ask the user to choose a supported profile and stop. Set `INSTANCE_LABEL` to
the label in the table before creating the instance.

For ablation profiles (`intervention-threshold`, `dagger-intervention-ratio`),
this command owns shared provisioning, instance setup, and sweep launch. The
ablation-specific launch commands, TensorBoard setup, backup prefix, local
tunnel, and follow-up commands are defined in the matching ablation document:
- `intervention-threshold` → @docs/ablation/intervention-threshold.md
- `dagger-intervention-ratio` → @docs/ablation/dagger-intervention-ratio.md

**Ablation sweep mode:** launch the **parallel** batch (four concurrent tmux
sessions on one instance) by default. Fall back to the **sequential** batch
only when the user explicitly requests it or the accepted instance has fewer
than 32 effective vCPUs after the hardware gate in Step 6. Do not provision one
instance per swept value in either mode.

This command is the source of truth for provisioning control flow, confirmation
gates, failure handling, and setup commands.

Use @docs/vast-ai/instance-filter-criteria.md as the source of truth for offer
filtering, CPU-family ranking, and post-provision hardware acceptance. Do not
weaken its hard requirements without explicit user approval.

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

Run this search command and parse the raw JSON output.

For `standard`, require `cpu_cores_effective>=24`. For ablation profiles
(`intervention-threshold`, `dagger-intervention-ratio`), require
`cpu_cores_effective>=32` (parallel default). If the user explicitly requested
sequential fallback for the ablation, use `cpu_cores_effective>=24` instead.

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 total_flops>=80 cpu_cores_effective>=MIN_EFFECTIVE_VCPUS cpu_ram>=64 disk_bw>=1000 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
  --order dph_total+ \
  --raw
```

Substitute `MIN_EFFECTIVE_VCPUS` with `32` for ablation profiles unless
sequential fallback was requested, otherwise `24`. For `standard`, use `24`.

Show results as a table with exactly these columns:

| Offer ID | CPU model | Effective vCPUs | RAM | Disk MB/s | PCIe GB/s | GPU power | Down/Up Mb/s | Reliability | $/hr | Location |

Recommend the best offer using this deterministic ordering:
1. **CPU generation**: EPYC 9005, EPYC 9004, Threadripper 7000, then EPYC
   7003. Modern Ryzen 7000/9000 is eligible only when its published physical
   core count is at least 24, but actual allocation still requires the
   post-provision check.
2. **$/hr** (lowest first within the CPU-generation tier).
3. **Disk bandwidth** (higher first when prices tie).
4. **Host reliability** (higher first when the preceding values tie).

Reject EPYC 7001/7002 and CPU models whose published physical-core count is
below 24. Treat raw `cpu_name` as a ranking hint only because offer metadata
can be stale or inconsistent; the SSH acceptance gate is authoritative. Do not
use `cpu_ghz` as a ranking criterion.

If the query returns no offers, show that no candidate meets all hard filters
and ask whether the user wants to wait or explicitly relax named criteria. Do
not silently remove or lower filters. If the user approves a relaxation, state
which criteria changed and preserve every other hard filter.

Show the recommendation with a brief rationale, then **ask the user to confirm**
or select a different offer before proceeding.

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

### 6. Verify provisioned hardware

Treat every new rental as provisional. After SSH becomes ready, run these
checks before cloning the repository, tuning configuration, generating data,
or launching a workload:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" '
  set -e
  echo "=== CPU topology ==="
  lscpu
  echo "=== Physical cores ==="
  lscpu -p=CORE,SOCKET | grep -v "^#" | sort -u | wc -l
  echo "=== Logical CPUs ==="
  nproc
  echo "=== CPU quota ==="
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    cat /sys/fs/cgroup/cpu.max
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  else
    echo "No readable CPU quota file"
  fi
  echo "=== GPU and PCIe ==="
  nvidia-smi --query-gpu=name,memory.total,power.limit,power.default_limit,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
'
```

Parse and show the results as an acceptance table. Compare actual values with
the selected offer and enforce all of these requirements:

| Check | Requirement |
|---|---|
| Physical cores | At least 24 unique `(CORE, SOCKET)` pairs |
| CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
| SMT | Prefer `Thread(s) per core: 1`; call out SMT when physical-core count still passes |
| CPU quota | At least 90% of advertised `cpu_cores_effective` |
| GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
| GPU power | At least 400 W |
| PCIe capability | At least Gen4 x16; measured offer `pcie_bw` at least 20 GB/s |
| Throttling | Hardware thermal and power-brake slowdown both inactive |

For cgroup v2, `cpu.max` contains `QUOTA PERIOD`; `max PERIOD` means no quota.
For cgroup v1, divide `cpu.cfs_quota_us` by `cpu.cfs_period_us`. Values between
90% and 100% of the advertised effective vCPUs pass but must be called out. A
finite quota that is below 90%, or an unreadable quota with no way to establish
the allocation, fails acceptance.

The active PCIe generation may downshift while idle. Do not reject an instance
solely because `pcie.link.gen.current` is below Gen4 at idle when the maximum
link is Gen4 x16 and the offer's measured `pcie_bw` passes. Recheck under CUDA
load if other evidence suggests a restricted link.

If any hard requirement fails, stop before setup, list every failed criterion,
and ask whether to destroy the provisional instance and return to Step 1. Do
not destroy it without confirmation. If confirmed, destroy it, verify it no
longer appears in `vastai show instances`, and obtain a fresh offer snapshot.

The repository does not currently provide a dedicated short workload
acceptance benchmark. Do not claim that workload throughput was validated by
the hardware checks above; use the documented benchmark as a separate manual
gate when one is available.

### 7. Tune config

Apply profile-specific overrides for `configs/flywheel/default.yaml`.

**`standard` profile** — all accepted offers have at least 24 verified physical
cores:

| Physical cores | Recommended `workers` | Recommended `dataloader_workers` | `batch_size` |
|---|---:|---:|---:|
| >= 24 | 12 | 0 | 768 (benchmarked best) |

Benchmark reference (`docs/vast-ai/vast-ai-1.md`):
- `workers: 12` — 12% faster evaluation, 31% faster DAgger collection vs 6
- `batch_size: 768` — 49% faster training than 200, best placement (36%)
- `dataloader_workers: 0` — dataset is in-memory; IPC overhead not justified

**Ablation profiles — parallel (default)** (`intervention-threshold`,
`dagger-intervention-ratio`) — accepted instance has at least 32 effective vCPUs:

| Effective vCPUs | Recommended `workers` | Recommended `dataloader_workers` | `batch_size` |
|---|---:|---:|---:|
| >= 32 | 6 | 2 | 768 |

Benchmark reference (`docs/vast-ai/ablation-study.md`): four concurrent runs on
one RTX 4090; identical checkpoints vs sequential with ~3.8× end-to-end speedup.

**Ablation profiles — sequential (fallback)** — use only when the user requested
sequential or the accepted instance has fewer than 32 effective vCPUs:

| Physical cores | Recommended `workers` | Recommended `dataloader_workers` | `batch_size` |
|---|---:|---:|---:|
| >= 24 | 12 | 0 | 768 |

After Step 6, set `ABLATION_SWEEP_MODE` to `parallel` unless sequential fallback
applies. Record the chosen mode in the final summary.

**Ask the user to confirm** the recommendation or adjust before applying it to
`configs/flywheel/default.yaml`.

**Do NOT edit, commit, push, or copy the local config file** — these are
instance-specific tuning values, not repository changes. Apply the overrides
only on the cloned repository on the instance via SSH after cloning (for
example, `sed -i -E 's/^workers:.*/workers: 12/' ...`). Verify the instance-side
overrides with `grep -E 'workers:|batch_size:|dataloader_workers:'`.

### 8. Setup on the instance

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

Run Batch 3 only for the `standard` profile. For ablation profiles, do not start
the `flywheel` session or this TensorBoard session; after Batches 1 and 2, run
the matching ablation doc's Batch 1 verify step, then:
- **parallel** (`ABLATION_SWEEP_MODE=parallel`): ablation doc parallel Batch 2
- **sequential** (`ABLATION_SWEEP_MODE=sequential`): ablation doc sequential Batch 2

Then complete ablation Batches 3–4 in the profile's ablation document and
return to Step 9 to create `vast-ssh` and `tb-ablation`:
- `intervention-threshold` → @docs/ablation/intervention-threshold.md
- `dagger-intervention-ratio` → @docs/ablation/dagger-intervention-ratio.md

**ckpt-bkp critical details** (based on `scripts/hf_backup.py`):
- `--repo` flag must come **before** the `upload` subcommand, not after  
  - Correct: `uv run python scripts/hf_backup.py --repo ORG/REPO upload --prefix PREFIX`
  - Wrong: `uv run python scripts/hf_backup.py upload --repo ORG/REPO --prefix PREFIX`
- Always pass `--components checkpoints,runs,results`. Do **not** upload the
  `dagger` component: its per-episode `.npz` files (tens of thousands on grid
  runs) exceed the HuggingFace Hub 20,000-file limit and the push is rejected.
- Use the full path `/root/.local/bin/uv` inside tmux sessions started via SSH
  (the tmux session doesn't inherit the SSH login PATH)
- Transfer `HF_TOKEN` over SSH standard input. Never interpolate its value into
  the SSH command string, command arguments, or output. Set it in the remote
  tmux server environment so the backup session inherits it.

For the `standard` profile, start this backup session after Batch 3 with the
timestamp prefix shown below. For ablation profiles, start it only after the
ablation sweep and TensorBoard sessions have been launched, using the
ablation-specific prefix specified in the matching ablation document. Do not
start a second `ckpt-bkp` session.

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
        --repo skpro19/toy-pickplace-flywheel upload --prefix $BACKUP_PREFIX \
        --components checkpoints,runs,results; \
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

### 9. Local tmux wrappers

On the dev machine, parse HOST/PORT from `vastai ssh-url INSTANCE_ID` and create
`vast-ssh` (SSH shell into the instance, using `ServerAliveInterval=30` to
prevent idle disconnects). For the `standard` profile, also create `tb-setup`.
For ablation profiles, create `tb-ablation` from the matching ablation document.

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

### 10. Local download and replay (follow-up)

Do not wait for training or a checkpoint before completing the `standard`
profile. Print these as follow-up commands after flywheel has produced at least
one checkpoint (or after the run completes). For ablation profiles, use the
offline download and replay instructions in the matching ablation document.

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
- For ablation profiles: active profile name, `ABLATION_SWEEP_MODE`
  (`parallel` or `sequential`), and applied `workers` / `dataloader_workers`
  values
- SSH URL retrieval command (`vastai ssh-url INSTANCE_ID`)
- Local attach commands for `vast-ssh` and the profile-specific TensorBoard
  tunnel (`tb-setup` or `tb-ablation`)
- SSH commands that attach directly to the remote workload session(s):
  - `standard`: `flywheel`
  - `intervention-threshold` parallel: `ablation-0.05`, `ablation-0.1`,
    `ablation-0.2`, `ablation-0.3`
  - `intervention-threshold` sequential: `ablation-sweep`
  - `dagger-intervention-ratio` parallel: `ablation-0.2`, `ablation-0.5`,
    `ablation-0.8`, `ablation-1.0`
  - `dagger-intervention-ratio` sequential: `ablation-sweep`
  - plus `tensorboard` and `ckpt-bkp` (for example,
    `ssh -t -p "$PORT" "root@$HOST" 'tmux attach -t flywheel'`)
- TensorBoard URL
- Download command (`uv run python scripts/hf_backup.py --repo REPO download ...`)
- Destroy command for cleanup

## Notes

- Do not store API keys, instance API keys, Jupyter tokens, SSH keys, or transient host/port in any file.
- **RTX 4090 offers are extremely volatile** — they appear and disappear within seconds. Do not check availability before creating; just try `--cancel-unavail` and move to the next offer on failure.
- The rapid-fire loop must **stop after the first successful create** to avoid creating multiple instances. Use a flag variable and `break` carefully.
