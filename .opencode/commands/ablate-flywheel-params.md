---
description: Provision one RTX 4090 and run a parameterized flywheel ablation grid
agent: build
---

Run a general-purpose flywheel ablation on one Vast.ai RTX 4090. This command
owns argument validation, grid planning, provisioning, hardware acceptance,
instance tuning, batched launch, checkpoint backup, and local monitoring.

Use @.opencode/commands/flywheel-4090.md for the unchanged detailed
provisioning mechanics. This command is the source of truth for its grid
routing, higher CPU gates, run naming, and launch batching.

## Command arguments and grid planning

Requested parameter names are `$1 $2 ... $N`. At least one name is required.

1. Accept a parameter name in its YAML `snake_case` form or CLI `kebab-case`
   form. Normalize CLI names by replacing `-` with `_`.
2. Accept only parameters with a matching `scripts/flywheel.py` CLI argument.
   Reject infrastructure-only values (`workers`, `dataloader_workers`) and
   reject duplicate normalized names. Show the supported parameter catalog and
   stop for an invalid request.
3. Ask for an exact comma- or newline-separated value list for every requested
   parameter. Display its default from `configs/flywheel/default.yaml`, its CLI
   flag, and the corresponding argument validation constraints before asking.
4. Let `GRID_CELLS` be the product of all confirmed value-list lengths:
   - one requested parameter is a 1D sweep;
   - two or more requested parameters form a full Cartesian grid, with every
     requested CLI override set on every cell.
5. Require `1 <= GRID_CELLS <= 30`. If the grid has more than 30 cells, show
   its computed size and ask the user to reduce the lists or split the work into
   multiple invocations. Do not provision.
6. Build `RUN_PLAN` in deterministic parameter-list order and value-list order.
   Each row includes:
   - a unique run name;
   - a unique tmux session;
   - all CLI overrides for that cell;
   - its batch number after Step 6 determines concurrency.
7. Show the complete `RUN_PLAN`, selected tier, search floor, physical-core
   acceptance gate, tuning values, planned concurrency range, and estimated
   batch count. Ask the user to confirm the entire plan before Step 0.

### Supported parameter catalog

Use the live parser in `scripts/flywheel.py` as authoritative. The common
ablation parameters are:

| YAML key | CLI flag | Default | Constraint / suggested values |
|---|---|---:|---|
| `intervention_threshold` | `--intervention-threshold` | `0.1` | `>=0`; suggested `0.05, 0.1, 0.2, 0.3` |
| `dagger_intervention_ratio` | `--dagger-intervention-ratio` | `0.8` | `0..1`; suggested `0.2, 0.5, 0.8, 1.0` |
| `expert_ratio` | `--expert-ratio` | `0.5` | `0 < value < 1`; suggested `0.3, 0.5, 0.7` |
| `dagger_rounds` | `--dagger-rounds` | `10` | integer `>=1`; suggested `5, 10, 20` |
| `intervention_steps` | `--intervention-steps` | `50` | integer `>=1`; suggested `25, 50, 100` |
| `dagger_episodes` | `--dagger-episodes` | `50` | integer `>=1`; suggested `25, 50, 100` |
| `batch_size` | `--batch-size` | `200` | integer `>=1`; suggested `200, 384, 768` |
| `early_stop_patience` | `--early-stop-patience` | `50` | integer `>=0`; suggested `25, 50, 100` |
| `mode` | `--mode` | `mode-b` | `mode-a` or `mode-b` |
| `eval_interval` | `--eval-interval` | `20` | integer `>=1`; suggested `10, 20, 40` |
| `eval_episodes` | `--eval-episodes` | `25` | integer `>=1`; suggested `25, 50` |

For additional parser-supported experiment parameters, validate against
`flywheel.py` and apply the same value validation it enforces. Do not permit
`workers` or `dataloader_workers` as swept dimensions; those are selected by
the tier below.

### Naming

For a 1D sweep, use:

```text
abl-{parameter_key}-{value_slug}
```

For a grid, use:

```text
abl-{parameter_key_1}-{value_slug_1}-{parameter_key_2}-{value_slug_2}-...
```

Convert a decimal point in each `value_slug` to `p` and replace other
non-alphanumeric separators with `-`. For general 1D and grid runs, use the
same slug in the tmux name, prefixed with `ablation-`.

Preserve the existing one-dimensional document conventions for the two legacy
profiles so their attach commands remain valid:

| Parameter | Run name | tmux session |
|---|---|---|
| `intervention_threshold` | `abl-intervention_threshold-{value}` | `ablation-{value}` |
| `dagger_intervention_ratio` | `abl-dagger_intervention_ratio-{value}` | `ablation-{value}` |

The command may use the concise `abl-it{T}-dir{R}` / `ablation-it{T}-dir{R}`
form only when the two parameters are exactly `intervention_threshold` and
`dagger_intervention_ratio`; it must still be unique.

Set:

```text
INSTANCE_LABEL=toy-pickplace-ablation-{parameter-slugs}
GLOBAL_SEED=<global_seed resolved from the selected flywheel config>
BACKUP_PREFIX=ablation-{parameter-slugs}-seed{GLOBAL_SEED}-YYYYMMDD-HHMMSS
```

Include the resolved `GLOBAL_SEED` in the displayed run plan and use the same
value in every grid-cell command.

## Tier routing

Select exactly one tier from the confirmed `GRID_CELLS`. The tier determines
the search `cpu_cores_effective` minimum, the post-provision physical-core
acceptance gate, parallelism limits, and instance-only tuning.

| Tier | Grid cells | Search effective vCPU minimum | Prefer effective vCPUs | SSH physical-core minimum | Maximum concurrent cells | `workers` / `dataloader_workers` |
|---|---:|---:|---:|---:|---:|---:|
| S | 1 | 24 | — | 24 | 1 | 12 / 0 |
| M | 2–4 | 32 | — | 24 | 4 | 6 / 2 |
| L | 5–8 | 48 | 64 | 48 | 8 | 6 / 2 |
| XL | 9–16 | 64 | 128 | 64 | 8 | 6 / 2 |
| XXL | 17–30 | 64 | 128 | 64 | 16 | 6 / 2 |

For all tiers, set `batch_size=768` only when it is not itself being swept.
When `batch_size` is a swept parameter, do not overwrite it in the
instance-side YAML; each cell supplies `--batch-size` itself.

For tiers L, XL, and XXL, reserve CPU headroom for simulator coordination,
training, TensorBoard, shell overhead, and cgroup scheduling:

```text
CPU_RESERVE = max(8, ceil(PHYSICAL_CORES * 0.20))
CPU_CAP = floor((PHYSICAL_CORES - CPU_RESERVE) / WORKERS)
ABLATION_CONCURRENCY = min(TIER_MAX_CONCURRENCY, CPU_CAP, GRID_CELLS)
```

For tiers S and M, use:

```text
CPU_CAP = floor(PHYSICAL_CORES / WORKERS)
ABLATION_CONCURRENCY = min(TIER_MAX_CONCURRENCY, CPU_CAP, GRID_CELLS)
```

If the accepted host cannot support at least one concurrent cell after this
calculation, reject it. Never exceed `ABLATION_CONCURRENCY`, and never launch
all grid cells together unless the grid itself is no larger than that value.

Examples:

| Verified physical cores | XXL CPU reserve | `CPU_CAP` at `workers=6` | XXL concurrency | 30-cell batches |
|---:|---:|---:|---:|---:|
| 64 | 13 | 8 | 8 | 4 |
| 96 | 20 | 12 | 12 | 3 |
| 128 | 26 | 17 | 16 (tier cap) | 2 |

## Prerequisites

Before starting, ensure these are available on the development machine:

| Tool / key | Check |
|---|---|
| `vastai` CLI | `which vastai` |
| `VAST_API_KEY` | Set in `.env` |
| `HF_TOKEN` | Set in `.env` |
| `tmux` | `which tmux` |
| `flock` | `which flock` |

Source `.env` at the start and abort if either token is missing. Point to
`.env.example`; never print secret values or enable shell tracing.

## Workflow

### 0. Load secrets

Follow `flywheel-4090.md` Step 0 exactly.

### 1. Search offers

Use the selected tier's `SEARCH_EFFECTIVE_VCPUS` in this query:

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 total_flops>=80 cpu_cores_effective>=SEARCH_EFFECTIVE_VCPUS cpu_ram>=64 disk_bw>=1000 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
  --order dph_total+ \
  --raw
```

Use @docs/vast-ai/instance-filter-criteria.md as the source of truth for all
non-CPU hard filters, raw-offer interpretation, GPU/PCIe requirements, and
post-provision verification.

Show exactly these columns:

| Offer ID | CPU model | Effective vCPUs | RAM | Disk MB/s | PCIe GB/s | GPU power | Down/Up Mb/s | Reliability | $/hr | Location |

Rank eligible offers by:

1. CPU family: EPYC 9005, EPYC 9004, Threadripper 7000, EPYC 7003, modern
   Ryzen 7000/9000;
2. offers at or above the tier's preferred effective-vCPU count, when present;
3. lowest $/hr;
4. highest disk bandwidth;
5. highest reliability.

Reject EPYC 7001/7002 and models with a published physical-core count below
the tier's SSH physical-core minimum. Treat `cpu_name` only as a ranking hint:
the SSH topology gate is authoritative. Do not use `cpu_ghz` as a criterion.

If no offer satisfies the hard filters, show that outcome and ask whether to
wait or explicitly relax named criteria. Do not silently lower filters. Show a
recommendation, rationale, the selected tier, and ask the user to confirm an
offer or an ordered shortlist of 3–5 offers.

### 2. User confirms priority list

Follow `flywheel-4090.md` Step 2 exactly. Do not run a separate availability
check; RTX 4090 offers are volatile.

### 3. Rapid-fire create

Follow `flywheel-4090.md` Step 3 exactly, using the derived `INSTANCE_LABEL`.
Use the same image, disk, SSH/direct options, `--cancel-unavail`, and
first-success-only loop. Do not proceed without a parsed instance ID.

### 4. Post-create duplicate cleanup

Follow `flywheel-4090.md` Step 4 exactly, using the derived `INSTANCE_LABEL`.

### 5. Poll for running and SSH readiness

Follow `flywheel-4090.md` Step 5 exactly, including:

- up to 30 status polls at 10-second intervals;
- up to 12 batch-mode SSH probes at 10-second intervals;
- the same SSH failure recovery flow;
- fresh offer search and user confirmation after a confirmed retry.

Do not clone, tune, launch tmux, or start backup until SSH is ready.

### 6. Verify provisioned hardware and select concurrency

Run the complete hardware inspection command from `flywheel-4090.md` Step 6,
including `lscpu`, physical-core counting, cgroup quota, and detailed
`nvidia-smi` output. Parse it into an acceptance table.

Apply every base requirement from `flywheel-4090.md` Step 6, except replace
its physical-core minimum with the selected tier's `SSH physical-core minimum`.

| Check | Requirement |
|---|---|
| Physical cores | At least `TIER_PHYSICAL_CORE_MIN` unique `(CORE, SOCKET)` pairs |
| CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
| SMT | Prefer one thread per core; passing physical count is mandatory |
| CPU quota | At least 90% of advertised effective vCPUs |
| GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
| GPU power | At least 400 W |
| PCIe | Gen4 x16 capability and offer `pcie_bw >= 20` GB/s |
| Throttling | Thermal and power-brake slowdown inactive |

For `GRID_CELLS=1`, the physical minimum is 24. For 2–4 cells it remains 24.
For 5–8 cells it is 48. For 9–30 cells it is 64. A host that passes the
general 24-core criterion but fails the selected tier's criterion is rejected.

On any failure, stop before setup, list every failed criterion, and ask whether
to destroy the provisional instance and return to Step 1. Do not destroy
without confirmation. If confirmed, destroy the instance, verify it no longer
appears in `vastai show instances`, obtain a fresh offer snapshot, and obtain a
fresh user-confirmed priority list before retrying.

After acceptance, calculate and display `CPU_RESERVE`, `CPU_CAP`,
`ABLATION_CONCURRENCY`, and `BATCH_COUNT`. Ask the user to confirm this
computed launch plan and the selected `workers`, `dataloader_workers`, and
batch-size policy before applying tuning.

### 7. Tune the instance-side config

Do not edit, commit, push, or copy the local config. Apply values only to the
cloned repository after Step 8 Batch 1.

Use the selected tier's `workers` and `dataloader_workers`. Use `batch_size=768`
unless `batch_size` is swept, in which case preserve the YAML baseline and let
each run use its CLI override. Verify with `grep`.

### 8. Setup and run batches on the instance

Use the SSH URL from `vastai ssh-url "$INSTANCE_ID"`. Break setup into these
batches.

#### Batch 1 — clone, uv, CUDA verification

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "git clone --branch dev --single-branch \
     https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace && \
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   /root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace && \
   cd /workspace/toy-pickplace && \
   /root/.local/bin/uv run python -c \
     'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'"
```

#### Batch 2 — tmux configuration and tier tuning

Substitute confirmed `TIER_WORKERS`, `TIER_DATALOADER_WORKERS`, and either the
literal `768` or no batch-size substitution when it is swept.

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "set -e; \
   touch ~/.no_auto_tmux; \
   printf '%s\n' 'set -g mouse on' > ~/.tmux.conf; \
   cd /workspace/toy-pickplace; \
   sed -i -E \
     -e 's/^workers:.*/workers: TIER_WORKERS/' \
     -e 's/^dataloader_workers:.*/dataloader_workers: TIER_DATALOADER_WORKERS/' \
     configs/flywheel/default.yaml; \
   grep -E '^(workers|batch_size|dataloader_workers):' \
     configs/flywheel/default.yaml"
```

When `batch_size` is not swept, include:

```bash
-e 's/^batch_size:.*/batch_size: 768/'
```

in the `sed` expression.

#### Batch 3 — start and verify TensorBoard, then create local wrappers

Start TensorBoard once before grid batch 1:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s tensorboard \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main \
      --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

Verify TensorBoard before launching a grid cell. Poll its remote endpoint for
up to 12 attempts at five-second intervals; on failure, print its last 20
lines and stop:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 12); do
     tmux has-session -t tensorboard 2>/dev/null &&
       curl -fsS http://127.0.0.1:6006/ >/dev/null && exit 0
     sleep 5
   done
   tmux capture-pane -t tensorboard -p -S -20 2>/dev/null || true
   exit 1"
```

Immediately create and verify the local wrappers. This is deliberately before
the first grid batch so TensorBoard is available while training is running.
Immediately before creating them, automatically select the lowest unused
non-negative `LOCAL_WORKFLOW_INDEX`. Set
`LOCAL_TB_PORT=6006 + LOCAL_WORKFLOW_INDEX`,
`LOCAL_SSH_SESSION=vast-ssh-LOCAL_WORKFLOW_INDEX`, and
`LOCAL_TB_SESSION=tb-ablation-LOCAL_WORKFLOW_INDEX`. Hold a local lock while
selecting and creating the wrappers so concurrently starting workflows cannot
select the same index. Do not ask the user for an index:

```bash
set -e
exec 9>/tmp/toy-pickplace-ablation-local-wrapper.lock
flock 9
LOCAL_WORKFLOW_INDEX=""
for index in $(seq 0 999); do
  candidate_ssh_session="vast-ssh-$index"
  candidate_tb_session="tb-ablation-$index"
  candidate_tb_port=$((6006 + index))
  if tmux has-session -t "$candidate_ssh_session" 2>/dev/null ||
    tmux has-session -t "$candidate_tb_session" 2>/dev/null ||
    ss -ltn "sport = :$candidate_tb_port" | grep -q LISTEN; then
    continue
  fi
  LOCAL_WORKFLOW_INDEX=$index
  LOCAL_SSH_SESSION=$candidate_ssh_session
  LOCAL_TB_SESSION=$candidate_tb_session
  LOCAL_TB_PORT=$candidate_tb_port
  break
done
test -n "$LOCAL_WORKFLOW_INDEX"
tmux new-session -d -s "$LOCAL_SSH_SESSION" \
  "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p $PORT root@$HOST"
tmux new-session -d -s "$LOCAL_TB_SESSION" \
  "ssh -N -L $LOCAL_TB_PORT:127.0.0.1:6006 -p $PORT root@$HOST"
tmux has-session -t "$LOCAL_SSH_SESSION" && tmux has-session -t "$LOCAL_TB_SESSION"
```

Print `http://localhost:$LOCAL_TB_PORT` now. If either wrapper cannot be
created, stop before the grid launch and report the relevant local tmux output.

#### Batch 4 — materialize a durable remote controller

The agent must not remain responsible for waiting for long-running cells.
On the development machine, set and retain the absolute remote control path
before launching the grid:

```text
CONTROL_DIR="/workspace/toy-pickplace/.ablation/${BACKUP_PREFIX}"
CONTROL_DIR/cells/
CONTROL_DIR/logs/
CONTROL_DIR/status/
CONTROL_DIR/state/
```

Materialize the confirmed `RUN_PLAN` there. Use a deterministic `plan` file
with one `batch|run_name|tmux_session` row per cell. For every row, create one
shell-safe executable `cells/{tmux_session}.sh` containing its full
`/root/.local/bin/uv run python scripts/flywheel.py` command, with
`--config`, `--run-name`, and every swept override. Do not commit or copy
these transient files back to the local repository.

Each cell wrapper must:

1. write `running` atomically to `status/{run_name}`;
2. `cd /workspace/toy-pickplace`, then stream stdout and stderr to both its
   tmux pane and `logs/{run_name}.log` with `tee -a`;
3. write `succeeded 0` or `failed EXIT_CODE` atomically to
   `status/{run_name}` before exiting with that same code.

Use a temporary file plus `mv` for each status update. A missing tmux session
is never a success signal; the terminal status file is authoritative. Set
`pipefail` and record `${PIPESTATUS[0]}` after the `tee` pipeline so a
successful `tee` never masks a failed flywheel command. The persisted log is
the durable record; pane output is a live convenience only.

Create an executable `controller.sh` in `CONTROL_DIR` and start it once in a
remote `ablation-controller` tmux session. The controller must:

1. read the materialized plan in deterministic order;
2. launch at most `ABLATION_CONCURRENCY` `ablation-*` sessions for a batch;
3. atomically write `state/batch-N-started` after those sessions are launched;
4. wait for a terminal status file for every cell in that batch;
5. treat a `running` cell whose tmux session disappears before writing a
   terminal status as `failed 125`;
6. write `state/batch-N-succeeded` and launch the next batch only if every
   status is `succeeded 0`;
7. on the first failure, write `state/failed` with the run name, exit code,
   and log path, then exit non-zero without launching later batches;
8. after the final successful batch, write `state/completed`.

`ablation-controller` is the durable batch scheduler. It may run for days
after the agent disconnects or times out. It must not auto-retry a failed cell
or continue after a failure: the user decides whether to retry, skip, or stop
when the agent later reads `state/failed`.

Start it with:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s ablation-controller \
     'cd '$CONTROL_DIR' && exec bash ./controller.sh'"
```

Verify `ablation-controller` exists and wait only until
`state/batch-1-started` appears:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 12); do
     test -f '$CONTROL_DIR/state/batch-1-started' && exit 0
     tmux has-session -t ablation-controller 2>/dev/null || exit 1
     sleep 5
   done
   tmux capture-pane -t ablation-controller -p -S -20 2>/dev/null || true
   exit 1"
```

Confirm that the listed first-batch cell sessions and `tensorboard` are
present with `tmux ls`. If the controller exits before this marker, capture its
pane and stop.

#### Batch 5 — start checkpoint backup once

After `state/batch-1-started` exists, start one `ckpt-bkp` session once.
Follow the `flywheel-4090.md` critical details exactly:

- pass `--repo` before `upload`;
- pass `--components checkpoints,runs,results` — never upload the `dagger`
  component; its per-episode `.npz` count on grid runs exceeds the HuggingFace
  Hub 20,000-file limit and the push is rejected;
- use `/root/.local/bin/uv` in tmux;
- transfer `HF_TOKEN` over SSH standard input only;
- never interpolate or print the token;
- use the derived `BACKUP_PREFIX`;
- do not start a second backup session.

Verify the backup session with both `tmux ls` and:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux capture-pane -t ckpt-bkp -p -S -10"
```

### 9. Monitor, resume, and handle failures

Use the remote controller state—not live tmux membership—to determine the
outcome. At any later time, including after an agent interruption, inspect:

```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cd '$CONTROL_DIR' && \
   { cat state/completed 2>/dev/null || true; } && \
   { cat state/failed 2>/dev/null || true; } && \
   find status -maxdepth 1 -type f -printf '%f: ' -exec cat {} \;"
```

Attach to the controller with
`ssh -t -p "$PORT" "root@$HOST" "tmux attach -t ablation-controller"`.
Attach to a currently running cell with its generated `ablation-*` session
name to view its live `tee` output. Read the authoritative output with
`tail -f "$CONTROL_DIR/logs/{run_name}.log"`; use the persisted log for completed
or failed cells and do not rely on `tmux capture-pane` after a cell has exited.

When `state/failed` exists, report the failed run and log. Ask the user whether
to retry it, continue without it, or stop. Do not modify the controller's plan
or launch later cells until the user answers. A retry or skip is a new,
explicitly confirmed controller invocation; preserve all existing log and
status files for auditability.

### 10. Follow-up download and replay

After the grid has completed, print:

```bash
set -a; . ./.env; set +a
uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel list
uv run python scripts/hf_backup.py --repo skpro19/toy-pickplace-flywheel \
  download BACKUP_PREFIX/RUN_NAME
uv run python scripts/final_score.py --run-name RUN_NAME
```

Generate one `final_score.py` command per `RUN_PLAN` row. Use one local
TensorBoard instance for all downloaded `abl-*` runs.

## Final output

Provisioning setup is complete after the selected instance passes the tier
gate, TensorBoard, `ablation-controller`, `ckpt-bkp`, and local wrappers are
verified. Grid execution completes only when `state/completed` exists, or a
failure has been explicitly handled by the user.

Print:

- parameter names, exact value lists, `GRID_CELLS`, selected tier, and
  `RUN_PLAN`;
- instance ID and `vastai ssh-url` retrieval command;
- advertised effective vCPUs, verified physical cores, CPU quota, CPU reserve,
  CPU cap, computed concurrency, and batch count;
- applied `workers`, `dataloader_workers`, and batch-size policy;
- `CONTROL_DIR`, controller state, current/completed batch status, and all
  run-specific tmux attach commands;
- local `LOCAL_SSH_SESSION` and `LOCAL_TB_SESSION` attach commands;
- the indexed local TensorBoard URL;
- HF backup prefix, download commands, and `final_score.py` commands;
- cleanup destroy command.

## Safety notes

- Do not store API keys, SSH details, tokens, or transient host/port in files.
- RTX 4090 offers are volatile: use the confirmed shortlist directly with
  `--cancel-unavail`; stop after the first successful create.
- A high advertised effective-vCPU count never bypasses the SSH physical-core
  gate.
- This command rejects grids larger than 30 cells rather than silently changing
  the experiment design or provisioning more instances.
