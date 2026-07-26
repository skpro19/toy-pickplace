---
description: Provision one RTX 4090 and run one config-defined flywheel
agent: build
---

Run one standard flywheel on one Vast.ai RTX 4090. This command owns baseline
confirmation, provisioning, hardware acceptance, setup, durable launch,
checkpoint backup, automatic held-out evaluation, result upload, and instance
destruction. It is
self-contained: follow this workflow without consulting another runbook.

The experiment is fixed by the committed YAML config. Do not collect sweep
parameters, construct a grid, or modify the config on the instance. Do not add
experiment CLI overrides. The only flywheel CLI arguments supplied by this
workflow are `--config` and `--run-name`. The run name encodes the
architecture (`arch`) and user-selected config parameters.

## Workflow at a glance

1. Confirm the remote branch, commit, exact config contents, selected
   run-name-encoded parameters, run name, and effective S3 path.
2. Load secrets, select an offer, and provision one instance.
3. Accept or reject the instance using SSH hardware checks.
4. Clone the exact commit and verify CUDA and headless MuJoCo.
5. Start TensorBoard, local wrappers, one durable flywheel runner, and backup.
6. Start a durable local supervisor that monitors terminal state.
7. After a successful fresh backup, automatically run held-out evaluation,
   upload its JSON and plots, then destroy the instance. On failure, upload
   diagnostics best-effort and destroy the instance.

## Baseline and run planning

This command accepts no experiment parameter arguments. If arguments are
provided, explain that this workflow runs the committed config unchanged and
stop.

Ask the user to select the Git branch and committed flywheel config to clone,
defaulting to `dev` and the standard config, then set:

```text
GIT_BRANCH=<confirmed branch>
FLYWHEEL_CONFIG=<confirmed committed config path>
```

The default config path is
`configs/flywheel/default_mlp_vision_instance.yaml`. A smoke test may use a
dedicated committed smoke config, but it remains immutable and follows every
normal workflow gate. Display and confirm its complete contents before
provisioning.

Validate and load the remote files without changing the local checkout:

```bash
case "$GIT_BRANCH" in
  ""|*[!A-Za-z0-9._/-]*)
    printf '%s\n' 'ERROR: Branch may contain only letters, digits, ., _, /, and -' >&2
    exit 1
    ;;
esac
git check-ref-format --branch "$GIT_BRANCH"
git fetch origin "$GIT_BRANCH:refs/remotes/origin/$GIT_BRANCH" || {
  printf 'ERROR: Could not fetch origin/%s\n' "$GIT_BRANCH" >&2
  exit 1
}
GIT_COMMIT=$(git rev-parse "origin/$GIT_BRANCH^{commit}") || exit 1
REMOTE_CONFIG=$(git show "$GIT_COMMIT:$FLYWHEEL_CONFIG") || {
  printf 'ERROR: %s does not exist on origin/%s\n' \
    "$FLYWHEEL_CONFIG" "$GIT_BRANCH" >&2
  exit 1
}
REMOTE_FLYWHEEL=$(git show "$GIT_COMMIT:scripts/flywheel.py") || {
  printf 'ERROR: scripts/flywheel.py does not exist at %s\n' \
    "$GIT_COMMIT" >&2
  exit 1
}
printf 'Branch: %s\nCommit: %s\nConfig: %s\n\n%s\n' \
  "$GIT_BRANCH" "$GIT_COMMIT" "$FLYWHEEL_CONFIG" "$REMOTE_CONFIG"
```

Confirm from `REMOTE_FLYWHEEL` that `--config` and `--run-name` are supported.
Parse the complete top-level config with `yaml.safe_load` and resolve every
parameter. Display the exact branch, commit, config path, and the entire config
as tables grouped by section (Run, Flywheel loop, Expert data collection,
Training, Dataset mixing, DAgger rollout, In-loop evaluation, Held-out
evaluation, Parallelism, Seeds). Resolve and list every value; do not use local
working-tree copies for planning. Then ask the user to confirm this immutable
experiment baseline. Never dump raw YAML in the response — always use the
grouped table format.

After the user confirms the baseline, ask which top-level config parameters
should be encoded in the run name. Present the parsed key-value pairs from
`REMOTE_CONFIG` and let the user select zero or more parameters by name.
`arch` is always included implicitly. Validate each selected name exists in
the parsed config and that its value is a scalar. Collect the ordered list:

```text
RUN_NAME_PARAMS=<ordered list of selected param names>
```

After confirmation, set each value once:

```bash
RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
UNIX_TIME_NS=$(date +%s%N)
ARCH=$(printf '%s\n' "$REMOTE_CONFIG" | uv run python -c \
  'import sys, yaml; print(yaml.safe_load(sys.stdin)["arch"])')
RUN_NAME_PARAM_SUFFIX=""
for param in $RUN_NAME_PARAMS; do
  val=$(printf '%s\n' "$REMOTE_CONFIG" | PARAM="$param" uv run python -c \
    'import os, sys, yaml; print(yaml.safe_load(sys.stdin)[os.environ["PARAM"]])')
  RUN_NAME_PARAM_SUFFIX="${RUN_NAME_PARAM_SUFFIX}${param}=${val}_"
done
RUN_NAME="${ARCH}-flywheel-${RUN_NAME_PARAM_SUFFIX}${RUN_TIMESTAMP}"
INSTANCE_LABEL="toy-pickplace-${RUN_NAME}-${UNIX_TIME_NS}"
CONTROL_DIR="/workspace/toy-pickplace/.flywheel/${RUN_NAME}"
```

The user may replace the generated `RUN_NAME` before provisioning. Validate a
custom name against `^[A-Za-z0-9][A-Za-z0-9._=-]*$`. Keep `RUN_NAME` unchanged
after confirmation and derive `INSTANCE_LABEL`, and
`CONTROL_DIR` from the final value. A new launch must not reuse an existing S3
prefix. A recovery may retain its previously recorded prefix only when the
matching instance and control state establish that it is the same workflow.

Display and explicitly confirm the complete plan before loading secrets:

| Item | Required value |
|---|---|
| Branch and commit | Confirmed `GIT_BRANCH` and immutable `GIT_COMMIT` |
| Config | Exact committed `FLYWHEEL_CONFIG` contents |
| Experiment parameters | Every value resolved from the config; no overrides |
| Root seed | Config-resolved `global_seed` |
| Encoded params | Selected `RUN_NAME_PARAMS` from config |
| Run name | `RUN_NAME` |
| Held-out evaluation | Committed `final_eval_episodes`, `final_eval_seed`, `final_eval_workers`, and `final_eval_capture_hz` |
| Hardware gate | One RTX 4090, 24 physical cores, 64 GB RAM |
| Launch command | `scripts/flywheel.py --config FLYWHEEL_CONFIG --run-name RUN_NAME` |

After a unique numeric `INSTANCE_ID` has been established, every unrecoverable
terminal failure must trigger best-effort diagnostics followed by automatic
destruction and removal verification. Before the local supervisor starts, the
agent performs this policy directly. After it starts, the supervisor owns the
policy. Never apply automatic destruction while duplicate-instance identity is
ambiguous; list duplicates and resolve the target with the user first.

## Fixed hardware profile

Use one fixed profile. There are no tiers, concurrency calculations, or
batches of experiment runs.

| Requirement | Value |
|---|---:|
| Search effective vCPU minimum | 24 |
| SSH physical-core minimum | 24 |
| Concurrent flywheel runs | 1 |
| `workers` | Committed config value |
| `dataloader_workers` | Committed config value |
| `batch_size` | Committed config value |
| Held-out evaluation | Committed config values; always runs after final backup |

## Prerequisites

| Tool / key | Check |
|---|---|
| `vastai` CLI | `which vastai` |
| `VAST_API_KEY` | Set in `.env` |
| `S3_BUCKET` and AWS credentials | Set in `.env`, unless the instance has an IAM role |

| `AWS_PROFILE` | Alternative to access-key variables via `~/.aws/credentials` |
| `aws` CLI | May be required to log in when `AWS_PROFILE` uses SSO |
| `tmux` | `which tmux` |
| `flock` | `which flock` |

## Workflow

### 0. Load secrets

Load `.env` without shell tracing and require `VAST_API_KEY` and `S3_BUCKET`:

```bash
set -a
. ./.env
set +a
test -n "${VAST_API_KEY:-}" || {
  printf '%s\n' 'ERROR: VAST_API_KEY is missing from .env' >&2
  exit 1
}
test -n "${S3_BUCKET:-}" || {
  printf '%s\n' 'ERROR: S3_BUCKET is missing from .env' >&2
  exit 1
}
```

Unless the instance has an IAM role, require either a complete access-key pair
or `AWS_PROFILE`. Reject a partial access-key pair. If only `AWS_PROFILE` is
set, resolve it before SSH transfer:

```bash
if test -n "${AWS_ACCESS_KEY_ID:-}" || test -n "${AWS_SECRET_ACCESS_KEY:-}"; then
  test -n "${AWS_ACCESS_KEY_ID:-}" && test -n "${AWS_SECRET_ACCESS_KEY:-}" || {
    printf '%s\n' 'ERROR: Both AWS access-key variables are required' >&2
    exit 1
  }
elif test -n "${AWS_PROFILE:-}"; then
  mapfile -t RESOLVED_AWS < <(uv run python -c '
import os
import boto3

session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
credentials = session.get_credentials()
if credentials is None:
    raise SystemExit(1)
frozen = credentials.get_frozen_credentials()
print(frozen.access_key)
print(frozen.secret_key)
print(frozen.token or "")
print(session.region_name or "")
') || exit 1
  test "${#RESOLVED_AWS[@]}" -eq 4 || {
    printf 'ERROR: Could not resolve credentials from AWS profile %s\n' \
      "$AWS_PROFILE" >&2
    exit 1
  }
  AWS_ACCESS_KEY_ID=${RESOLVED_AWS[0]}
  AWS_SECRET_ACCESS_KEY=${RESOLVED_AWS[1]}
  AWS_SESSION_TOKEN=${RESOLVED_AWS[2]}
  test -n "${AWS_REGION:-}" || AWS_REGION=${RESOLVED_AWS[3]}
  test -n "$AWS_ACCESS_KEY_ID" && test -n "$AWS_SECRET_ACCESS_KEY" || exit 1
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION
  unset RESOLVED_AWS
fi
```

Tell the user which required variable is missing. Never print secret values or
enable shell tracing. Before provisioning, use `boto3` to list at most one
object under the exact effective prefix
`S3_PREFIX/RUN_NAME/`. If an object exists, stop and require the user to
choose a new run name. Continue only when recovering the matching instance and
control state rather than launching a new run. Explain that the backup sync
deletes remote keys absent locally, so prefix reuse can be destructive.

### 1. Search and select offers

Search with every hard searchable requirement:

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 total_flops>=80 cpu_cores_effective>=24 cpu_ram>=64 disk_bw>=1000 pci_gen>=4 pcie_bw>=20 inet_down>=500 inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false' \
  --order dph_total+ \
  --raw
```

Show exactly these columns:

| Offer ID | CPU model | Effective vCPUs | RAM | Disk MB/s | PCIe GB/s | GPU power | Down/Up Mb/s | Reliability | $/hr | Location |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

Rank eligible offers by CPU family (EPYC 9005, EPYC 9004, Threadripper 7000,
EPYC 7003, then modern Ryzen 7000/9000), lowest price, disk bandwidth, and
reliability. Reject EPYC 7001/7002. CPU model is only a ranking hint because
the SSH allocation gate is authoritative. Do not use `cpu_ghz`.

Do not weaken a hard filter without explicit approval. If no offer passes, ask
whether to wait or relax named criteria. Otherwise recommend an offer and ask
the user to confirm one offer or an ordered shortlist of three to five current
offers. Do not run a separate availability check.

### 2. Provision one instance

Try the confirmed offers in order:

```bash
CREATED=false
OFFER_IDS=(OFFER_1 OFFER_2 OFFER_3)
for id in "${OFFER_IDS[@]}"; do
  output=$(vastai create instance "$id" \
    --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
    --disk 100 --ssh --direct --label "$INSTANCE_LABEL" \
    --cancel-unavail 2>&1)
  if printf '%s\n' "$output" | grep -q "new_contract"; then
    CREATED=true
    INSTANCE_ID=$(printf '%s\n' "$output" | grep -oP "new_contract': \K\d+")
    break
  fi
  for reconcile_attempt in 1 2 3; do
    INSTANCE_ID=$(vastai show instances --raw | \
      INSTANCE_LABEL="$INSTANCE_LABEL" uv run python -c '
import json
import os
import sys

instances = json.load(sys.stdin)
if isinstance(instances, dict):
    instances = instances.get("instances", [instances])
matches = [
    item
    for item in instances
    if item.get("label") == os.environ["INSTANCE_LABEL"]
]
if len(matches) > 1:
    ids = ", ".join(str(item.get("id")) for item in matches)
    print(f"ERROR: Multiple exact-label instances: {ids}", file=sys.stderr)
    raise SystemExit(2)
if matches:
    print(matches[0]["id"])
')
    reconcile_status=$?
    test "$reconcile_status" -eq 0 || exit "$reconcile_status"
    if test -n "$INSTANCE_ID"; then
      CREATED=true
      break
    fi
    sleep 2
  done
  if test "$CREATED" = true; then
    break
  fi
  sleep 1
done
test "$CREATED" = true ||
  printf '%s\n' 'WARNING: Create output did not yield an instance ID; reconciling by label' >&2
```

Reconcile by exact label after every create response that does not report
`new_contract`, before trying the next offer. Stop after the first reported or
reconciled success. Whether ID parsing succeeds or not, always run
`vastai show instances --raw` and identify every instance whose label exactly
equals `INSTANCE_LABEL`. This reconciliation prevents a changed or truncated
create response from leaving an untracked billed instance. If no matching
instance exists, stop without polling or cleanup against an empty ID. If one
exists, use its numeric ID as `INSTANCE_ID`. If multiple instances exist, list
their IDs, status, hardware, and price; never destroy one before obtaining user
confirmation. Verify exactly one remains before continuing.

### 3. Wait for running and secure SSH

Poll `vastai show instance "$INSTANCE_ID" --raw` every ten seconds for up to
30 attempts. Parse `actual_status` with `uv run python`; stop if it does not
become `running`. Then resolve a fresh endpoint with
`vastai ssh-url "$INSTANCE_ID"`; do not reuse create-response host data.

Probe SSH for up to 12 attempts at ten-second intervals. The first successful
probe may use `StrictHostKeyChecking=accept-new`. Verify the key is pinned with
`ssh-keygen -F "[$HOST]:$PORT"`, then require
`StrictHostKeyChecking=yes` and `BatchMode=yes` for every later connection,
including credential transfers.

If running or SSH readiness fails, report the instance ID, latest status,
endpoint when available, and attempts. Automatically destroy the instance and
verify removal. Ask whether to retry only after destruction; a retry must repeat
the offer search and obtain a fresh shortlist.

### 4. Verify provisioned hardware

Treat the rental as provisional. Before cloning, collect:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" '
  set -e
  lscpu
  lscpu -e=CPU,CORE,SOCKET,ONLINE
  lscpu -p=CORE,SOCKET | grep -v "^#" | sort -u | wc -l
  nproc
  grep "^Cpus_allowed_list:" /proc/self/status
  free -b
  test ! -r /sys/fs/cgroup/memory.max || cat /sys/fs/cgroup/memory.max
  test ! -r /sys/fs/cgroup/memory/memory.limit_in_bytes || cat /sys/fs/cgroup/memory/memory.limit_in_bytes
  test ! -r /sys/fs/cgroup/cpu.max || cat /sys/fs/cgroup/cpu.max
  test ! -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
  test ! -r /sys/fs/cgroup/cpu/cpu.cfs_period_us || cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  df -hT /workspace /
  nvidia-smi --query-gpu=name,memory.total,power.limit,power.default_limit,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv
'
```

Parse and display an acceptance table:

| Check | Requirement |
|---|---|
| Physical cores | At least 24 allowed unique `(CORE, SOCKET)` pairs |
| CPU generation | Zen 3 or newer; reject EPYC 7001/7002 |
| CPU quota | At least 90% of advertised effective vCPUs |
| RAM | At least 64 GB allocated |
| GPU | Exactly one RTX 4090 with approximately 24 GB VRAM |
| GPU power | At least 400 W |
| PCIe | Gen4 x16 capability and offer `pcie_bw >= 20` GB/s |
| Throttling | Thermal and power-brake slowdown inactive |

Count physical cores only from online CPU IDs in `Cpus_allowed_list`, counting
unique `(CORE, SOCKET)` pairs. Fail if the allowed cpuset cannot be established.
Interpret finite cgroup CPU and memory limits as the allocation gates; use
visible memory only when the cgroup limit is unlimited. An idle PCIe link may
downshift, so maximum Gen4 x16 capability plus the passing offer measurement is
sufficient unless other evidence indicates restriction.

Compare CPU model, logical CPUs, RAM, GPU identity, GPU power, and PCIe with the
recorded offer and list every mismatch. Disk and network values come from the
offer; do not claim the SSH checks measured workload, disk, or upload speed.

On failure, stop before setup, automatically destroy the provisional instance,
verify removal, and ask whether to return to Step 1.
After acceptance, display the committed `workers`, `dataloader_workers`,
`batch_size`, and all four `final_eval_*` values and ask for final launch
confirmation. No instance-side config override is permitted.

### 5. Clone and verify the environment

Clone only after all prior confirmations. Substitute the confirmed values:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "git clone --branch 'GIT_BRANCH' --single-branch \
     https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace && \
   cd /workspace/toy-pickplace && \
   test \"\$(git branch --show-current)\" = 'GIT_BRANCH' && \
   test \"\$(git rev-parse HEAD)\" = 'GIT_COMMIT' && \
   test -f 'FLYWHEEL_CONFIG' && \
   apt-get update -qq && apt-get install -y -qq \
     libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa libglfw3 && \
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   /root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace && \
   MUJOCO_GL=egl /root/.local/bin/uv run python -c \
     'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import mujoco, glfw; print(mujoco.__version__, glfw.__version__)'"
```

If the checked-out commit differs, stop before setup and restart baseline
confirmation. Configure tmux only; do not edit YAML:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "touch ~/.no_auto_tmux && printf '%s\n' 'set -g mouse on' > ~/.tmux.conf"
```

### 6. Start TensorBoard and local wrappers

Start one remote TensorBoard session:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s tensorboard \
     'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main \
      --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'"
```

Poll its local endpoint for up to 12 attempts at five-second intervals. Stop
before launch if the session exits or `http://127.0.0.1:6006/` does not become
reachable:

```bash
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "for i in \$(seq 1 12); do
     tmux has-session -t tensorboard 2>/dev/null &&
       curl -fsS http://127.0.0.1:6006/ >/dev/null && exit 0
     sleep 5
   done
   tmux capture-pane -t tensorboard -p -S -20 2>/dev/null || true
   exit 1"
```

Immediately allocate the lowest unused non-negative `LOCAL_WORKFLOW_INDEX`
under `flock` on `/tmp/toy-pickplace-flywheel-local-wrapper.lock`. Set:

```text
LOCAL_TB_PORT=6006 + LOCAL_WORKFLOW_INDEX
LOCAL_SSH_SESSION=vast-ssh-LOCAL_WORKFLOW_INDEX
LOCAL_TB_SESSION=tb-flywheel-LOCAL_WORKFLOW_INDEX
```

An index is occupied if either tmux session exists or its port is listening.
Select and create the wrappers while holding the lock:

```bash
set -e
exec 9>/tmp/toy-pickplace-flywheel-local-wrapper.lock
flock 9
LOCAL_WORKFLOW_INDEX=""
for index in $(seq 0 999); do
  candidate_ssh_session="vast-ssh-$index"
  candidate_tb_session="tb-flywheel-$index"
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
  "ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -o ServerAliveInterval=30 -p $PORT root@$HOST"
tmux new-session -d -s "$LOCAL_TB_SESSION" \
  "ssh -o StrictHostKeyChecking=yes -o BatchMode=yes \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N \
    -L $LOCAL_TB_PORT:127.0.0.1:6006 -p $PORT root@$HOST"
tmux has-session -t "$LOCAL_SSH_SESSION"
tmux has-session -t "$LOCAL_TB_SESSION"
LOCAL_TB_READY=false
for i in $(seq 1 12); do
  if tmux has-session -t "$LOCAL_TB_SESSION" 2>/dev/null &&
    curl -fsS "http://127.0.0.1:$LOCAL_TB_PORT/" >/dev/null; then
    LOCAL_TB_READY=true
    break
  fi
  sleep 1
done
test "$LOCAL_TB_READY" = true
flock -u 9
exec 9>&-
```

Print `http://localhost:$LOCAL_TB_PORT`. Stop before launch and report the
relevant local tmux output if either wrapper fails. Do not ask the user to
choose an index.

### 7. Materialize and launch the durable runner

Create these paths on the instance:

```text
CONTROL_DIR/
CONTROL_DIR/logs/
CONTROL_DIR/state/
CONTROL_DIR/state/history/
```

Build `runner.sh` locally using a quoted heredoc (`<< 'EOF'` so shell variables
are literal), then base64-encode it and decode it into `CONTROL_DIR`. Use short
bash variable names and distinct placeholder tokens that `sed` can replace
safely. Never use `${PLACEHOLDER}` as a token because after `sed` substitution
it becomes `${value}` — an invalid bash reference. Use uppercase tokens
surrounded by underscores (e.g. `__RUN_NAME__`). Replace with `sed` using `|`
delimiter (safe with paths containing `/`). Do not copy transient files back.

Example runner template:

```bash
cat > /tmp/runner.sh << 'RUNEOF'
#!/bin/bash
set -o pipefail
export MUJOCO_GL=egl

_R=__RUN_NAME__
_F=__FLYWHEEL_CONFIG__
_C=__CONTROL_DIR__

echo "running" > "${_C}/state/run-status.tmp"
mv "${_C}/state/run-status.tmp" "${_C}/state/run-status"

cd /workspace/toy-pickplace

/root/.local/bin/uv run python scripts/flywheel.py \
  --config "${_F}" --run-name "${_R}" 2>&1 | \
  tee -a "${_C}/logs/${_R}.log"
exit_code=${PIPESTATUS[0]}

if [ "$exit_code" -eq 0 ]; then
  echo "succeeded 0" > "${_C}/state/completed.tmp"
  mv "${_C}/state/completed.tmp" "${_C}/state/completed"
else
  echo "failed ${exit_code}" > "${_C}/state/failed.tmp"
  mv "${_C}/state/failed.tmp" "${_C}/state/failed"
fi

exit "${exit_code}"
RUNEOF
RUN_NAME=<value>; FLYWHEEL_CONFIG=<value>; CONTROL_DIR=<value>
sed -i "s|__RUN_NAME__|${RUN_NAME}|g; s|__FLYWHEEL_CONFIG__|${FLYWHEEL_CONFIG}|g; s|__CONTROL_DIR__|${CONTROL_DIR}|g" /tmp/runner.sh
```

Base64-encode and transfer, then start the runner.

The executable runner must:

1. use `set -o pipefail` and export `MUJOCO_GL=egl`;
2. atomically write `running` to `state/run-status`;
3. run exactly `/root/.local/bin/uv run python scripts/flywheel.py --config
   FLYWHEEL_CONFIG --run-name RUN_NAME` from `/workspace/toy-pickplace`;
4. stream stdout and stderr to both the pane and `logs/RUN_NAME.log` with
   `tee -a`;
5. preserve the flywheel exit code using `${PIPESTATUS[0]}`;
6. atomically write `succeeded 0` and `state/completed` on success, or
   `failed EXIT_CODE` and `state/failed` on failure;
7. exit with the flywheel process's exit code.

Every state update must use a temporary file followed by `mv`. A missing tmux
session is never a success signal. The persisted log and state files are
authoritative.

Before the first launch, fail if any of these run-specific artifact paths
already exists, even if the control directory is new:

```text
data/flywheel/RUN_NAME
checkpoints/flywheel/RUN_NAME
runs/flywheel/RUN_NAME
results/flywheel/RUN_NAME
```

List every collision and require a new run name for a new launch. Existing
artifacts may be inspected during recovery, but this workflow never relaunches
`flywheel.py` into them. Never treat an unrelated existing run as a recovery
target. Start the runner once in remote session `flywheel-run`. Refuse to launch
if the session or any run/control artifact already exists. Poll until
`state/run-status` says `running`, verify `flywheel-run` and `tensorboard` exist,
and stop if the runner exits before the marker appears. The runner may continue
after the agent disconnects.

### 8. Start checkpoint backup

After the run is marked `running`, start one `ckpt-bkp` session. Its wrapper
must wait until an artifact exists specifically under
`checkpoints/flywheel/RUN_NAME`, `runs/flywheel/RUN_NAME`,
`data/flywheel/RUN_NAME`, or `results/flywheel/RUN_NAME`, then run every 120
seconds:

```bash
/root/.local/bin/uv run python scripts/s3_backup.py upload \
  --prefix "$RUN_NAME" \
  --components checkpoints,runs,results,dagger \
  "$RUN_NAME"
```

The positional `RUN_NAME` is mandatory. Never upload whole component roots;
that can mix artifacts from other runs and can let old files satisfy the new
run's initial-backup gate.

Before each upload, atomically update `state/backup-cycle-started`. After a
successful upload, atomically update `state/backup-last-succeeded`. On failure,
atomically write `state/backup-failed` and exit non-zero.

Transfer credentials to the instance by piping a heredoc through SSH that
writes a temporary env file with 600 permissions. Have the backup wrapper
source that file, then shred it after the first upload. Never place
credentials in the SSH command string, wrapper script file, tmux global
environment, pane output, or logs. Use the absolute `uv` path.

Example credential transfer (run locally before launching ckpt-bkp):

Do not include `S3_ENDPOINT_URL` — an empty value causes `Invalid endpoint`. Omit it entirely:

```bash
set -a; . ./.env; set +a
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "cat > '${CONTROL_DIR}/s3-creds.sh' << 'CREDEOF'
S3_BUCKET=${S3_BUCKET}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
AWS_REGION=${AWS_REGION}
S3_PREFIX=${S3_PREFIX}
CREDEOF
chmod 600 '${CONTROL_DIR}/s3-creds.sh'"
```

The backup wrapper must source that file before each upload and shred it
after the first successful cycle. Build the wrapper locally and transfer it
via base64, using the same `_C`, `_R` placeholder pattern as `runner.sh`:

```bash
cat > /tmp/ckpt-bkp-wrapper.sh << 'WRAPEOF'
#!/bin/bash
set -o pipefail
_R=__RUN_NAME__
_C=__CONTROL_DIR__

# wait for any artifact
for i in $(seq 1 60); do
  for d in checkpoints/flywheel/${_R} runs/flywheel/${_R} \
           data/flywheel/${_R} results/flywheel/${_R}; do
    if test -d "/workspace/toy-pickplace/$d" && \
       find "/workspace/toy-pickplace/$d" -type f 2>/dev/null | \
       head -1 | grep -q .; then break 2; fi
  done; sleep 5
done

# source credentials once
. "${_C}/s3-creds.sh"
shred -u "${_C}/s3-creds.sh" 2>/dev/null

while true; do
  touch "${_C}/state/backup-cycle-started"
  cd /workspace/toy-pickplace
  S3_BUCKET="$S3_BUCKET" S3_PREFIX="$S3_PREFIX" \
    AWS_REGION="$AWS_REGION" \
    AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
    /root/.local/bin/uv run python scripts/s3_backup.py upload \
      --prefix "${_R}" --components checkpoints,runs,results,dagger "${_R}"
  if [ "$?" -eq 0 ]; then
    touch "${_C}/state/backup-last-succeeded"
  else
    echo "failed" > "${_C}/state/backup-failed.tmp"
    mv "${_C}/state/backup-failed.tmp" "${_C}/state/backup-failed"
    exit 1
  fi
  sleep 120
done
WRAPEOF
sed -i "s|__RUN_NAME__|${RUN_NAME}|g; s|__CONTROL_DIR__|${CONTROL_DIR}|g" \
  /tmp/ckpt-bkp-wrapper.sh
B64=$(base64 -w0 /tmp/ckpt-bkp-wrapper.sh)
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "printf '%s' '${B64}' | base64 -d > '${CONTROL_DIR}/ckpt-bkp-wrapper.sh'
   chmod +x '${CONTROL_DIR}/ckpt-bkp-wrapper.sh'"
ssh -o StrictHostKeyChecking=yes -o BatchMode=yes -p "$PORT" "root@$HOST" \
  "tmux new-session -d -s ckpt-bkp \
     'cd /workspace/toy-pickplace && exec bash ${CONTROL_DIR}/ckpt-bkp-wrapper.sh'"
```

Refuse to start if `ckpt-bkp` already exists on the remote host.

Poll for up to ten minutes for `state/backup-last-succeeded`. Fail immediately
if `state/backup-failed` appears or `ckpt-bkp` exits. After success, inspect the
last ten pane lines without exposing credentials and verify the session remains
active. A later `backup-failed` marker or unexpected stopped session is a
workflow failure even if an earlier upload succeeded. If the initial backup
fails before the local supervisor starts, upload diagnostics best-effort,
destroy the instance, and verify removal directly.

### 9. Materialize automatic held-out evaluation

Materialize `CONTROL_DIR/heldout-eval.sh` before starting the local supervisor,
using the same local-build and base64-transfer method as the runner. It must:

1. use `set -o pipefail`, export `MUJOCO_GL=egl`, and refuse to run if
   `state/heldout-completed` or `state/heldout-failed` exists;
2. parse `results/flywheel/RUN_NAME/metrics.json`, require a non-empty `rounds`
   list, and verify every listed round has
   `checkpoints/flywheel/RUN_NAME/round-NNN/best.pt`;
3. read `final_eval_episodes`, `final_eval_seed`, `final_eval_workers`, and
   `final_eval_capture_hz` from `metrics.json.config`, validate them against the
   same constraints used by `scripts/flywheel.py`, and never prompt or invent
   overrides;
4. fail if any expected output already exists. Automated first launch never
   archives, overwrites, or treats an old output as current;
5. run exactly `/root/.local/bin/uv run python scripts/final_score.py --run-name
   RUN_NAME` from `/workspace/toy-pickplace`. The evaluator inherits all four
   held-out values from `metrics.json`;
6. stream stdout and stderr to `logs/heldout-eval-RUN_NAME.log` and preserve the
   evaluator exit code using `${PIPESTATUS[0]}`;
7. validate that these three non-empty outputs exist:

```text
results/flywheel/RUN_NAME/final_scores.json
results/flywheel/RUN_NAME/final_scores_comparison.png
results/RUN_NAME-final-score-curve.png
```

8. validate that `final_scores.json` records the exact committed episodes,
   seed, workers, capture rate, and every round from `metrics.json`;
9. strip leading/trailing `/` from `S3_PREFIX`, then upload the three files
   with `boto3.client("s3").upload_file` to these effective keys (omit the
   leading `S3_PREFIX/` segment when it is empty):

```text
S3_PREFIX/RUN_NAME/results/RUN_NAME/final_scores.json
S3_PREFIX/RUN_NAME/results/RUN_NAME/final_scores_comparison.png
S3_PREFIX/RUN_NAME/results/RUN_NAME/RUN_NAME-final-score-curve.png
```

10. verify each exact object with `head_object` and matching content length;
11. atomically write `state/heldout-completed` only after all three
    verifications, or `state/heldout-failed` with the exit code on any failure.

Do not use a component-wide results upload after evaluation. It can delete or
mix objects outside this exact output set. Give AWS credentials only to the
single `heldout-eval` process through SSH standard input and its inherited
environment. Never put credentials in the wrapper, command arguments, tmux
global environment, logs, or pane output.

### 10. Start the durable local supervisor

Set `LOCAL_SUPERVISOR_SESSION=flywheel-supervisor-LOCAL_WORKFLOW_INDEX` and
reserve it under the same local wrapper lock. Build a generic local supervisor
at `/tmp/toy-pickplace-flywheel-supervisor.sh`; the file may refer to `.env` but
must not contain secret values or a transient endpoint. Pass `INSTANCE_ID`,
`HOST`, `PORT`, `RUN_NAME`, `CONTROL_DIR`, and local tmux session names as
arguments when starting it. Source `.env` inside the supervisor with tracing
disabled so `VAST_API_KEY` and AWS credentials remain local.

The supervisor must implement this state machine:

| State | Required action |
|---|---|
| Running | Poll remote state files every 30 seconds; tmux membership is diagnostic only |
| Training failed | Record the exit code, upload control logs/state best-effort, then clean up |
| Backup failed or stopped | Record the failure, upload control logs/state best-effort, then clean up |
| Training completed | Wait up to ten minutes for a backup cycle started after `state/completed` and succeeded after its start |
| Final backup fresh | Stop `ckpt-bkp`, then launch `heldout-eval` exactly once via secure credential transfer; set `HELDOUT_LAUNCHED` flag and advance to evaluation-running state. Do not re-enter the completed/fresh-backup branch after this transition |
| Evaluation running | Poll only for `heldout-completed`, `heldout-failed`, SSH degradation, or instance terminal. Never re-check `state/completed` or `state/backup-failed` while evaluation is running. Log progress every 30 s |
| Evaluation failed | Upload control logs/state best-effort, then clean up |
| Evaluation completed | Recheck the three exact S3 objects and sizes from the local supervisor, then clean up successfully |
| SSH degraded | Refresh `actual_status` and `vastai ssh-url`; repin a changed endpoint and continue polling while Vast reports `running` |
| Instance terminal | When Vast reports a terminal status, upload diagnostics when reachable, then clean up |

The fresh-backup gate remains:

```bash
test "$CONTROL_DIR/state/backup-cycle-started" -nt "$CONTROL_DIR/state/completed"
test "$CONTROL_DIR/state/backup-last-succeeded" -nt "$CONTROL_DIR/state/backup-cycle-started"
```

For best-effort failure diagnostics, upload regular files under
`CONTROL_DIR/logs` and `CONTROL_DIR/state` below
`S3_PREFIX/RUN_NAME/control/logs/` and `S3_PREFIX/RUN_NAME/control/state/`,
preserving relative names and omitting the `S3_PREFIX/` segment when empty.
Never upload credential material. Diagnostic upload failure must be recorded
locally but must not block destruction, per the selected cleanup policy. Bound
the entire diagnostic attempt to five minutes so cleanup cannot hang on S3.

On every terminal state, successful or failed, the supervisor must:

1. run `vastai destroy instance -y "$INSTANCE_ID"` locally; the `-y` flag is required to skip the interactive confirmation prompt which would otherwise stall cleanup silently
2. poll `vastai show instances --raw` until the numeric ID is absent, for up to
   30 attempts at ten-second intervals;
3. retry the destroy command once if the ID remains, then record a prominent
   local cleanup failure instead of claiming success;
4. stop the local SSH and TensorBoard wrapper sessions after destruction;
5. atomically write a terminal summary to
   `/tmp/toy-pickplace-flywheel-RUN_NAME.status`, including workflow outcome,
   diagnostic-upload outcome, destruction outcome, and uploaded result keys.

Start the supervisor in detached tmux and verify it remains active through its
first successful remote poll. Print its attach command. Never transfer
`VAST_API_KEY` to the instance. The local machine and tmux server must remain
running; a terminal or OpenCode disconnect is safe, but a local reboot is not.
Do not treat SSH or local-network unavailability as terminal. Refresh the Vast
status and endpoint after three failed probes. If the endpoint changes, pin its
host key before reconnecting. While Vast reports `running`, keep polling and
never destroy based only on failed SSH probes. After 120 consecutive failures,
write a prominent local alert and continue at a slower five-minute interval.
If both the Vast API and SSH are unavailable, keep retrying and record the
degraded state; do not infer instance failure from missing connectivity.

### 11. Follow-up download and replay

Print these as optional commands only; do not download automatically:

```bash
set -a; . ./.env; set +a
uv run python scripts/s3_backup.py download RUN_NAME/RUN_NAME
uv run python scripts/final_score.py --run-name RUN_NAME/RUN_NAME
uv run python scripts/final_score.py --run-name RUN_NAME/RUN_NAME --plot-only
```

## Final output

Provisioning setup is complete after hardware acceptance, TensorBoard, local
wrappers, `flywheel-run`, initial backup, held-out wrapper, and local supervisor
are verified. The workflow is complete only when the supervisor reaches a
terminal state and verifies instance destruction. A successful workflow also
requires training completion, a fresh final backup, held-out completion, and
all three result objects verified on S3.

Print:

- confirmed branch, commit, config path, exact fixed parameters, and root seed;
- run name (encoding arch and selected params), instance label, instance ID, and SSH URL command;
- advertised effective vCPUs, verified physical cores, CPU quota, RAM, GPU,
  power, and PCIe acceptance results;
- committed `workers`, `dataloader_workers`, `batch_size`, and all four
  `final_eval_*` values;
- `CONTROL_DIR`, durable run state, log path, and remote attach command;
- local SSH, TensorBoard, and supervisor attach commands and TensorBoard URL;
- backup state, S3 prefix, optional download and replay commands;
- held-out settings, state, log, and all three exact uploaded result paths;
- terminal supervisor status path and verified instance-destruction state.

## Safety notes

- Never store API keys, AWS credentials, tokens, or transient SSH endpoints in
  repository or remote workflow files.
- RTX 4090 offers are volatile. Use the confirmed shortlist directly with
  `--cancel-unavail` and stop after the first successful create.
- Advertised effective vCPUs never bypass the SSH physical-core gate.
- The committed config is immutable for this workflow; there are no parameter
  sweeps, per-run experiment overrides, concurrency calculations, or batches.
- Held-out evaluation always runs after a fresh final backup and uploads only
  its JSON and two plots under `S3_PREFIX/RUN_NAME/results/RUN_NAME/`.
- Every terminal state triggers destruction. Failure diagnostics are
  best-effort so an upload outage cannot keep a billed instance alive.
