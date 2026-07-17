# Vast.ai Flywheel Run Log: Instance 45152919

Templated provisioning workflow plus the record of what was done for this run.
Do not store API keys, instance API keys, Jupyter tokens, private SSH keys, or
the transient host and port after they change.

## Current run

| Field | Value |
|---|---|
| Instance ID | `45152919` |
| Offer | `40764560` — Denmark, 32 vCPUs @ 5.5 GHz |
| GPU | 1x RTX 4090, 24.6 GB VRAM |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Disk | 100 GB |
| Label | `toy-pickplace-flywheel` |
| Flywheel run | `run-001` — complete. Final scoring done. |
| Best checkpoint | Round 006 — score 0.8520, placement 0.7800 |
| Final scores | `results/flywheel/run-001/final_scores.json` |

### How to connect

```bash
# Get current endpoint (changes on reboot)
vastai ssh-url 45152919

# SSH into the instance via a local tmux session (so it persists after detach)
tmux new-session -d -s vast-ssh \
  'ssh -o StrictHostKeyChecking=no -p PORT root@HOST'
tmux attach -t vast-ssh

# SSH tunnel for TensorBoard (run on dev machine)
tmux new-session -d -s tb-setup \
  'ssh -N -L 6006:127.0.0.1:6006 -p PORT root@HOST'

# Open in browser
# http://127.0.0.1:6006

# Attach to sessions
tmux attach -t vast-ssh     # SSH shell on the instance
tmux attach -t flywheel      # flywheel log (inside the instance)
tmux attach -t tensorboard   # TensorBoard log (inside the instance)
tmux attach -t ckpt-bkp      # HF Hub backup loop (inside the instance)
tmux attach -t tb-setup      # SSH tunnel (on the dev machine)
```

### Download backed-up checkpoints

```bash
# List available sessions in the HF repo
uv run python scripts/hf_backup.py list --repo skpro19/toy-pickplace-flywheel

# Download a specific run
uv run python scripts/hf_backup.py download 20260717-153000/run-003 \
  --repo skpro19/toy-pickplace-flywheel

# View TensorBoard locally
tensorboard --logdir runs/flywheel/
```

The instance's default auto-attach to a session named `ssh_tmux` has been
disabled with `touch ~/.no_auto_tmux`. Plain `ssh` now gives a bare shell
without wrapping you into a tmux session.

Mouse support is enabled on the instance (set in `~/.tmux.conf`):

```bash
echo "set -g mouse on" > ~/.tmux.conf
tmux set-option -g mouse on
```

New sessions inherit this automatically. For existing sessions (`flywheel`,
`tensorboard`), the global option applies immediately.

If the instance IP/port changes, kill and re-create the tunnel:

```bash
vastai ssh-url 45152919
tmux kill-session -t tb-setup
tmux new-session -d -s tb-setup \
  'ssh -N -L 6006:127.0.0.1:6006 -p PORT root@HOST'
```

### Cleanup

```bash
vastai destroy instance 45152919 -y
```

Status: **destroyed**.

---

## Provisioning workflow (for reuse)

### Search offers

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 cpu_cores_effective>=16 rentable=true verification=verified' \
  --order dph_total+
```

Results must be presented as a table with **exactly these columns**:

| Offer ID | GPU frac | VRAM | Effective vCPUs | CPU GHz | $/hr | Host reliability | Driver | Location |

The command must recommend the best offer based on this priority:
1. **$/hr** (lowest first)
2. **Effective vCPUs** (≥ 32 preferred for 12-worker recommendation)
3. **CPU GHz** (higher is better for simulator workers)
4. **Host reliability** (≥ 99% preferred)

Show the recommendation with a brief rationale, then **ask the user to confirm**
or select a different offer from the table before proceeding.

### Pre-creation availability check

Confirm the chosen offer is still rentable before creating:

```bash
vastai search offers \
  'id=OFFER_ID gpu_name=RTX_4090 gpu_frac=1 num_gpus=1 cpu_cores_effective>=16 rentable=true verification=verified'
```

### Create instance

Use `--cancel-unavail` so Vast.ai fails instead of creating a stopped instance
if the offer is taken during launch:

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --disk 100 \
  --ssh \
  --direct \
  --label toy-pickplace-flywheel \
  --cancel-unavail
```

### Poll for running status and get SSH URL

```bash
for i in $(seq 1 30); do
  status=$(vastai show instance INSTANCE_ID --raw 2>/dev/null | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null)
  echo "Poll $i: status=$status"
  [ "$status" = "running" ] && { echo "READY"; break; }
  sleep 10
done
vastai ssh-url INSTANCE_ID
```

### Tune config for provisioned hardware

Based on the instance's effective vCPUs (shown in the search results and
confirmed in the offer table), recommend these overrides for
`configs/flywheel/default.yaml`:

| Effective vCPUs | Recommended `workers` | Recommended `dataloader_workers` | `batch_size` (optional) |
|---|---:|---:|---:|
| ≥ 24 | 12 | 0 | 768 (benchmarked best) |
| 16–23 | 6 | 0 | 768 (benchmarked best) |

Benchmark reference (`vast-ai-1.md` § Performance tuning):
- `workers: 12` — 12% faster evaluation, 31% faster DAgger collection vs 6
- `batch_size: 768` — 49% faster training than 200, best placement (36%)
- `dataloader_workers: 0` — dataset is in-memory; IPC overhead not justified

Present the recommendation based on the provisioned instance's vCPUs, then
**ask the user to confirm** or adjust before proceeding.

### Setup on the instance

```bash
git clone --branch dev --single-branch \
  https://github.com/skpro19/toy-pickplace.git /workspace/toy-pickplace
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv sync --locked --directory /workspace/toy-pickplace

# Verify CUDA
cd /workspace/toy-pickplace
/root/.local/bin/uv run python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

# Disable auto-tmux on SSH login (Vast.ai default behaviour)
touch ~/.no_auto_tmux

# Enable mouse support in tmux
echo "set -g mouse on" > ~/.tmux.conf
tmux set-option -g mouse on

# Generate expert data (100 episodes, ~30 s)
/root/.local/bin/uv run python scripts/data.py \
  --episodes 100 --out-dir data/expert/rand-100 --seed 0 --max-steps 8000

# Set HF token for checkpoint backup (picked up from dev machine's .env)
export HF_TOKEN="$HF_TOKEN"

# Launch flywheel (tmux: flywheel)
tmux new-session -d -s flywheel \
  'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python scripts/flywheel.py --config configs/flywheel/default.yaml'

# Launch TensorBoard (tmux: tensorboard)
tmux new-session -d -s tensorboard \
  'cd /workspace/toy-pickplace && exec /root/.local/bin/uv run python -m tensorboard.main --logdir /workspace/toy-pickplace/runs/flywheel --host 127.0.0.1 --port 6006'

# Launch HF Hub backup (tmux: ckpt-bkp)
# Syncs checkpoints + TensorBoard logs to HuggingFace Hub every 2 minutes.
# The prefix is a human-readable timestamp to avoid run-name conflicts.
BACKUP_PREFIX=$(date +%Y%m%d-%H%M%S)
tmux new-session -d -s ckpt-bkp \
  "cd /workspace/toy-pickplace && while true; do \
    uv run python scripts/hf_backup.py upload --repo skpro19/toy-pickplace-flywheel --prefix \$BACKUP_PREFIX; \
    sleep 120; \
  done"
```

### Local tmux wrappers (on the dev machine)

After the instance is running, create persistent local tmux sessions for SSH
access and the TensorBoard tunnel. These use the instance endpoint reported by
`vastai ssh-url INSTANCE_ID`:

```bash
# Parse HOST and PORT from the SSH URL
INSTANCE_ID=45152919
SSH_URL=$(vastai ssh-url $INSTANCE_ID)           # ssh://root@HOST:PORT
HOST=$(echo "$SSH_URL" | sed 's/.*@//;s/:.*//')
PORT=$(echo "$SSH_URL" | sed 's/.*://')

# Local tmux: SSH shell into the instance
tmux new-session -d -s vast-ssh \
  "ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"

# Local tmux: TensorBoard tunnel
tmux new-session -d -s tb-setup \
  "ssh -N -L 6006:127.0.0.1:6006 -p $PORT root@$HOST"

echo "Attach: tmux attach -t vast-ssh"
echo "TensorBoard: http://127.0.0.1:6006"
```

### Local download and replay

After the flywheel run has produced checkpoints (or after the run completes),
download from HuggingFace Hub and evaluate locally:

```bash
# List available sessions
uv run python scripts/hf_backup.py list --repo skpro19/toy-pickplace-flywheel

# Download a specific run
uv run python scripts/hf_backup.py download 20260717-153000/run-003 \
  --repo skpro19/toy-pickplace-flywheel

# View TensorBoard on the downloaded logs (no SSH tunnel needed)
tensorboard --logdir runs/flywheel/

# Re-evaluate on held-out seeds
uv run python scripts/final_score.py --run-name run-003
```

---

## Search history (this run)

Offers returned when the search was performed. Availability changes
continuously; always re-search before provisioning.

| Offer ID | GPU frac | VRAM | Effective vCPUs | CPU GHz | $/hr | Host reliability | Driver | Location |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `39709814` | 1 | 24.6 GB | 16 | 2.6 | 0.2676 | 99.2% | 570.133.20 | South Africa |
| `40764560` | 1 | 24.6 GB | 32 | 5.5 | 0.3214 | 98.0% | 595.71.05 | Denmark |
| `27791335` | 1 | 24.6 GB | 32 | 3.4 | 0.3214 | 99.9% | 575.57.08 | Turkiye |
| `44694364` | 1 | 24.6 GB | 32 | 6.0 | 0.3259 | 95.6% | 580.159.03 | Australia |
| `41885882` | 1 | 24.6 GB | 28 | 3.3 | 0.3347 | 99.7% | 535.230.02 | South Korea |
| `32944337` | 1 | 24.6 GB | 28 | 3.3 | 0.3347 | 99.9% | 570.133.07 | South Korea |
| `44330457` | 1 | 24.6 GB | 32 | 3.5 | 0.3613 | 99.0% | 595.71.05 | New Jersey, US |
| `37347938` | 1 | 24.6 GB | 32 | 2.5 | 0.3637 | 99.4% | 580.119.02 | California, US |
| `24838874` | 1 | 24.6 GB | 32 | 5.8 | 0.3747 | 99.7% | 575.57.08 | Nebraska, US |
| `44396536` | 1 | 24.6 GB | 24 | 5.6 | 0.3747 | 92.9% | 580.142 | North Macedonia |

**Selection rationale**: `40764560` was chosen for its balance of 32 vCPUs at
5.5 GHz, $0.3214/hr, and 98.0% reliability. The original pick (`45101552`,
$0.2947/hr) disappeared between search and creation — always re-confirm
availability just before `vastai create instance` and use `--cancel-unavail`.
