---
description: Provision a Vast.ai RTX 4090 instance and start the flywheel pipeline
agent: general
---

Provision and set up a Vast.ai instance to run the flywheel training pipeline.

Read `docs/vast-ai/vast-ai-2.md` for the full runbook. Follow the "Provisioning workflow" section step by step.

## Workflow

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

### 2. Pre-creation check

Run the availability check on the chosen offer. If the offer is gone, go back to step 1.

### 3. Create instance

Create the instance with the documented command (100 GB disk, PyTorch image, direct SSH, `--cancel-unavail`).

### 4. Poll for running

Poll every 10 seconds until `actual_status` is `"running"`, then get the SSH URL.

### 5. Tune config

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

### 6. Setup on the instance

SSH in and run all commands from the doc's "Setup on the instance" section in order:
- Clone the repo (`dev` branch)
- Install uv
- `uv sync --locked`
- Verify CUDA (`torch.cuda.is_available()`)
- Disable auto-tmux (`touch ~/.no_auto_tmux`)
- Enable tmux mouse
- Generate 100 expert episodes
- Launch flywheel in tmux:`flywheel`
- Launch TensorBoard in tmux:`tensorboard`

### 7. Local tmux wrappers

On the dev machine, parse HOST/PORT from `vastai ssh-url INSTANCE_ID` and create:
- tmux:`vast-ssh` (SSH shell into the instance, use `ServerAliveInterval=30` to prevent idle disconnects)
- tmux:`tb-setup` (TensorBoard port tunnel)

Print the TensorBoard URL and attach commands.

## Final output

When complete, print a summary with:
- Instance ID
- SSH URL retrieval command (`vastai ssh-url INSTANCE_ID`)
- Attach commands for all 4 tmux sessions
- TensorBoard URL
- Destroy command for cleanup

## Notes

- Do not store API keys, instance API keys, Jupyter tokens, SSH keys, or transient host/port in any file.
- If an offer disappears between search and creation, loop back to search.
- Update `docs/vast-ai/vast-ai-2.md` with the new instance details as you go.
