# Instance Checkpoint Sync

Backup flywheel checkpoints and TensorBoard logs to HuggingFace Hub during
instance runs, then download locally for replay and evaluation.

## Architecture

```
Instance (Vast.ai)                       HuggingFace Hub                 Local machine
─────────────────────                    ────────────────                ──────────────
tmux: flywheel ───┐                                                    ┌── tensorboard
                  │                                                    │
                  │   every 120s                                       │
tmux: ckpt-bkp ───┼── upload ──────────► skpro19/                      ├── final_score.py
                  │                       toy-pickplace-                ├── eval.py
tmux: tensorboard │                       flywheel/                    └── hf_backup.py
  (on-instance TB) │                       └── <timestamp>/                 download
                  │                           ├── checkpoints/...
                                                └── runs/...
```

## New files

| File | Purpose |
|---|---|
| `scripts/hf_backup.py` | CLI: `upload`, `download`, `list`, `rm` for HF Hub sync |
| `docs/instance-ckpt-sync.md` | This file — overview of the sync mechanism |

## Modified files

| File | Change |
|---|---|
| `pyproject.toml` | Added `huggingface_hub>=0.30.0` dependency |
| `.opencode/commands/flywheel-4090.md` | Step 6: `HF_TOKEN` + `ckpt-bkp` tmux. Step 7: download instructions |
| `docs/vast-ai/vast-ai-2.md` | Setup section: `HF_TOKEN`, `ckpt-bkp` tmux. Connect section: download instructions |

## CLI reference

```bash
# Upload (runs in ckpt-bkp tmux on the instance)
uv run python scripts/hf_backup.py upload --prefix $(date +%Y%m%d-%H%M%S)
uv run python scripts/hf_backup.py upload --prefix 20260717-153000 run-003

# Download (run locally after instance is done)
uv run python scripts/hf_backup.py list
uv run python scripts/hf_backup.py download 20260717-153000/run-003

# Remove
uv run python scripts/hf_backup.py rm 20260717-153000
uv run python scripts/hf_backup.py rm 20260717-153000/run-003
```

## HF repo structure

```
skpro19/toy-pickplace-flywheel/
├── 20260717-153000/             ← timestamp prefix, unique per instance run
│   ├── checkpoints/run-003/round-000/best.pt
│   ├── checkpoints/run-003/round-000/last.pt
│   ├── checkpoints/run-003/round-001/best.pt
│   ├── runs/run-003/round-000/events.out.tfevents.*
│   └── runs/run-003/round-001/events.out.tfevents.*
└── 20260718-090000/
    └── ...
```

The timestamp prefix prevents run-name conflicts when provisioning multiple
instances (each instance uses its own prefix, so `run-003` from instance A and
`run-003` from instance B coexist without collision).

## Local replay

After `download`:

```bash
# TensorBoard on the full run history
tensorboard --logdir runs/flywheel/

# Re-evaluate on held-out seeds
uv run python scripts/final_score.py --run-name run-003

# Single checkpoint eval
uv run python scripts/eval.py --ckpt checkpoints/flywheel/run-003/round-006/best.pt
```
