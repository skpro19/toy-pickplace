# Data Flywheel

## Project Structure

Keep expert data immutable and store each DAgger collection in its own round directory.

```text
data/
├── expert/
│   └── rand-100/                  # Never modified
└── flywheel/
    └── run-001/
        ├── round-001/
        │   └── dagger/
        │       ├── pick_place_000000.npz
        │       └── ...
        ├── round-002/
        │   └── dagger/
        └── round-003/
            └── dagger/

checkpoints/
└── flywheel/
    └── run-001/
        ├── round-000/             # Expert-only training
        ├── round-001/
        └── round-002/

runs/
└── flywheel/
    └── run-001/
        ├── round-000/             # TensorBoard logs
        ├── round-001/
        └── round-002/

results/
└── flywheel/
    └── run-001/
        └── metrics.json
```

## Round Flow

```text
round-000:
  train: expert
  output: best checkpoint 0

round-001:
  collect: dagger-001 using checkpoint 0
  train: expert + dagger-001
  output: best checkpoint 1

round-002:
  collect: dagger-002 using checkpoint 1
  train: expert + dagger-001 + dagger-002
  output: best checkpoint 2
```

Train from scratch in each round, evaluate checkpoints on fixed seeds, and use the best checkpoint for the next DAgger collection. Decrease `beta` between rounds.

## Data Inputs

Maintain a list of immutable directories instead of physically merging `.npz` files:

```python
expert_dir = "data/expert/rand-100"
dagger_dirs = []

for round_idx in range(rounds):
    dagger_dir = collect_dagger(best_checkpoint)
    dagger_dirs.append(dagger_dir)

    train_dirs = [expert_dir, *dagger_dirs]
    best_checkpoint = train_and_select_best(train_dirs)
```

All DAgger rounds remain in the training set so the policy does not forget states visited by earlier policies.

## Sampling Ratios

For a simple fixed 50/50 expert-to-DAgger mixture, reserve half of the samples for expert data and divide the other half among the accumulated DAgger directories:

```text
1 DAgger round:  [0.50, 0.50]
2 DAgger rounds: [0.50, 0.25, 0.25]
3 DAgger rounds: [0.50, 0.167, 0.167, 0.166]
```

These ratios correspond to `[expert, dagger-001, dagger-002, ...]` and are timestep sampling ratios, not trajectory counts.

## Round Metadata

Record the minimum information required to understand each round in `metrics.json`:

```json
{
  "rounds": [
    {
      "round": 1,
      "beta": 0.5,
      "data_dirs": [
        "data/expert/rand-100",
        "data/flywheel/run-001/round-001/dagger"
      ],
      "best_checkpoint": "checkpoints/flywheel/run-001/round-001/model_epoch_0010.pt",
      "best_score": 0.82
    }
  ]
}
```

This structure preserves dataset provenance without requiring a database or duplicated data.
