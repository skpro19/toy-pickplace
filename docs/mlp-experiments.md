# MLP Experiments

## Key Constraints

Training on one episode can test memorization and closed-loop robustness, but not generalization. The current setup has three major bottlenecks:

1. `train.py` default is `--epochs 1`, far too low to overfit one episode.
2. Action dim 7 is gripper command `0/255`, so raw `MSELoss` is dominated by the gripper scale.
3. The policy is partially observable: phase/time is not in the observation, but the expert has hidden phase state. Similar observations may require different actions, especially open/close/settle phases.

The one-episode dataset currently has:

```text
obs:     (1260, 40), float32
actions: (1260, 8),  float32
gripper action values: 0.0 or 255.0
```

## Experiment Ladder

| Stage | Goal | Change | Success Criterion |
| --- | --- | --- | --- |
| 0 | Prove data/model pipeline works | Train current MLP for many epochs | Near-zero train MSE |
| 1 | Fix scaling | Normalize obs/actions or weight action dims | Arm + gripper both fit |
| 2 | Improve closed loop | Add action clipping/smoothing in inference | Stable rollout without exploding controls |
| 3 | Push MLP capacity | Wider/deeper MLP, residual MLP, LayerNorm | Lower rollout drift |
| 4 | Fix partial observability | Add timestep/phase/history input | Better gripper timing and phase transitions |
| 5 | Stress test | Run headless closed-loop eval with metrics | Cube lifted/placed, final error measured |

## Recommended Experiments

### 1. Overfit Baseline MLP

Use the current architecture, but train hard:

```bash
uv run python scripts/train.py --epochs 5000 --checkpoint_dir checkpoints/overfit_mlp_baseline
uv run python scripts/infer.py --model checkpoints/overfit_mlp_baseline/model.pt
```

Expected result: training loss should get very small. If closed-loop still fails, the issue is not one-step fitting; it is compounding error or missing state.

### 2. Normalize Observations And Actions

Save `obs_mean`, `obs_std`, `action_mean`, and `action_std` in the checkpoint. Train on normalized targets, then unnormalize predictions in `infer.py`.

Why: right now gripper values `0/255` dominate MSE compared to arm joint targets around `-2.5..3.0`.

Expected result: arm action fitting improves substantially.

### 3. Split Arm Regression And Gripper Classification

Instead of predicting all 8 action dims with one MSE:

1. Predict arm controls: 7 outputs with MSE.
2. Predict gripper state: 1 binary logit with BCE.
3. Convert gripper prediction to `0.0` or `255.0` during inference.

This directly matches the data: gripper is binary, not continuous.

Expected result: cleaner gripper behavior and less loss interference.

### 4. Add Time Or Progress Input

Because this is one fixed episode, add normalized timestep `t / T` to the observation.

Current `obs_dim=40`; make it `41`.

This helps the model distinguish phase-dependent actions when physical observations are similar.

Expected result: much better memorization of open/close timing.

### 5. Add Observation History

Use last `k` observations instead of one observation:

```text
input_dim = 40 * k
k = 2, 4, 8
```

This gives velocity/trend information without changing simulator state.

Experiment sweep:

```text
k = 1 baseline
k = 2
k = 4
k = 8
```

Expected result: smoother arm commands and better phase transition behavior.

### 6. MLP Capacity Sweep

Try these architectures:

```text
small:    40 -> 128 -> 128 -> 128 -> 8
medium:   40 -> 256 -> 256 -> 256 -> 256 -> 8
large:    40 -> 512 -> 512 -> 512 -> 512 -> 8
residual: input -> 256 blocks with LayerNorm/ReLU skip connections
```

For one episode, overfitting capacity is fine. Use no dropout.

Expected result: larger models should reduce one-step error, but may not fix closed-loop failure unless normalization/history/time is added.

### 7. Closed-Loop Evaluation Script

`infer.py` currently only launches viewer and runs forever. Add a headless eval mode that reports:

```text
max cube lift
final cube position
tray XY error
min distance grasp-to-cube
whether cube reached tray
number of steps survived
```

This lets experiments be compared without judging visually.

Suggested command shape:

```bash
uv run python scripts/infer.py --model checkpoints/.../model.pt --headless --max-steps 8000
```

## Recommended Experiment Order

1. Current MLP, `5000` epochs.
2. Add obs/action normalization.
3. Split gripper into classification.
4. Add timestep input.
5. Add history input with `k=4`.
6. Sweep MLP size.
7. Only then try residual/LayerNorm MLP.

## Most Likely Winning Setup

For one-episode closed-loop imitation, the most promising setup is:

```text
input: normalized obs history, k=4, plus normalized timestep
model: 4-layer MLP, width 256 or 512
arm loss: MSE on normalized 7-dim arm action
gripper loss: BCE binary classifier
inference: unnormalize arm, threshold gripper to 0/255, clip actions to valid range
```

This still keeps the model MLP-based but removes the biggest avoidable limits in the current pipeline.
