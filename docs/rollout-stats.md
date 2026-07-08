# Rollout Stats

The rollout path now supports saving per-step diagnostics from policy inference so model outputs can be inspected after a viewer run.

## Log Location

Rollout logs are written under:

```text
logs/rollouts/<timestamp>_<checkpoint-name>/
```

Each rollout run contains:

```text
metadata.json
episode_000000.npz
episode_000001.npz
...
```

`metadata.json` records the checkpoint path, action space, normalization flag, seed, randomization setting, episode count, max steps, and device.

## Episode Arrays

Each `episode_*.npz` stores per-step arrays:

```text
obs
obs_norm
joints_pred_raw
joints_pred_unnorm
arm_qpos
arm_ctrl
gripper_logit
gripper_prob
gripper_ctrl
applied_actions
cube_pos
tray_pos
grasp_pos
```

For `action_space == "joint_delta"`, `joints_pred_unnorm` is the predicted joint delta before adding the current arm qpos. `arm_ctrl` is the final absolute arm control sent to MuJoCo after converting the delta to a target.

For `action_space == "absolute"`, `joints_pred_unnorm` is the predicted absolute arm target and `arm_ctrl` should match it.

## Rollout CLI

Logging is enabled by default:

```bash
uv run python scripts/rollout.py --model checkpoints/.../model.pt
```

It can be disabled with:

```bash
uv run python scripts/rollout.py --model checkpoints/.../model.pt --no-log-rollout
```

The log root can be changed with:

```bash
uv run python scripts/rollout.py --model checkpoints/.../model.pt --log-dir logs/rollouts
```

## Visualization

Rollout logs can be visualized with:

```bash
uv run python scripts/vis_rollout.py --rollout-dir logs/rollouts/<run-dir>
```

Plots are saved under:

```text
plots/rollouts/<run-dir>/
```

Per-episode plots include:

```text
delta_timeseries_all_joints.png
delta_abs_max.png
qpos_vs_ctrl.png
gripper.png
distance_metrics.png
task_geometry.png
```

The summary folder includes:

```text
summary/arm_output_percentiles.png
```

## What To Inspect

Use `delta_timeseries_all_joints.png` and `delta_abs_max.png` to check whether predicted arm outputs spike during rollout.

Use `qpos_vs_ctrl.png` to inspect how far the applied arm target is from the current joint position.

Use `gripper.png` to check whether the gripper probability is near the threshold or switches too early.

Use `distance_metrics.png` to relate action behavior to task progress: grasp-to-cube distance, cube-to-tray XY distance, and cube height.

For joint-delta experiments, the most important diagnostic is whether `joints_pred_unnorm` stays in the same scale as the expert delta distribution:

```text
expert_delta = actions[:, :7] - obs[:, :7]
```

If rollout deltas frequently exceed the expert delta range, add delta clipping before converting deltas to absolute controls.
