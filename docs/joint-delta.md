# Joint Delta Action Space

This note describes the high-level changes needed to switch the learned arm action representation from absolute joint targets to joint deltas.

## High-Level Change

The current policy predicts absolute arm controls:

```text
actions[t, :7] = expert_ctrl[:7]
```

The delta-action version trains the model to predict a target relative to the current arm joint positions:

```text
delta_actions[t, :7] = expert_ctrl[:7] - obs[t, :7]
```

At inference, the predicted delta is converted back into the absolute MuJoCo control target:

```text
pred_delta = model(obs)
ctrl[:7] = current_qpos[:7] + clipped(pred_delta)
```

The gripper can stay as a binary open/close classifier.

## Why This Helps

Joint deltas do not remove closed-loop observation mismatch. If the policy sees states that are very different from training, it can still fail.

The benefit is that each arm command is local to the current physical state. A bad absolute prediction can command a large target jump. A clipped delta prediction limits how far the next control target can move from the current joint configuration.

This changes the model from predicting:

```text
go to this exact joint target
```

to predicting:

```text
move a bounded amount from where the arm is now
```

That can slow down closed-loop drift and make failures less explosive, especially when the policy is only slightly or moderately off the expert trajectory.

## Files Likely Affected

- `scripts/dataset.py`
- `scripts/train.py`
- `scripts/rollout.py`
- `scripts/replay_npz.py`
- `scripts/vis_actions.py`
- `scripts/data.py`
- `scripts/constant.py`
- related docs such as `docs/policy-robustness.md` and `docs/mlp-experiments.md`

## Dataset And Target Construction

The simplest migration keeps existing `.npz` files unchanged. They continue to store absolute expert controls as the source of truth:

```text
obs:     current observation
actions: absolute expert controls
```

Then training derives delta targets from the loaded arrays:

```text
arm_qpos = obs[:, :7]
expert_arm_ctrl = actions[:, :7]
arm_delta_target = expert_arm_ctrl - arm_qpos
```

This conversion can happen in `scripts/dataset.py` or `scripts/train.py`. Keeping raw `.npz` actions absolute preserves `scripts/replay_npz.py` as a ground-truth expert action replay tool.

## Training Changes

`scripts/train.py` should train the arm head on delta targets instead of absolute controls.

The arm normalization statistics should be computed over deltas:

```text
arm_delta_mean
arm_delta_std
```

The arm loss can remain MSE. The gripper loss can remain `BCEWithLogitsLoss`.

Checkpoint metadata should record the action representation, for example:

```text
action_representation = "joint_delta"
max_arm_delta = ...
arm_delta_mean = ...
arm_delta_std = ...
```

This prevents accidentally running a delta-trained checkpoint as if it predicted absolute controls.

## Rollout Changes

`scripts/rollout.py` needs the main behavior change.

Current rollout applies the unnormalized arm prediction directly as MuJoCo controls:

```text
ctrl[:7] = predicted_absolute_arm_ctrl
```

Delta rollout should instead do:

```text
pred_delta = unnormalize(model_output)
pred_delta = clip(pred_delta, -max_arm_delta, +max_arm_delta)
current_qpos = sim.data.qpos[:7]
arm_ctrl = current_qpos + pred_delta
arm_ctrl = clamp_to_actuator_ctrl_range(arm_ctrl)
ctrl[:7] = arm_ctrl
ctrl[7] = gripper_command
```

The MuJoCo actuator interface does not need to change. The environment can still receive absolute controls; only the model's learned action representation changes.

## Constants

`scripts/constant.py` is a natural place for experiment-level constants such as:

```text
MAX_ARM_DELTA
ACTION_REPRESENTATION
```

Start with a conservative `MAX_ARM_DELTA` and tune based on rollout smoothness and task completion.

## Data Collection

`scripts/data.py` does not need to change if `.npz` files keep storing absolute expert controls.

That is the recommended first version. It keeps demonstrations reusable for both absolute-action and delta-action experiments.

Only change data collection if you explicitly want to store metadata or extra arrays, such as:

```text
actions_abs
actions_delta
```

## Replay And Visualization

If `.npz` files continue storing absolute expert controls, `scripts/replay_npz.py` can remain unchanged.

`scripts/vis_actions.py` can also keep visualizing absolute expert actions. If needed, add a separate delta-target visualization that plots:

```text
actions[:, :7] - obs[:, :7]
```

This is useful for checking whether the learned delta targets are small, smooth, and well-scaled.

## Recommended Migration Path

1. Keep existing `.npz` schema unchanged.
2. Derive arm delta targets during dataset loading or training.
3. Normalize delta targets instead of absolute arm controls.
4. Save checkpoint metadata saying the model predicts joint deltas.
5. In rollout, unnormalize predicted deltas.
6. Clip predicted deltas.
7. Convert deltas back to absolute controls using current `qpos`.
8. Clamp final controls to actuator ranges.
9. Compare rollout against the absolute-action baseline on exact saved training initial positions.

## Important Caveat

Delta actions mainly improve control stability. They do not solve every behavior-cloning failure mode.

If the model cannot infer the task phase, add phase labels, timestep input, or observation history.

If the model sees states far outside the expert demonstrations, collect recovery data or use DAgger-style expert relabeling.
