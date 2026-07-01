# Architecture Evolution Plan

Start simple, make each step runnable end-to-end, and only add complexity after there are useful metrics.

## Stage 0: Expert and Dataset

Goal: create the robot learning substrate.

```text
scripted expert -> demos: obs_t, action_t, success, phase
```

Build:

- Demo collector from `PickPlaceController`.
- Dataset loader.
- Replay script.
- Success metrics.

Covers:

- Robot learning from demonstration.
- Data flywheel basics.
- Demonstration curation.

Key lessons:

- What makes a good demo.
- Why dataset quality matters more than model complexity.
- How to define observations and actions cleanly.

## Stage 1: Low-Dimensional Behaviour Cloning

Goal: first real PyTorch policy.

```text
[qpos, qvel, gripper, cube pose, tray pose, ee pose] -> MLP -> ctrl
```

Model:

- 3-5 layer MLP.
- MSE or L1 loss on expert `ctrl`.
- Closed-loop MuJoCo evaluation.

Deliverables:

- `train_bc.py`.
- `eval_bc.py`.
- Success rate table.
- Predicted vs expert action plots.

Covers:

- PyTorch fundamentals.
- Imitation learning.
- Closed-loop evaluation.

Key lesson: low training loss does not imply successful rollouts because of covariate shift.

## Stage 2: Better State-Based Policies

Goal: learn stronger robot policy design patterns.

Try these variants:

```text
MLP + phase input
MLP + relative observations
MLP + action delta
MLP + temporal history
```

Recommended progression:

1. Absolute state -> absolute `ctrl`.
2. Relative state -> absolute `ctrl`.
3. Relative state -> delta action.
4. History stack -> action.

Useful observations:

- Gripper-to-cube relative pose.
- Cube-to-tray relative pose.
- Current end-effector pose.
- Previous action.
- Phase one-hot.

Covers:

- Feature engineering for robotics.
- Contact and grasp phase sensitivity.
- Policy failure debugging.

Key lessons:

- Why relative observations generalize better.
- Why action smoothing matters.
- Why contact policies fail near grasp and release.

## Stage 3: DAgger-Style Correction

Goal: handle distribution shift.

```text
learned policy rollout -> states visited by learner -> expert labels those states -> retrain
```

Implementation:

- Roll out learned policy.
- At every state, query the scripted expert for corrective action.
- Append `(learner_state, expert_action)` to the dataset.
- Retrain.
- Compare behaviour cloning vs DAgger success.

Covers:

- Imitation learning beyond naive behaviour cloning.
- Data flywheel.
- Online correction.
- Robot learning from demonstration.

JD mapping:

- DAgger-style online correction.
- Demonstration curation.
- Debugging why policy fails.

## Stage 4: Sequence Models and Action Chunking

Goal: move toward modern robot learning architectures.

```text
obs history -> sequence model -> action chunk
```

Start with an ACT-style policy:

```text
history of observations -> transformer/MLP mixer -> next K actions
```

Simpler first version:

- Input: last `H` observations.
- Output: next `K` actions.
- Execute the first action or average overlapping chunks.

Why this matters:

- Many modern robot policies predict action chunks.
- It improves temporal consistency.
- It maps toward ACT, Diffusion Policy, and Octo-like setups.

Experiment:

| Model | Input | Output | Metric |
| --- | --- | --- | --- |
| MLP BC | obs | action | success % |
| History MLP | obs history | action | success % |
| ACT-lite | obs history | action chunk | success % |

## Stage 5: Vision-Based Policy

Goal: move from privileged state to perception.

```text
camera image + proprioception -> CNN/ViT encoder -> MLP/action head -> ctrl
```

Start simple:

- Use MuJoCo camera images.
- Add a CNN encoder.
- Concatenate image embedding with robot state.
- Predict action.

Then improve:

- Use multiple cameras.
- Add depth or segmentation if available.
- Add image augmentation.
- Compare image-only vs image-plus-state.

Covers:

- Vision-conditioned robotics.
- VLA preparation.
- Sim-to-real perception issues.

JD mapping:

- VLA models.
- Contact-rich manipulation.
- Open-source VLA implementations.

## Stage 6: VLA-Lite Interface

Goal: make the project structurally compatible with VLA models.

```text
image + language instruction + proprioception -> policy -> action
```

Instruction examples:

- Pick the red cube and place it in the blue tray.
- Move cube to tray.
- Grasp cube.

First create the data interface:

```text
{
  "image": camera image,
  "state": robot proprioception,
  "language": task instruction,
  "action": ctrl or delta ee action
}
```

Then try:

- Frozen CLIP/text encoder plus policy head.
- Frozen vision encoder plus policy head.
- Small transformer fusion model.

Covers:

- VLA data formatting.
- Multimodal policy inputs.
- Foundation-model-style robotics interfaces.

Interview framing:

> I built the pipeline so it can move from state behaviour cloning to vision-language-action policies without changing the simulator or evaluation loop.

## Stage 7: Diffusion Policy

Goal: cover a current strong robot imitation learning method.

```text
obs/image history -> diffusion model -> action trajectory
```

Scope:

- Predict action chunks, not single actions.
- Start with low-dimensional observations.
- Then add image conditioning.

Covers:

- Advanced imitation learning.
- Generative action modeling.
- Research-codebase modification.

Practical note: Diffusion Policy is heavier than ACT-lite, so do this after the MLP and sequence baselines.

## Stage 8: World Model

Goal: cover the JD's world model requirement.

```text
obs_t + action_t -> predict obs_{t+1}
```

Start simple:

- Predict next cube pose.
- Predict next end-effector pose.
- Predict success or failure risk.

Then advanced:

```text
latent encoder(obs) -> latent dynamics -> decoder/predictor
```

Use cases:

- Anomaly detection.
- Rollout prediction.
- Offline policy evaluation.
- Detect likely failure before deployment.

Covers:

- World models.
- Predictive dynamics.
- Anomaly detection.

JD mapping:

- Surface state prediction.
- World models.
- Offline evaluation metrics.

## Stage 9: Safety Layer

Goal: mimic a production architecture.

```text
learned policy proposes action -> deterministic safety layer filters action -> robot executes
```

Safety checks:

- Joint limits.
- Max action delta.
- Workspace bounds.
- Gripper constraints.
- Collision and contact heuristics.
- Abort conditions.
- Fallback to scripted controller.

Covers:

- Production-safe learned components.
- Learned plus classical hybrid control.
- Deployment mindset.

JD mapping:

- Learned policies propose actions and deterministic safety layers enforce constraints.

## Stage 10: Deployment Profiling

Goal: show deployment awareness even without real robot hardware.

```text
PyTorch -> ONNX -> benchmark latency -> optional TensorRT
```

Measure:

- Inference latency.
- Policy frequency.
- Memory usage.
- Rollout degradation with artificial action delay.
- Effect of action smoothing.

Covers:

- Edge deployment thinking.
- Latency profiling.
- Real-time constraints.

JD mapping:

- TensorRT.
- Latency profiling.
- Why 200 ms policy latency hurts contact control.

## Recommended Order

1. Demo collection.
2. State behaviour cloning MLP.
3. Closed-loop evaluation.
4. Relative observations plus phase conditioning.
5. DAgger.
6. Action chunking or ACT-lite.
7. Vision-conditioned policy.
8. VLA-lite data interface.
9. Diffusion Policy.
10. World model.
11. Safety layer.
12. ONNX/TensorRT-style profiling.

## Highest-ROI Items For The JD

1. State behaviour cloning plus closed-loop evaluation.
2. DAgger.
3. Vision-conditioned policy.
4. ACT-lite or Diffusion Policy.
5. Safety wrapper.
6. Latency benchmark.
7. World model mini-project.

## Portfolio Story

> I built a MuJoCo pick-place robot learning stack from scratch: scripted expert demonstrations, PyTorch behaviour cloning, DAgger correction, sequence/action-chunking policies, vision-conditioned policies, VLA-compatible data formatting, world-model prediction, deterministic safety filtering, and deployment latency profiling.

This covers most of the JD except actual real-hardware deployment. If a small part can later run on a real arm or Jetson, the project becomes much stronger.
