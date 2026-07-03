# Minimal Raw-MSE Inference Plan

1. Update `scripts/train.py` just enough to save a checkpoint.
   - Add `--checkpoint-path`, default like `checkpoints/bc_mlp.pt`.
   - After training, create the parent directory and save `model.state_dict()`.
   - Keep architecture unchanged: `MLP(obs_dim=40, action_dim=8)`.
   - Keep raw observations/actions unchanged.

2. Add `scripts/infer.py`.
   - Import `MLP` from `train.py`.
   - Import `SimEnv` and `DataCollector`.
   - Load checkpoint with `torch.load(..., map_location=device)`.
   - Instantiate `MLP(obs_dim=40, action_dim=8)`.
   - Call `model.load_state_dict(...)`, then `model.eval()`.

3. Reuse the existing observation builder.
   - Create `collector = DataCollector(sim=sim)`.
   - Every step:
     - `obs = collector.build_observation()`.
     - Convert to `torch.float32` and add a batch dimension.
     - Run the model under `torch.no_grad()`.
     - Convert output to a NumPy action with shape `(8,)`.

4. Apply predicted action directly to MuJoCo controls.
   - `data.ctrl[:7] = action[:7]`.
   - `data.ctrl[7] = np.clip(action[7], 0.0, 255.0)`.
   - Also clamp arm controls to actuator ctrl ranges from `model.actuator_ctrlrange` if enabled, to prevent wild predictions from destabilizing sim.

5. Add real-time MuJoCo viewer mode.
   - Use `mujoco.viewer.launch_passive`, copied from `run_pick_place.py`.
   - Use the same camera setup.
   - Use the same `q` key close behavior.
   - Loop while the viewer is running and `step < max_steps`.
   - Run policy, step sim, sync viewer, and sleep `model.opt.timestep * slowdown`.

6. Add minimal diagnostics.
   - CLI args:
     - `--checkpoint-path`.
     - `--max-steps`, default `8000`.
     - `--slowdown`, default maybe `10.0`.
     - `--headless`.
   - Print every N steps:
     - step.
     - cube position.
     - tray XY error.
     - gripper command.
   - At the end, print max cube lift and final tray error.

7. Verification sequence.
   - Train: `uv run python scripts/train.py --num_epochs 500 --checkpoint-path checkpoints/bc_mlp.pt`.
   - Headless rollout: `uv run python scripts/infer.py --checkpoint-path checkpoints/bc_mlp.pt --headless`.
   - Viewer rollout: `uv run python scripts/infer.py --checkpoint-path checkpoints/bc_mlp.pt`.

This keeps the first version intentionally simple: no normalization, no validation split, no metadata schema, no action-loss weighting, and no model registry.
