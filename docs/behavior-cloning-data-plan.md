# Behaviour Cloning Data Collection Plan


1. Refactor scripted controller
   - Keep `PickPlaceController` as the expert policy.
   - Add a separate data collection script, e.g. `scripts/collect_pick_place_demos.py`.
   - Reuse `reset_home`, `run_headless`, phase logic, and MuJoCo scene loading.

**[IGNORE]**
2. Randomize initial conditions
   - Randomize cube pose within the reachable workspace.
   - Optionally randomize tray pose slightly.
   - Keep randomization modest at first so the scripted controller succeeds reliably.
   - Save the random seed per episode.

3. Record demonstrations
   - Log `qpos`, `qvel`, and `ctrl` at every simulation step.
   - Log the end-effector pose from the `grasp` site.
   - Log cube pose and velocity.
   - Log tray pose.
   - Log the current `Phase`.
   - Log the action target, preferably the next full `ctrl` vector.
   - Save success or failure metadata for each episode.

4. Define observation and action spaces
   - Observation: robot joint positions and velocities, gripper state, cube pose, tray pose, and end-effector pose.
   - Action: full MuJoCo `ctrl`, i.e. 7 arm position targets plus gripper command.
   - This matches the existing controller and is the simplest baseline for behaviour cloning.

**[IGNORE]**
5. Filter successful episodes
   - Run each episode headless.
   - Initially save only successful episodes for training.
   - Success criteria: final phase is `DONE`, cube lift exceeds `CUBE_LIFT_MIN_DELTA`, and tray XY error is below `TRAY_PLACE_TOL`.
   - Store failed attempts separately only for debugging.

6. Store the dataset
   - Use compressed NumPy files, for example `data/demos/pick_place_000001.npz`.
   - Each file should contain `obs`, `actions`, `qpos`, `qvel`, `ctrl`, `phase`, `success`, and `seed`.
   - Later, switch to HDF5 or Zarr if the dataset becomes large.

7. Add a collection CLI
   - Support `--episodes`.
   - Support `--out-dir`.
   - Support `--seed`.
   - Support `--max-steps`.
   - Support `--save-failures`.
   - Support `--randomize-cube`.
   - Support `--randomize-tray`.

8. Run sanity checks
   - Collect 10 demonstrations first.
   - Replay saved `ctrl` trajectories in simulation.
   - Verify replay success rate.
   - Plot episode lengths, phase durations, and action ranges.

9. Train a baseline policy
   - Start with a simple MLP policy.
   - Supervise `obs -> ctrl`.
   - Split train and validation sets by episode, not by timestep.
   - Evaluate closed-loop in MuJoCo with randomized starts.

10. Iterate if needed
    - Add more demonstrations around grasping and contact-heavy phases.
    - Consider a phase-conditioned policy.
    - Add action smoothing if predictions are noisy.
    - Prefer relative observations such as cube relative to gripper and tray relative to cube.
    - Use DAgger-style correction later if pure behaviour cloning is insufficient.

## First Implementation Step

Create `scripts/collect_pick_place_demos.py` that imports the existing controller, runs randomized successful headless episodes, and writes `.npz` trajectory files.
