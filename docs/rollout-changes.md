# Rollout Changes

`scripts/rollout.py` should evolve from a single viewer-driven policy demo into a closed-loop evaluation script for learned policies.

## Goal

Evaluate a trained behavior cloning model on the same randomized scene distribution used during data collection, then use the same script for held-out randomized evaluation with a different seed.

## Distribution Controls

The rollout script should expose the same environment controls as data collection:

- `--seed`
- `--randomize-scene` / `--no-randomize-scene`
- `--episodes`
- `--max-steps`

`SimEnv` should be constructed with the requested randomization settings so every episode reset samples from the configured distribution.

## Episode Loop

Rollout should support more than one closed-loop episode.

Each episode should:

- reset the environment
- run the policy in closed loop
- stop at success, failure, viewer quit, or `max_steps`
- record episode-level metrics

Viewer mode can remain useful for qualitative inspection, but headless mode should be the primary evaluation path for many episodes.

## Evaluation Metrics

At minimum, closed-loop evaluation should report:

- final timestep
- whether the cube was lifted
- whether the cube dropped before placement
- final cube-to-tray XY error
- whether the episode succeeded
- gripper switch behavior

Useful later metrics:

- actuator range violations
- maximum actuator violation magnitude
- maximum joint velocity
- raw policy action statistics

## Output

The script should print a concise per-episode summary and an aggregate summary across all episodes.

Aggregate summary should include:

- number of episodes
- success count and rate
- mean final cube-to-tray XY error
- mean episode length
- failure reasons, if tracked

Saving metrics to disk can be added later once the console summary is reliable.

## Seed Usage

Using the same seed as data collection is useful for debugging because it reproduces the same randomized scene sequence.

For generalization evaluation, use a different seed so the model is tested on held-out randomized layouts from the same distribution.

Recommended pattern:

- `seed=0`: reproduce/debug training-distribution rollouts
- `seed=1` or higher: held-out randomized evaluation

## Viewer Versus Headless

Viewer mode should be used for visual diagnosis of a small number of episodes.

Headless mode should be used for actual evaluation because it can run many episodes quickly and produce stable metrics.

## Success Criteria

The rollout script is ready for closed-loop evaluation when it can:

- run multiple randomized episodes from a trained checkpoint
- evaluate with the same randomization API as `scripts/data.py`
- produce per-episode and aggregate success metrics
- support both visual debugging and headless evaluation
