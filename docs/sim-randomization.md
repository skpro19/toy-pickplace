# Simulation Randomization

The scene now supports reset-time randomization for data collection and visual expert rollouts. The XML still defines the cube, tray, geoms, sites, materials, and default poses; Python only changes episode-specific initial positions.

## Scene Setup

- `scenes/panda_pick_place.xml` keeps the cube and tray bodies in the MuJoCo scene.
- The cube freejoint is named `cube_freejoint` so `SimEnv` can update its pose through `data.qpos`.
- The tray remains a static body; `SimEnv` updates its body position through `model.body_pos` during reset.
- The XML pose remains the deterministic default when randomization is disabled.

## SimEnv Changes

`scripts/sim.py` owns episode initialization through `SimEnv.reset_episode(...)`. Randomization is configured on each `SimEnv` instance so every reset uses the same private, seeded random stream.

Randomized environment setup uses:

```python
sim = SimEnv(randomize_scene=True, seed=0)
sim.reset_episode()
```

Deterministic reset remains:

```python
sim = SimEnv()
sim.reset_episode()
```

Current randomized fields:

- cube XY position
- tray XY position

The sampled layout enforces a minimum cube-tray XY distance so the objects do not start too close together. After applying positions, `mj_forward` updates derived body and site transforms.

## Data Collection

`scripts/data.py` supports randomized multi-episode data collection.

Example:

```bash
uv run scripts/data.py --episodes 10 --randomize-scene --seed 0 --out-dir data/demos/randomized_10
```

Each saved `.npz` includes:

- `obs`
- `actions`
- `cube_init_pos`
- `tray_init_pos`

The script also shows nested `tqdm` progress:

- outer progress over episodes
- inner progress over steps in the current episode
- current expert phase and completed step count

## Visual Rollouts

`scripts/run_pick_place.py` can visually test randomized scenes with automatic episode advancement.

Example:

```bash
uv run scripts/run_pick_place.py --episodes 10 --randomize-scene --seed 0
```

Useful viewer options:

```bash
uv run scripts/run_pick_place.py --episodes 10 --randomize-scene --seed 0 --slowdown 5 --episode-pause 0.5
```

Headless validation also supports multiple randomized episodes:

```bash
uv run scripts/run_pick_place.py --headless --episodes 10 --randomize-scene --seed 0
```

## Recommended Use

- Use `run_pick_place.py` for visual sanity checks of sampled cube/tray layouts.
- Use `data.py` for collecting randomized behavior cloning demonstrations.
- Start with conservative randomization ranges and widen them only after the scripted expert succeeds reliably.
