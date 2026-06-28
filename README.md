# Toy Pick-Place

Minimal MuJoCo scene setup for a Franka Panda tabletop pick-and-place task.

## Included

- Menagerie `franka_emika_panda` model vendored under `third_party/`
- Custom tabletop scene with a cube, a shallow tray, task sites, and fixed cameras
- Viewer and smoke-test scripts

## Setup

```bash
uv sync
```

## Run the viewer

```bash
uv run python scripts/view_scene.py
```

The scene defines two fixed cameras named `overview` and `workspace`.

## Run a smoke test

```bash
uv run python scripts/smoke_test_scene.py
```
