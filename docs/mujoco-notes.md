## Important notes for the MuJoCo simulator

### `mj_forward` vs `mj_step`

| Function | Advances time? | Use case |
|---|---|---|
| `mj_forward` | No - computes derived state from the current state without integration | After load/reset to put `mjData` in a valid state |
| `mj_step` | Yes - computes dynamics and integrates one simulation timestep | Normal simulation stepping |

Use `mj_forward` after manually setting or resetting state, such as after `mj_resetDataKeyframe`. It updates derived fields like body positions, site positions, contacts, and actuator quantities so they match the current `qpos`/`qvel`/`ctrl`, but it does not move the simulation forward.

Use `mj_step` only when you want physics to advance. Calling it in reset code would make the returned state be "one timestep after reset" rather than the exact reset state.

### `qpos` vs `xpos`

Both `qpos` and `xpos` live in `mjData`, not `mjModel`.

`mjModel` stores the static compiled model: body definitions, joint definitions, sizes like `nq`/`nbody`, geometry, masses, limits, and other parameters loaded from XML. `mjData` stores the mutable simulation state and derived runtime quantities.

| Field | Size | Meaning | Writable? |
|---|---|---|---|
| `data.qpos` | `model.nq` | Generalized positions: joint angles, sliders, free-body poses | Yes - set state here |
| `data.xpos` | `(model.nbody, 3)` | Cartesian world-frame position of each body | No - derived from `qpos` via kinematics; overwritten every call |

`qpos` is the source position state. `xpos` is a computed world-space body position. After changing `qpos`, call `mj_forward` before reading `xpos`.

### Why `nq == 16` in `panda_pick_place.xml`

Only joints contribute entries to `qpos`. Fixed bodies still have world positions, but their positions are derived from their parent body and are not independent state variables.

The current scene has:

| Model element | Joint type | Count | `qpos` entries each | Total `qpos` entries |
|---|---|---:|---:|---:|
| Panda arm joints `joint1` ... `joint7` | Hinge | 7 | 1 | 7 |
| Gripper joints `finger_joint1`, `finger_joint2` | Slide | 2 | 1 | 2 |
| Cube | Freejoint | 1 | 7 | 7 |
| Table, tray, floor, sites, geoms, lights | Fixed / no joint | - | 0 | 0 |

Total:

```text
7 arm hinge joints + 2 finger slide joints + 7 cube freejoint entries = 16
```

The scene keyframe matches that layout:

```xml
qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04 0.54 -0.12 0.815 1 0 0 0"
```

Breakdown:

```text
1-7:   Panda arm hinge joint angles
8-9:   Gripper slide joint positions
10-12: Cube freejoint position: x y z
13-16: Cube freejoint orientation quaternion: qw qx qy qz
```

The tray and table do not appear in `qpos` because they have no joints in the XML. Their `pos` attributes define fixed offsets from the world body.

### `nq`, `nv`, and `nu`

| Field | Runtime array | Current size | Meaning | Current layout / use |
|---|---|---:|---|---|
| `model.nq` | `data.qpos` | 16 | Generalized position coordinates | 7 arm joints, 2 gripper joints, 7 cube freejoint coordinates |
| `model.nv` | `data.qvel` | 15 | Generalized velocity coordinates | 7 arm velocities, 2 gripper velocities, 6 cube freejoint velocities |
| `model.nu` | `data.ctrl` | 8 | Actuator control inputs | `ctrl[0:7]` arm commands, `ctrl[7]` gripper command; record full actions with `data.ctrl[: model.nu].copy()` |
