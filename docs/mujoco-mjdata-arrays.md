# MuJoCo `MjData` Arrays Reference

This document describes the `MjData` fields most useful for control, demo collection, and behaviour cloning in this repo. It is scoped to the `panda_pick_place` scene (`scenes/panda_pick_place.xml` via `SimEnv`).

`MjData` exposes 100+ arrays. Most are internal solver state (`efc_*`, `qH*`, `solver_*`, etc.). For day-to-day work, focus on the arrays below.

## Quick counts (this scene)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `model.nq` | 16 | Generalized positions |
| `model.nv` | 15 | Generalized velocities |
| `model.nu` | 8 | Actuators (7 arm + gripper) |
| `model.nbody` | 16 | Bodies (world, panda links, cube, tray, …) |
| `model.nsite` | 8 | Named sites (grasp, cube targets, tray, …) |
| `model.ngeom` | 89 | Geoms (meshes, boxes, floor, …) |

Access via `sim = SimEnv()` → `data = sim.data`, `model = sim.model`.

---

## Primary state vectors

These are the main signals for logging demos and building observations.

### `data.qpos` — generalized positions

- **Shape:** `(nq,)` → `(16,)` for this scene
- **Updated:** every `mj_step` / `mj_forward`
- **Use:** robot configuration, free-object pose, episode reset checks

**Index layout (`panda_pick_place`):**

| Index | Joint | Type | Content |
|-------|-------|------|---------|
| `0–6` | `joint1`–`joint7` | hinge | Arm joint angles (rad) |
| `7` | `finger_joint1` | slide | Left finger position |
| `8` | `finger_joint2` | slide | Right finger position |
| `9–11` | cube free joint | free | Cube position `[x, y, z]` |
| `12–15` | cube free joint | free | Cube orientation quaternion `[w, x, y, z]` |

At `task_home`, `qpos` is:

```text
[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04, 0.54, -0.12, 0.815, 1, 0, 0, 0]
```

**Named access:** cube pose is also available as `data.body("cube").xpos` / `.xquat` (world frame, after `mj_forward`).

---

### `data.qvel` — generalized velocities

- **Shape:** `(nv,)` → `(15,)` for this scene
- **Use:** object velocity, contact/transient dynamics, optional observation feature

**Index layout:**

| Index | DOF | Content |
|-------|-----|---------|
| `0–6` | arm hinges | Joint angular velocities |
| `7–8` | finger slides | Finger linear velocities |
| `9–11` | cube free | Linear velocity `[vx, vy, vz]` |
| `12–14` | cube free | Angular velocity `[wx, wy, wz]` |

---

### `data.ctrl` — actuator commands (your action space)

- **Shape:** `(nu,)` → `(8,)`
- **You write this** before each `mj_step`; it is the expert/BC action vector
- **Use:** primary action for behaviour cloning

| Index | Actuator | Role |
|-------|----------|------|
| `0–6` | `actuator1`–`actuator7` | Arm position targets (rad) |
| `7` | `actuator8` | Gripper command (`255` = open, `0` = closed) |

Reference: `scripts/expert.py` (`GRIPPER_ACTUATOR = 7`, `GRIPPER_OPEN = 255.0`).

---

## World-frame kinematics

Computed by `mj_forward` / `mj_step`. Prefer named accessors when possible.

### `data.xpos` / `data.xquat` / `data.xmat` — body pose in world frame

| Field | Shape | Content |
|-------|-------|---------|
| `xpos` | `(nbody, 3)` | Body origin world position |
| `xquat` | `(nbody, 4)` | Body orientation quaternion `[w,x,y,z]` |
| `xmat` | `(nbody, 9)` | Body orientation as 3×3 rotation matrix (row-major) |

**Named access (preferred):**

```python
data.body("cube").xpos    # shape (3,)
data.body("cube").xquat   # shape (4,)
data.body("hand").xpos
data.body("tray").xpos
```

**Task-relevant bodies:** `link0` … `link7`, `hand`, `left_finger`, `right_finger`, `cube`, `tray`, `table`.

Each body view also exposes `xmat`, `cvel`, `cfrc_ext`, `cfrc_int`, and subtree COM/velocity fields for dynamics debugging.

---

### `data.site_xpos` / `data.site_xmat` — site pose in world frame

| Field | Shape | Content |
|-------|-------|---------|
| `site_xpos` | `(nsite, 3)` | Site world position |
| `site_xmat` | `(nsite, 9)` | Site orientation (3×3 matrix) |

**Sites in this scene:**

| Index | Name | Typical use |
|-------|------|-------------|
| `0` | `mount_center` | Robot base reference |
| `1` | `grasp` | End-effector pose (IK / EE observation) |
| `2` | `table_center` | Table surface reference |
| `3` | `cube_center` | Cube body frame origin |
| `4` | `cube_hover` | Approach target above cube |
| `5` | `cube_grasp` | Grasp approach target |
| `6` | `cube_lift` | Lift target after grasp |
| `7` | `tray_center` | Place target |

**Named access:**

```python
data.site("grasp").xpos
data.site_xpos[model.site("grasp").id]   # equivalent
```

Site views expose: `id`, `name`, `xpos`, `xmat`.

---

### `data.geom_xpos` / `data.geom_xmat` — geom pose in world frame

| Field | Shape | Content |
|-------|-------|---------|
| `geom_xpos` | `(ngeom, 3)` | Collision/visual geom world position |
| `geom_xmat` | `(ngeom, 9)` | Geom orientation |

Useful for fine-grained collision debugging. For task logic, prefer `body` or `site` accessors (`cube`, `tray`, `grasp`).

---

## Actuator feedback (post-step)

Read after `mj_step` to inspect what the actuators actually did.

| Field | Shape | Content |
|-------|-------|---------|
| `actuator_force` | `(nu,)` | Force/torque applied by each actuator |
| `actuator_length` | `(nu,)` | Actuator length (transmission coordinate) |
| `actuator_velocity` | `(nu,)` | Actuator velocity |

---

## Generalized forces

Forces in joint space (`nv` DOFs). Useful when debugging grasp forces or constraint fighting.

| Field | Shape | Content |
|-------|-------|---------|
| `qfrc_actuator` | `(nv,)` | Force from actuators mapped to DOFs |
| `qfrc_applied` | `(nv,)` | External applied generalized force |
| `qfrc_constraint` | `(nv,)` | Force from equality/contact constraints |

---

## Body wrenches

Forces and torques on each body (6D wrench per body: 3 force + 3 torque).

| Field | Shape | Content |
|-------|-------|---------|
| `cfrc_ext` | `(nbody, 6)` | External contact/constraints on body |
| `cfrc_int` | `(nbody, 6)` | Internal forces (passive, actuator effects) |

**Named access:** `data.body("cube").cfrc_ext`, `data.body("hand").cfrc_ext`.

---

## Contacts

| Field | Type | Content |
|-------|------|---------|
| `data.ncon` | `int` | Number of active contacts this step |
| `data.contact` | contact list | Per-contact geom pair, position, friction, normal frame |

Each contact entry includes `geom1`, `geom2`, `pos`, `frame`, `dist`, `friction`, etc. Empty at rest with no touch; populated when gripper meets cube or cube meets tray.

---

## Sensors

| Field | Shape | Content |
|-------|-------|---------|
| `sensordata` | `(nsensor,)` | Stacked sensor readings defined in XML |

This scene has `nsensor = 0`; no sensors are defined yet. Add sensors in XML if you want force/touch/proprioception channels here.

---

## Arrays to skip unless debugging physics

These are populated internally by MuJoCo's constraint solver and integrator. Ignore them for BC, control, and demo logging:

- **Constraint solver:** `efc_*`, `iefc_*`
- **Mass matrix / factorization:** `qM`, `qLD`, `qLU`, `qH`, `qDeriv`, `M`
- **Solver stats:** `solver_niter`, `solver_nnz`, `solver_fwdinv`
- **Island/sleeping:** `island_*`, `tree_*`, `body_awake*`, `dof_island`
- **BVH broadphase:** `bvh_*`

---

## Recommended fields for demo logging

Aligned with `docs/behavior-cloning-data-plan.md`:

| Log field | Source | Notes |
|-----------|--------|-------|
| `qpos` | `data.qpos.copy()` | Full 16-dim vector |
| `qvel` | `data.qvel.copy()` | Full 15-dim vector |
| `ctrl` | `data.ctrl.copy()` | 8-dim action |
| EE pose | `data.site("grasp").xpos` (+ optional `xmat`) | 3D position; orientation from `xmat` if needed |
| Cube pose | `data.body("cube").xpos` | Or `qpos[9:16]` |
| Tray pose | `data.site("tray_center").xpos` | Fixed unless tray is randomized |
| Phase | controller state | Not in `MjData`; from `PickPlaceController.phase` |

---

## Quick inspection snippet

Run from the repo root:

```bash
uv run python -c "
from scripts.sim import SimEnv
sim = SimEnv()
m, d = sim.model, sim.data
for name in ('qpos', 'qvel', 'ctrl', 'xpos', 'site_xpos', 'actuator_force'):
    arr = getattr(d, name)
    print(f'{name:16} {arr.shape}')
print('cube xpos:', d.body('cube').xpos)
print('grasp xpos:', d.site('grasp').xpos)
print('ctrl:', d.ctrl)
"
```

Or explore interactively:

```bash
uv run python -i scripts/demos.py
```

---

## Related

- `sim.model` vs `sim.data`: model is static (structure, names, limits); data is dynamic (current state). See `scripts/sim.py`.
- Scene layout and keyframes: `scenes/panda_pick_place.xml`
- Expert action space: `scripts/expert.py`
