# Pick-Place Implementation Plan

## Recommended Plan

Use a scripted feedback controller with a small finite-state machine and Jacobian-based IK.

Why this fits this repo:
- The only existing runtime code just holds the home keyframe in `scripts/view_scene.py:55-58`.
- The Panda model exposes 7 arm position-like actuators plus 1 gripper actuator in `third_party/franka_emika_panda/panda.xml:266-279`.
- The scene already gives stable object names and target geometry in `scenes/panda_pick_place.xml:40-55`.

## Key Findings

- `actuator8` is the gripper command, and `255` means open, not closed.
  Reference: `third_party/franka_emika_panda/panda.xml:277-283`
- There is no built-in end-effector site on the Panda model.
  That makes `mj_jacSite` awkward unless we add one.
- The current scene already has `cube_center` and `tray_center` sites, but the hover sites are commented out.
  Reference: `scenes/panda_pick_place.xml:44-55`

## Implementation Shape

1. Add a grasp site on the robot.
   Suggested: a `site` on the `hand` body at the midpoint between fingers, so the controller can use `mj_jacSite`.
2. Add or restore task-space helper sites.
   Suggested: `cube_hover`, `cube_grasp`, `tray_hover`, `tray_drop`.
3. Create a new script, likely `scripts/run_pick_place.py`.
4. Implement a phase-based controller:
   - `move_above_cube`
   - `descend_to_grasp`
   - `close_gripper`
   - `lift`
   - `move_above_tray`
   - `lower_into_tray`
   - `open_gripper`
   - `retreat`
5. For each motion phase, use damped least-squares IK:
   - compute site position error
   - optionally add orientation error to keep the gripper vertical
   - use `mj_kinematics`, `mj_comPos`, `mj_jacSite`
   - update only the first 7 arm controls
6. Drive the gripper separately:
   - open: `data.ctrl[7] = 255`
   - close: `data.ctrl[7] = 0` or a small near-closed value if full closure causes instability
7. Advance phases on thresholds:
   - position tolerance
   - grasp settle time
   - cube lift detected by cube height increase
8. Add a rollout test that steps the controller and asserts the cube ends inside the tray footprint.

## Minimal File Changes

- `scenes/panda_pick_place.xml`
- new controller script under `scripts/`
- likely one new test script, or extend `tests/smoke_test_scene.py`

## Verification

- Run the scripted rollout headless and assert:
  - cube XY is within tray inner bounds
  - cube Z is above tray base and below wall top
- Run the viewer to visually confirm the sequence

## One Design Choice

I recommend adding a dedicated end-effector `site` and using task-space IK. It is cleaner and less brittle than hard-coding joint waypoints.

## Follow-up

The next implementation choice is whether to build this as:

1. A standalone demo script that runs the full pick-and-place automatically.
2. A reusable controller module that the viewer and test scripts can call.

Recommendation: start with `1`, then factor into `2` if needed.
