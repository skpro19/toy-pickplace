"""Run a scripted pick-place demo from cube grasp to tray release."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from expert import (
    CUBE_LIFT_MIN_DELTA,
    GRIPPER_ACTUATOR,
    Phase,
    PickPlaceController,
    TRAY_PLACE_TOL,
)
from sim import SimEnv

VIEWER_SLOWDOWN = 10.0


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    max_steps: int,
) -> tuple[Phase, PickPlaceController]:
    controller = PickPlaceController(model, data)
    for _ in range(max_steps):
        controller.control()
        mujoco.mj_step(model, data)
        controller.max_cube_z = max(
            controller.max_cube_z,
            float(data.body("cube").xpos[2]),
        )
        controller.update_phase()
        if controller.phase == Phase.DONE:
            break
    return controller.phase, controller


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    slowdown: float = VIEWER_SLOWDOWN,
) -> None:
    controller = PickPlaceController(model, data)
    viewer_handle: mujoco.viewer.Handle | None = None

    def key_callback(keycode: int) -> None:
        if chr(keycode).lower() == "q" and viewer_handle is not None:
            viewer_handle.close()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer_handle = viewer
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.45, 0.0, 0.78)
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        while viewer.is_running():
            print(f"Phase: {controller.phase}")
            controller.control()
            mujoco.mj_step(model, data)
            controller.update_phase()
            viewer.sync()
            time.sleep(model.opt.timestep * slowdown)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pick-place demo controller.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer (default opens viewer).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8000,
        help="Step limit for headless runs.",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=VIEWER_SLOWDOWN,
        help="Viewer pacing multiplier (>1 runs slower than real time).",
    )
    args = parser.parse_args()

    sim = SimEnv()
    model = sim.model
    data = sim.data
    initial_cube_z = sim.reset_episode()

    if args.headless:
        final_phase, controller = run_headless(model, data, args.max_steps)
        grasp_pos = data.site_xpos[model.site("grasp").id]
        cube_pos = data.body("cube").xpos
        tray_pos = data.site_xpos[model.site("tray_center").id]
        cube_lift = controller.max_cube_z - initial_cube_z
        tray_error = float(np.linalg.norm(cube_pos[:2] - tray_pos[:2]))
        print(f"Final phase: {final_phase.name}")
        print(f"Grasp site: {grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}")
        print(f"Cube center: {cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}")
        print(f"Max cube lift: {cube_lift:.3f} m")
        print(f"Tray XY error: {tray_error:.3f} m")
        print(f"Gripper ctrl: {data.ctrl[GRIPPER_ACTUATOR]:.1f}")
        print(
            f"Sim tracking error: pos={controller.last_pos_err:.4f} m, "
            f"ori={controller.last_ori_err:.4f} rad"
        )
        if final_phase != Phase.DONE:
            raise SystemExit(f"Controller did not finish within {args.max_steps} steps.")
        if cube_lift < CUBE_LIFT_MIN_DELTA:
            raise SystemExit("Cube was not lifted by the gripper.")
        if tray_error > TRAY_PLACE_TOL:
            raise SystemExit("Cube was not placed inside the tray.")
        print("Pick-place demo slice completed.")
    else:
        run_viewer(model, data, slowdown=args.slowdown)


if __name__ == "__main__":
    main()
