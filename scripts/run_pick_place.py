"""Run a scripted pick-place demo: approach cube, descend, close gripper."""

from __future__ import annotations

import argparse
import time
from enum import Enum, auto
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"

ARM_DOF = 7
GRIPPER_ACTUATOR = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

HOVER_TOL = 0.02
GRASP_TOL = 0.05
GRASP_SETTLE_STEPS = 150
DAMPING = 0.05
IK_GAIN = 0.5

# Offsets in the parent site/body local frame (cube_center / tray_center).
CUBE_GRASP_OFFSET = np.array([0.0, 0.0, 0.03])
TRAY_HOVER_OFFSET = np.array([0.0, 0.0, 0.085])
TRAY_DROP_OFFSET = np.array([0.0, 0.0, 0.05])


class Phase(Enum):
    MOVE_ABOVE_CUBE = auto()
    DESCEND_TO_GRASP = auto()
    CLOSE_GRIPPER = auto()
    DONE = auto()


def find_reset_key(model: mujoco.MjModel) -> int:
    for name in ("task_home", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return key_id
    return -1


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = find_reset_key(model)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        if model.nu:
            data.ctrl[: model.nu] = model.key_ctrl[key_id, : model.nu]
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def site_target(data: mujoco.MjData, site_id: int, local_offset: np.ndarray) -> np.ndarray:
    site_rot = data.site_xmat[site_id].reshape(3, 3)
    return data.site_xpos[site_id] + site_rot @ local_offset


def clip_arm_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for i in range(ARM_DOF):
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.ctrl[i], lo, hi)


def ik_toward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_id: int,
    target_pos: np.ndarray,
) -> float:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    err = target_pos - data.site_xpos[grasp_id]
    err_norm = float(np.linalg.norm(err))
    if err_norm < 1e-9:
        return err_norm

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, grasp_id)

    j_arm = jacp[:, :ARM_DOF]
    dq = j_arm.T @ np.linalg.solve(
        j_arm @ j_arm.T + DAMPING**2 * np.eye(3),
        err,
    )
    data.ctrl[:ARM_DOF] = data.qpos[:ARM_DOF] + IK_GAIN * dq
    clip_arm_ctrl(model, data)
    return err_norm


class PickPlaceController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.phase = Phase.MOVE_ABOVE_CUBE
        self.settle_steps = 0
        self.grasp_id = model.site("grasp").id
        self.cube_center_id = model.site("cube_center").id
        self.cube_hover_id = model.site("cube_hover").id

    def target_for_phase(self) -> np.ndarray | None:
        if self.phase in (Phase.MOVE_ABOVE_CUBE,):
            return self.data.site_xpos[self.cube_hover_id].copy()
        if self.phase in (Phase.DESCEND_TO_GRASP, Phase.CLOSE_GRIPPER):
            return site_target(self.data, self.cube_center_id, CUBE_GRASP_OFFSET)
        return None

    def step(self) -> None:
        if self.phase == Phase.DONE:
            return

        if self.phase == Phase.CLOSE_GRIPPER:
            self.data.ctrl[GRIPPER_ACTUATOR] = GRIPPER_CLOSE
            target = self.target_for_phase()
            if target is not None:
                ik_toward(self.model, self.data, self.grasp_id, target)
            self.settle_steps += 1
            if self.settle_steps >= GRASP_SETTLE_STEPS:
                self.phase = Phase.DONE
            return

        self.data.ctrl[GRIPPER_ACTUATOR] = GRIPPER_OPEN
        target = self.target_for_phase()
        if target is None:
            return

        err_norm = ik_toward(self.model, self.data, self.grasp_id, target)
        if self.phase == Phase.MOVE_ABOVE_CUBE and err_norm < HOVER_TOL:
            self.phase = Phase.DESCEND_TO_GRASP
        elif self.phase == Phase.DESCEND_TO_GRASP and err_norm < GRASP_TOL:
            self.phase = Phase.CLOSE_GRIPPER
            self.settle_steps = 0


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, max_steps: int) -> Phase:
    controller = PickPlaceController(model, data)
    for _ in range(max_steps):
        controller.step()
        mujoco.mj_step(model, data)
        if controller.phase == Phase.DONE:
            break
    return controller.phase


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
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
            controller.step()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


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
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_home(model, data)

    if args.headless:
        final_phase = run_headless(model, data, args.max_steps)
        grasp_pos = data.site_xpos[model.site("grasp").id]
        print(f"Final phase: {final_phase.name}")
        print(f"Grasp site: {grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}")
        print(f"Gripper ctrl: {data.ctrl[GRIPPER_ACTUATOR]:.1f}")
        if final_phase != Phase.DONE:
            raise SystemExit(f"Controller did not finish within {args.max_steps} steps.")
        print("Pick-place demo slice completed.")
    else:
        run_viewer(model, data)


if __name__ == "__main__":
    main()
