"""Run a scripted pick-place demo: approach cube, close gripper."""

from __future__ import annotations

import argparse
import time
from enum import Enum, auto
from pathlib import Path

import mujoco
import mujoco.viewer
import mink
import numpy as np

SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"

ARM_DOF = 7
GRIPPER_ACTUATOR = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

POS_TOL = 0.02
ORI_TOL = 0.15
GRASP_SETTLE_STEPS = 150
VIEWER_SLOWDOWN = 10.0
IK_SOLVER = "daqp"
IK_DAMPING = 1e-3
MAX_IK_ITERS = 20

# Offsets in the parent site/body local frame (cube_center / tray_center).
CUBE_GRASP_OFFSET = np.array([0.0, 0.0, 0.03])
TRAY_HOVER_OFFSET = np.array([0.0, 0.0, 0.085])
TRAY_DROP_OFFSET = np.array([0.0, 0.0, 0.05])


class Phase(Enum):
    MOVE_ABOVE_CUBE = auto()
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


def site_pose(data: mujoco.MjData, site_id: int) -> mink.SE3:
    pos = data.site_xpos[site_id].copy()
    rot = mink.SO3.from_matrix(data.site_xmat[site_id].reshape(3, 3))
    return mink.SE3.from_rotation_and_translation(rot, pos)


def site_target(data: mujoco.MjData, site_id: int, local_offset: np.ndarray) -> np.ndarray:
    site_rot = data.site_xmat[site_id].reshape(3, 3)
    return data.site_xpos[site_id] + site_rot @ local_offset


def converge_ik(
    configuration: mink.Configuration,
    grasp_task: mink.FrameTask,
    tasks: list[mink.Task],
    dt: float,
) -> tuple[float, float]:
    """Run IK substeps on configuration; return final position/orientation error norms."""
    pos_err = ori_err = 0.0
    for _ in range(MAX_IK_ITERS):
        vel = mink.solve_ik(configuration, tasks, dt, IK_SOLVER, damping=IK_DAMPING)
        configuration.integrate_inplace(vel, dt)
        err = grasp_task.compute_error(configuration)
        pos_err = float(np.linalg.norm(err[:3]))
        ori_err = float(np.linalg.norm(err[3:]))
        if pos_err <= POS_TOL and ori_err <= ORI_TOL:
            break
    return pos_err, ori_err


class PickPlaceController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.phase = Phase.MOVE_ABOVE_CUBE
        self.settle_steps = 0
        self.cube_hover_id = model.site("cube_hover").id
        self.dt = model.opt.timestep
        self.last_pos_err = float("inf")
        self.last_ori_err = float("inf")

        self.configuration = mink.Configuration(model)
        self.configuration.update(data.qpos)
        self.grasp_task = mink.FrameTask(
            frame_name="grasp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model, cost=1e-2)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.ik_tasks: list[mink.Task] = [self.grasp_task, self.posture_task]

    def target_for_phase(self) -> mink.SE3 | None:
        if self.phase in (Phase.MOVE_ABOVE_CUBE, Phase.CLOSE_GRIPPER):
            return site_pose(self.data, self.cube_hover_id)
        return None

    def run_ik(self, target: mink.SE3) -> tuple[float, float]:
        self.grasp_task.set_target(target)
        pos_err, ori_err = converge_ik(
            self.configuration,
            self.grasp_task,
            self.ik_tasks,
            self.dt,
        )
        self.data.ctrl[:ARM_DOF] = self.configuration.q[:ARM_DOF]
        return pos_err, ori_err

    def step(self) -> None:
        if self.phase == Phase.DONE:
            return

        # if self.phase == Phase.CLOSE_GRIPPER:
        #     self.data.ctrl[GRIPPER_ACTUATOR] = GRIPPER_CLOSE
        #     target = self.target_for_phase()
        #     if target is not None:
        #         self.last_pos_err, self.last_ori_err = self.run_ik(target)
        #     self.settle_steps += 1
        #     if self.settle_steps >= GRASP_SETTLE_STEPS:
        #         self.phase = Phase.DONE
        #     return

        self.data.ctrl[GRIPPER_ACTUATOR] = GRIPPER_OPEN
        target = self.target_for_phase()
        if target is None:
            return

        self.last_pos_err, self.last_ori_err = self.run_ik(target)
        if (
            self.phase == Phase.MOVE_ABOVE_CUBE
            and self.last_pos_err <= POS_TOL
            and self.last_ori_err <= ORI_TOL
        ):
            self.phase = Phase.CLOSE_GRIPPER
            self.settle_steps = 0


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    max_steps: int,
) -> tuple[Phase, PickPlaceController]:
    controller = PickPlaceController(model, data)
    for _ in range(max_steps):
        controller.step()
        mujoco.mj_step(model, data)
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
            controller.step()
            mujoco.mj_step(model, data)
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

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_home(model, data)

    if args.headless:
        final_phase, controller = run_headless(model, data, args.max_steps)
        grasp_pos = data.site_xpos[model.site("grasp").id]
        print(f"Final phase: {final_phase.name}")
        print(f"Grasp site: {grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}")
        print(f"Gripper ctrl: {data.ctrl[GRIPPER_ACTUATOR]:.1f}")
        print(
            f"IK error: pos={controller.last_pos_err:.4f} m, "
            f"ori={controller.last_ori_err:.4f} rad"
        )
        if final_phase != Phase.DONE:
            raise SystemExit(f"Controller did not finish within {args.max_steps} steps.")
        print("Pick-place demo slice completed.")
    else:
        run_viewer(model, data, slowdown=args.slowdown)


if __name__ == "__main__":
    main()
