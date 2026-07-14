"""Expert pick-place controller reused by demo collection, training, and eval."""

from __future__ import annotations

from enum import Enum, auto

import mujoco
import mink
import numpy as np

from sim import SimEnv

ARM_DOF = 7
GRIPPER_ACTUATOR = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

POS_TOL = 0.02
ORI_TOL = 0.15
ARRIVAL_SETTLE_STEPS = 15
GRASP_SETTLE_STEPS = 150
GRASP_CONTACT_STABLE_STEPS = 5
RELEASE_SETTLE_STEPS = 100
IK_SOLVER = "daqp"
IK_DAMPING = 1e-3
MAX_IK_ITERS = 20
CUBE_LIFT_MIN_DELTA = 0.05
TRAY_PLACE_TOL = 0.08
HOME_JOINT_TOL = 0.03

CUBE_GRASP_OFFSET = np.array([0.0, 0.0, 0.03])
TRAY_HOVER_OFFSET = np.array([0.0, 0.0, 0.11])
TRAY_DROP_OFFSET = np.array([0.0, 0.0, 0.05])


class Phase(Enum):
    MOVE_ABOVE_CUBE = auto()
    MOVE_TO_CUBE_GRASP = auto()
    CLOSE_GRIPPER = auto()
    LIFT_CUBE = auto()
    MOVE_TO_TRAY = auto()
    LOWER_TO_TRAY = auto()
    RELEASE = auto()
    RETREAT = auto()
    HOME = auto()
    DONE = auto()


class PickPlaceController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.model = model
        self.data = data

        self.phase = Phase.MOVE_ABOVE_CUBE
        self.settle_steps = 0
        self.grasp_contact_steps = 0
        self.grasp_id = model.site("grasp").id
        self.cube_hover_id = model.site("cube_hover").id
        self.cube_grasp_id = model.site("cube_grasp").id
        self.cube_lift_id = model.site("cube_lift").id
        self.tray_center_id = model.site("tray_center").id
        self.cube_geom_id = model.geom("cube_geom").id
        self.left_finger_body_id = model.body("left_finger").id
        self.right_finger_body_id = model.body("right_finger").id

        self.dt = model.opt.timestep
        self.last_pos_err = float("inf")
        self.last_ori_err = float("inf")
        self.max_cube_z = float(data.body("cube").xpos[2])
        self.lift_target: mink.SE3 | None = None
        self.tray_hover_target: mink.SE3 | None = None
        self.tray_drop_target: mink.SE3 | None = None
        self.home_qpos = data.qpos.copy()
        self.home_ctrl = data.ctrl.copy()
        home_key_id = SimEnv.find_reset_key(model)
        if home_key_id >= 0:
            self.home_qpos = model.key_qpos[home_key_id].copy()
            self.home_ctrl = model.key_ctrl[home_key_id].copy()

        # mink config
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

    def site_pose(self, site_id: int) -> mink.SE3:
        pos = self.data.site_xpos[site_id].copy()
        rot = mink.SO3.from_matrix(self.data.site_xmat[site_id].reshape(3, 3))
        return mink.SE3.from_rotation_and_translation(rot, pos)

    def site_target(self, site_id: int, local_offset: np.ndarray) -> np.ndarray:
        site_rot = self.data.site_xmat[site_id].reshape(3, 3)
        return self.data.site_xpos[site_id] + site_rot @ local_offset

    def offset_site_pose(
        self,
        site_id: int,
        local_offset: np.ndarray,
        rotation: mink.SO3) -> mink.SE3:
        return mink.SE3.from_rotation_and_translation(
            rotation,
            self.site_target(site_id, local_offset),
        )

    @staticmethod
    def rotation_error(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
        rot_err = rot_a.T @ rot_b
        cos_angle = (np.trace(rot_err) - 1.0) / 2.0
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def converge_ik(self) -> tuple[float, float]:
        pos_err = ori_err = 0.0
        for _ in range(MAX_IK_ITERS):
            vel = mink.solve_ik(
                self.configuration,
                self.ik_tasks,
                self.dt,
                IK_SOLVER,
                damping=IK_DAMPING,
            )
            self.configuration.integrate_inplace(vel, self.dt)
            err = self.grasp_task.compute_error(self.configuration)
            pos_err = float(np.linalg.norm(err[:3]))
            ori_err = float(np.linalg.norm(err[3:]))
            if pos_err <= POS_TOL and ori_err <= ORI_TOL:
                break
        return pos_err, ori_err

    def target_site_id_for_phase(self) -> int | None:
        if self.phase == Phase.MOVE_ABOVE_CUBE:
            return self.cube_hover_id
        if self.phase in (Phase.MOVE_TO_CUBE_GRASP, Phase.CLOSE_GRIPPER):
            return self.cube_grasp_id
        if self.phase == Phase.LIFT_CUBE:
            return self.cube_lift_id
        return None

    def fixed_target_for_phase(self) -> mink.SE3 | None:
        if self.phase == Phase.LIFT_CUBE:
            return self.lift_target
        if self.phase == Phase.MOVE_TO_TRAY:
            return self.tray_hover_target
        if self.phase in (Phase.LOWER_TO_TRAY, Phase.RELEASE):
            return self.tray_drop_target
        if self.phase == Phase.RETREAT:
            return self.tray_hover_target
        return None

    def target_for_phase(self) -> mink.SE3 | None:
        fixed_target = self.fixed_target_for_phase()
        if fixed_target is not None:
            return fixed_target
        target_id = self.target_site_id_for_phase()
        if target_id is None:
            return None
        return self.site_pose(target_id)

    def sim_tracking_error(self) -> tuple[float, float]:
        fixed_target = self.fixed_target_for_phase()
        if fixed_target is not None:
            grasp_pos = self.data.site_xpos[self.grasp_id]
            grasp_rot = self.data.site_xmat[self.grasp_id].reshape(3, 3)
            target_pos = fixed_target.translation()
            target_rot = fixed_target.rotation().as_matrix()
            pos_err = float(np.linalg.norm(grasp_pos - target_pos))
            ori_err = self.rotation_error(target_rot, grasp_rot)
            return pos_err, ori_err

        target_id = self.target_site_id_for_phase()
        if target_id is None:
            return float("inf"), float("inf")
        pos_err = float(
            np.linalg.norm(
                self.data.site_xpos[self.grasp_id] - self.data.site_xpos[target_id]
            )
        )
        grasp_rot = self.data.site_xmat[self.grasp_id].reshape(3, 3)
        target_rot = self.data.site_xmat[target_id].reshape(3, 3)
        ori_err = self.rotation_error(target_rot, grasp_rot)
        return pos_err, ori_err

    def cube_finger_contacts(self) -> set[int]:
        finger_contacts: set[int] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == self.cube_geom_id:
                other_geom_id = contact.geom2
            elif contact.geom2 == self.cube_geom_id:
                other_geom_id = contact.geom1
            else:
                continue

            other_body_id = int(self.model.geom_bodyid[other_geom_id])
            if other_body_id in (
                self.left_finger_body_id,
                self.right_finger_body_id,
            ):
                finger_contacts.add(other_body_id)

        return finger_contacts

    def cube_has_any_finger_contact(self) -> bool:
        return bool(self.cube_finger_contacts())

    def cube_has_two_finger_contact(self) -> bool:
        return self.cube_finger_contacts() == {
            self.left_finger_body_id,
            self.right_finger_body_id,
        }

    def cube_is_in_tray(self) -> bool:
        cube_pos = self.data.body("cube").xpos
        tray_pos = self.data.site_xpos[self.tray_center_id]
        tray_error = float(np.linalg.norm(cube_pos[:2] - tray_pos[:2]))
        return tray_error <= TRAY_PLACE_TOL

    def run_ik(self, target: mink.SE3) -> np.ndarray:
        self.grasp_task.set_target(target)
        self.converge_ik()
        return self.configuration.q[:ARM_DOF].copy()

    def compute_actions(self) -> np.ndarray:

        self.configuration.update(self.data.qpos)
        actions = self.data.ctrl[: self.model.nu].copy()

        if self.phase == Phase.DONE:
            return actions

        if self.phase == Phase.HOME:
            return self.home_ctrl[: self.model.nu].copy()

        if self.phase in (
            Phase.CLOSE_GRIPPER,
            Phase.LIFT_CUBE,
            Phase.MOVE_TO_TRAY,
            Phase.LOWER_TO_TRAY,
        ):
            actions[GRIPPER_ACTUATOR] = GRIPPER_CLOSE
        else:
            actions[GRIPPER_ACTUATOR] = GRIPPER_OPEN

        target = self.target_for_phase()
        if target is None:
            return actions

        actions[:ARM_DOF] = self.run_ik(target)
        return actions

    def control(self) -> None:

        self.data.ctrl[:self.model.nu] = self.compute_actions()


    def update_phase(self) -> None:
        if self.phase == Phase.DONE:
            return

        if self.phase == Phase.CLOSE_GRIPPER:
            self.settle_steps += 1
            self.grasp_contact_steps = (
                self.grasp_contact_steps + 1
                if self.cube_has_two_finger_contact()
                else 0
            )
            if (
                self.settle_steps >= GRASP_SETTLE_STEPS
                and self.grasp_contact_steps >= GRASP_CONTACT_STABLE_STEPS
            ):
                self.lift_target = self.site_pose(self.cube_lift_id)
                self.phase = Phase.LIFT_CUBE
                self.settle_steps = 0
                self.grasp_contact_steps = 0

            return

        if self.phase == Phase.RELEASE:
            self.settle_steps += 1
            if self.settle_steps >= RELEASE_SETTLE_STEPS:
                self.phase = Phase.RETREAT
                self.settle_steps = 0
            return

        if self.phase == Phase.HOME:
            joint_err = float(
                np.max(np.abs(self.data.qpos[:ARM_DOF] - self.home_qpos[:ARM_DOF]))
            )
            self.last_pos_err = joint_err
            self.last_ori_err = 0.0
            if joint_err <= HOME_JOINT_TOL and self.cube_is_in_tray():
                self.settle_steps += 1
            else:
                self.settle_steps = 0
            if self.settle_steps >= ARRIVAL_SETTLE_STEPS:
                self.phase = Phase.DONE
                self.settle_steps = 0
            return

        self.last_pos_err, self.last_ori_err = self.sim_tracking_error()
        if self.last_pos_err <= POS_TOL and self.last_ori_err <= ORI_TOL:
            self.settle_steps += 1
        else:
            self.settle_steps = 0

        if self.settle_steps < ARRIVAL_SETTLE_STEPS:
            return

        if self.phase == Phase.MOVE_ABOVE_CUBE:
            self.phase = Phase.MOVE_TO_CUBE_GRASP
            self.settle_steps = 0
        elif self.phase == Phase.MOVE_TO_CUBE_GRASP:
            self.phase = Phase.CLOSE_GRIPPER
            self.settle_steps = 0
        elif self.phase == Phase.LIFT_CUBE:
            if not self.cube_has_two_finger_contact():
                return

            lift_rotation = (
                self.lift_target.rotation()
                if self.lift_target
                else self.site_pose(self.cube_lift_id).rotation()
            )
            self.tray_hover_target = self.offset_site_pose(
                self.tray_center_id,
                TRAY_HOVER_OFFSET,
                lift_rotation,
            )
            self.phase = Phase.MOVE_TO_TRAY
            self.settle_steps = 0
        elif self.phase == Phase.MOVE_TO_TRAY:
            tray_rotation = (
                self.tray_hover_target.rotation()
                if self.tray_hover_target
                else self.site_pose(self.grasp_id).rotation()
            )
            self.tray_drop_target = self.offset_site_pose(
                self.tray_center_id,
                TRAY_DROP_OFFSET,
                tray_rotation,
            )
            self.phase = Phase.LOWER_TO_TRAY
            self.settle_steps = 0
        elif self.phase == Phase.LOWER_TO_TRAY:
            self.phase = Phase.RELEASE
            self.settle_steps = 0
        elif self.phase == Phase.RETREAT:
            self.phase = Phase.HOME
            self.settle_steps = 0
