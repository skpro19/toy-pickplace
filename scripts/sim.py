"""Simulation setup helpers for the pick-place task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"

CUBE_X_RANGE = (0.48, 0.60)
CUBE_Y_RANGE = (-0.20, -0.04)
TRAY_X_RANGE = (0.62, 0.74)
TRAY_Y_RANGE = (0.08, 0.22)
MIN_CUBE_TRAY_DIST = 0.22


class SimEnv:
    def __init__(self, *, scene_path: Path = SCENE_PATH) -> None:
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.cube_joint_id = self.model.joint("cube_freejoint").id
        self.cube_qpos_addr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.tray_body_id = self.model.body("tray").id
        self.default_tray_pos = self.model.body_pos[self.tray_body_id].copy()
        self.initial_cube_z = 0.0
        self.reset_episode()

    def sample_scene_layout(
        self,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample valid cube and tray start positions on the table."""
        for _ in range(100):
            cube_pos = np.array(
                [
                    rng.uniform(*CUBE_X_RANGE),
                    rng.uniform(*CUBE_Y_RANGE),
                    0.815,
                ],
                dtype=np.float64,
            )
            tray_pos = np.array(
                [
                    rng.uniform(*TRAY_X_RANGE),
                    rng.uniform(*TRAY_Y_RANGE),
                    self.default_tray_pos[2],
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(cube_pos[:2] - tray_pos[:2]) >= MIN_CUBE_TRAY_DIST:
                return cube_pos, tray_pos

        raise RuntimeError("Failed to sample a valid randomized scene layout.")

    def build_observation(self) -> np.ndarray:
        """Return one low-dimensional observation for the current simulator state."""
        # arm joints
        arm_qpos = self.data.qpos[:7].copy()
        gripper_qpos = self.data.qpos[7:9].copy()
        
        # cube body
        cube_xpos = self.data.body("cube").xpos.copy()
        # cube_xquat = self.data.body("cube").xquat.copy()
        cube_xmat = self.data.body("cube").xmat.copy()
        
        # tray center site
        tray_xpos = self.data.site("tray_center").xpos.copy()
        tray_xmat = self.data.site("tray_center").xmat.copy()
        
        # ee site
        grasp_xpos = self.data.site("grasp").xpos.copy()
        grasp_xmat = self.data.site("grasp").xmat.copy()

        # print(f"type(arm_qpos) => {type(arm_qpos)} arm_qpos.shape=>{arm_qpos.shape}")

        obs = np.concatenate(
            [
                arm_qpos,
                gripper_qpos,
                cube_xpos,
                cube_xmat,
                tray_xpos,
                tray_xmat,
                grasp_xpos,
                grasp_xmat,
            ]
        ).astype(np.float32)
        return obs

    def build_action(self) -> np.ndarray:
        """Return the expert action for the current simulator state."""
        arm_ctrl = self.data.ctrl[:7].copy()
        gripper_ctrl = self.data.ctrl[7:8].copy()

        action = np.concatenate(
            [
                arm_ctrl,
                gripper_ctrl,
            ]
        ).astype(np.float32)
        return action

    @staticmethod
    def find_reset_key(model: mujoco.MjModel) -> int:
        for name in ("task_home", "home"):
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
            if key_id >= 0:
                return key_id
        return -1

    def reset_home(self) -> None:
        key_id = self.find_reset_key(self.model)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            if self.model.nu:
                self.data.ctrl[: self.model.nu] = self.model.key_ctrl[
                    key_id, : self.model.nu
                ]
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset_episode(
        self,
        *,
        randomize: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Reset robot, cube, and tray to the episode start state."""
        self.reset_home()
        self.model.body_pos[self.tray_body_id] = self.default_tray_pos

        if randomize:
            if rng is None:
                raise ValueError("rng is required when randomize=True.")
            cube_pos, tray_pos = self.sample_scene_layout(rng=rng)
            self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3] = cube_pos
            self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            self.model.body_pos[self.tray_body_id] = tray_pos

        mujoco.mj_forward(self.model, self.data)
        self.initial_cube_z = float(self.data.body("cube").xpos[2])
