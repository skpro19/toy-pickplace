"""Simulation setup helpers for the pick-place task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"


class SimEnv:
    def __init__(self, *, scene_path: Path = SCENE_PATH) -> None:
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.initial_cube_z = 0.0
        self.reset_episode()

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

    def reset_episode(self) -> None:
        """Reset robot, cube, and tray to the episode start state."""
        self.reset_home()
        self.initial_cube_z = float(self.data.body("cube").xpos[2])
