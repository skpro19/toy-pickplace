"""Simulation setup helpers for the pick-place task."""

from __future__ import annotations

from pathlib import Path

import mujoco


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"


class SimEnv:
    def __init__(self, *, scene_path: Path = SCENE_PATH) -> None:
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.initial_cube_z = 0.0
        self.reset_episode()

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

    def reset_episode(self) -> float:
        """Reset robot, cube, and tray to the episode start state."""
        self.reset_home()
        self.initial_cube_z = float(self.data.body("cube").xpos[2])
        return self.initial_cube_z
