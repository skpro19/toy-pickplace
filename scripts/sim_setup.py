"""Simulation setup helpers for the pick-place task."""

from __future__ import annotations

from pathlib import Path

import mujoco


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"


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


class SimEnv:
    def __init__(self, *, scene_path: Path = SCENE_PATH) -> None:
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.initial_cube_z = 0.0
        self.reset_episode()

    def reset_home(self) -> None:
        reset_home(self.model, self.data)

    def reset_episode(self) -> float:
        """Reset robot, cube, and tray to the episode start state."""
        self.reset_home()
        self.initial_cube_z = float(self.data.body("cube").xpos[2])
        return self.initial_cube_z
