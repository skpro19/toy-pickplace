"""Collect expert pick-place demonstrations for behaviour cloning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from expert import (
    CUBE_LIFT_MIN_DELTA,
    Phase,
    PickPlaceController,
    SCENE_PATH,
    TRAY_PLACE_TOL,
    reset_home,
)



class EpisodeBuffers:
    """Per-episode trajectory storage (lists grow each sim step)."""

    def __init__(self) -> None:
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.phase: list[int] = []
        self.ee_pos: list[np.ndarray] = []
        self.cube_pos: list[np.ndarray] = []
        self.tray_pos: list[np.ndarray] = []

    def append_step(
        self,
        *,
        obs: np.ndarray,
        action: np.ndarray,
        data: mujoco.MjData,
        phase: Phase,
        model: mujoco.MjModel,
    ) -> None:
        """Record one timestep."""
        self.obs.append(obs)
        self.actions.append(action)
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.ctrl.append(data.ctrl.copy())
        self.phase.append(phase.value)

        # TODO: log ee_pos, cube_pos, tray_pos from MuJoCo data
        _ = model
        self.ee_pos.append(np.zeros(3))
        self.cube_pos.append(np.zeros(3))
        self.tray_pos.append(np.zeros(3))

    def as_dict(self, *, success: bool) -> dict[str, np.ndarray]:
        """Stack lists into arrays for np.savez_compressed."""
        return {
            "obs": np.stack(self.obs),
            "actions": np.stack(self.actions),
            "qpos": np.stack(self.qpos),
            "qvel": np.stack(self.qvel),
            "ctrl": np.stack(self.ctrl),
            "phase": np.array(self.phase, dtype=np.int32),
            "ee_pos": np.stack(self.ee_pos),
            "cube_pos": np.stack(self.cube_pos),
            "tray_pos": np.stack(self.tray_pos),
            "success": np.array(success),
        }


class EpisodeResult:
    
    def __init__(self) -> None:
        self.buffers: EpisodeBuffers = EpisodeBuffers()
        self.final_phase: Phase = Phase.INIT
        self.max_cube_z: float = 0.0
        self.initial_cube_z: float = 0.0
        self.success: bool = False


def build_obs(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Build the BC observation vector.

    Target: [qpos, qvel, gripper, cube pose, tray pose, ee pose]

    TODO: choose a fixed layout and document dim order in a comment.
    """
    _ = model, data
    raise NotImplementedError


def reset_episode(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """
    Reset sim to a new episode start. Returns initial_cube_z for lift metric.

    TODO:
      1. Call reset_home(model, data)
      2. Call mujoco.mj_forward(model, data) if needed after qpos edits
      3. Return float(data.body("cube").xpos[2])
    """
    reset_home(model, data)
    return float(data.body("cube").xpos[2])


# def evaluate_success(
#     *,
#     final_phase: Phase,
#     cube_pos: np.ndarray,
#     tray_pos: np.ndarray,
#     initial_cube_z: float,
#     max_cube_z: float,
# ) -> bool:
#     """
#     Same criteria as run_pick_place headless checks.

#     TODO:
#       - final_phase == Phase.DONE
#       - (max_cube_z - initial_cube_z) >= CUBE_LIFT_MIN_DELTA
#       - ||cube_xy - tray_xy|| <= TRAY_PLACE_TOL
#     """
#     _ = final_phase, cube_pos, tray_pos, initial_cube_z, max_cube_z
#     _ = CUBE_LIFT_MIN_DELTA, TRAY_PLACE_TOL
#     raise NotImplementedError


def rollout_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    max_steps: int,
) -> EpisodeResult:
    """Run one expert episode and record trajectories."""
    initial_cube_z = reset_episode(model, data)

    controller = PickPlaceController(model, data)
    buffers = EpisodeBuffers()
    max_cube_z = initial_cube_z

    for _ in range(max_steps):
        obs = build_obs(model, data)

        controller.control()
        action = data.ctrl.copy()

        buffers.append_step(
            obs=obs,
            action=action,
            data=data,
            phase=controller.phase,
            model=model,
        )

        mujoco.mj_step(model, data)

        max_cube_z = max(max_cube_z, float(data.body("cube").xpos[2]))
        controller.max_cube_z = max_cube_z
        controller.update_phase()

        if controller.phase == Phase.DONE:
            break

    success = evaluate_success(
        final_phase=controller.phase,
        cube_pos=data.body("cube").xpos,
        tray_pos=data.site_xpos[model.site("tray_center").id],
        initial_cube_z=initial_cube_z,
        max_cube_z=max_cube_z,
    )

    return EpisodeResult(
        buffers=buffers,
        final_phase=controller.phase,
        max_cube_z=max_cube_z,
        initial_cube_z=initial_cube_z,
        success=success,
    )


def episode_path(out_dir: Path, episode_index: int, success: bool) -> Path:
    """TODO: choose naming scheme, e.g. success/ vs failures/ subdirs."""
    sub = "success" if success else "failures"
    return out_dir / sub / f"pick_place_{episode_index:06d}.npz"


def save_episode(path: Path, result: EpisodeResult) -> None:
    """Write one compressed npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.buffers.as_dict(success=result.success)
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect expert pick-place demos.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("data/demos"))
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--save-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    saved = 0
    attempted = 0

    while saved < args.episodes:
        attempted += 1

        result = rollout_episode(
            model,
            data,
            max_steps=args.max_steps,
        )

        # if result.success or args.save_failures:
        #     path = episode_path(
        #         args.out_dir,
        #         saved if result.success else attempted,
        #         result.success,
        #     )
        #     save_episode(path, result)
        #     if result.success:
        #         saved += 1

        # print(
        #     f"attempt={attempted} saved={saved}/{args.episodes} "
        #     f"phase={result.final_phase.name} success={result.success}"
        # )

    print(f"Done. Saved {saved} successful demos to {args.out_dir}")


if __name__ == "__main__":
    main()
