"""Collect scripted pick-place demonstrations for behaviour cloning.

This is intentionally a high-level scaffold. Fill in the TODOs as an exercise:
define the observation vector, record actions, decide success, and save episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from expert import Phase, PickPlaceController
from sim import SimEnv


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demos"


def build_observation(*, model: mujoco.MjModel, data: mujoco.MjData) -> object:
    """Return one low-dimensional observation for the current simulator state."""
    # TODO: Build the Stage 1 observation vector:
    # qpos, qvel, gripper state, cube pose, tray pose, and end-effector pose.
    # Returning object keeps this scaffold importable before you choose an array format.
    raise NotImplementedError


def build_action(*, model: mujoco.MjModel, data: mujoco.MjData) -> object:
    """Return the expert action for the current simulator state."""
    # TODO: Record the full MuJoCo control vector, usually data.ctrl[: model.nu].copy().
    raise NotImplementedError


def is_successful_episode(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: PickPlaceController,
    initial_cube_z: float,
) -> bool:
    """Decide whether an episode should be kept for behaviour cloning."""
    # TODO: Match the checks used by scripts/run_pick_place.py:
    # final phase DONE, enough cube lift, and small tray XY placement error.
    raise NotImplementedError


def save_episode(
    *,
    out_dir: Path,
    episode_idx: int,
    observations: list[object],
    actions: list[object],
    phases: list[str],
    success: bool,
    seed: int,
) -> None:
    """Persist one episode to disk."""
    # TODO: Create out_dir and save a compressed .npz file.
    # Suggested arrays: obs, actions, phase, success, seed.
    # Suggested filename: pick_place_000001.npz.
    raise NotImplementedError


def collect_episode(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    max_steps: int,
) -> tuple[list[object], list[object], list[str], bool, PickPlaceController]:
    """Run one scripted expert episode and return trajectory buffers."""
    # initial_cube_z = float(data.body("cube").xpos[2])
    controller = PickPlaceController(model, data)

    observations: list[object] = []
    actions: list[object] = []
    phases: list[str] = []

    for _ in range(max_steps):
        # TODO: Decide whether to record before or after controller.control().
        # For BC, a common choice is obs_t before control and action_t after control.
        observations.append(build_observation(model=model, data=data))

        controller.control()
        actions.append(build_action(model=model, data=data))
        # phases.append(controller.phase.name)

        mujoco.mj_step(model, data)
        controller.max_cube_z = max(
            controller.max_cube_z,
            float(data.body("cube").xpos[2]),
        )
        controller.update_phase()

        if controller.phase == Phase.DONE:
            break

    success = is_successful_episode(
        model=model,
        data=data,
        controller=controller,
        initial_cube_z=initial_cube_z,
    )
    return observations, actions, phases, success, controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scripted pick-place demonstrations for BC."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Also save failed episodes for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = SimEnv()

    successes = 0
    saved = 0

    for episode_idx in range(args.episodes):
        # TODO: Use args.seed + episode_idx once you add episode randomization.
        episode_seed = args.seed + episode_idx
        sim.reset_episode()

        observations, actions, phases, success, _controller = collect_episode(
            model=sim.model,
            data=sim.data,
            max_steps=args.max_steps,
        )

        if success:
            successes += 1

        if success or args.save_failures:
            save_episode(
                out_dir=args.out_dir,
                episode_idx=episode_idx,
                observations=observations,
                actions=actions,
                phases=phases,
                success=success,
                seed=episode_seed,
            )
            saved += 1

        print(
            f"episode={episode_idx:04d} "
            f"steps={len(actions)} "
            f"success={success} "
            f"saved={success or args.save_failures}"
        )

    print(f"Collected {successes}/{args.episodes} successful episodes; saved {saved}.")


if __name__ == "__main__":
    main()
