"""Collect scripted pick-place demonstrations for behaviour cloning.

This is intentionally a high-level scaffold. Fill in the TODOs as an exercise:
define the observation vector, record actions, decide success, and save episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
from tqdm import tqdm
from datetime import datetime


from expert import CUBE_LIFT_MIN_DELTA, Phase, PickPlaceController, TRAY_PLACE_TOL
from sim import SimEnv


# DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demos"


class DataCollector:
    def __init__(self, *, sim: SimEnv) -> None:
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.tray_center_id = self.model.site("tray_center").id
        self.grasp_id = self.model.site("grasp").id


    @staticmethod
    def save_episode(
        *,
        out_dir: str | Path,
        episode_idx: int,
        observations: list[object],
        actions: list[object],
        cube_init_pos: np.ndarray,
        tray_init_pos: np.ndarray,
    ) -> None:
        """Persist one episode to disk."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_path = out_dir / f"pick_place_{episode_idx:06d}.npz"

        np.savez_compressed(
            episode_path,
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            cube_init_pos=np.asarray(cube_init_pos, dtype=np.float32),
            tray_init_pos=np.asarray(tray_init_pos, dtype=np.float32),
        )

    def collect_episode(
        self,
        *,
        max_steps: int,
        episode_idx: int,
    ) -> dict[str, object]:
        """Run one scripted expert episode and return trajectory buffers."""
        initial_cube_z = float(self.data.body("cube").xpos[2])
        controller = PickPlaceController(self.model, self.data)

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []

        step_count = 0
        progress = tqdm(
            range(max_steps),
            desc=f"episode {episode_idx:06d}",
            leave=False,
            unit="step",
        )
        for step_count, _ in enumerate(progress, start=1):
            observations.append(self.sim.build_observation())

            controller.control()
            actions.append(self.sim.build_action())

            mujoco.mj_step(self.model, self.data)

            controller.max_cube_z = max(
                controller.max_cube_z,
                float(self.data.body("cube").xpos[2]),
            )
            controller.update_phase()
            progress.set_postfix(phase=controller.phase.name)

            if controller.phase == Phase.DONE:
                break

        return {
            "observations": observations,
            "actions": actions,
            "final_phase": controller.phase,
            "steps": step_count,
        }

    def collect_episodes(
        self,
        *,
        episodes: int,
        out_dir: Path,
        seed: int,
        max_steps: int,
        randomize_scene: bool,
    ) -> None:
        """Collect and optionally save multiple scripted expert episodes."""

        rng = np.random.default_rng(seed)

        episode_progress = tqdm(range(episodes), desc="episodes", unit="episode")
        for episode_idx in episode_progress:
            self.sim.reset_episode(randomize=randomize_scene, rng=rng)
            cube_init_pos = self.data.body("cube").xpos.copy()
            tray_init_pos = self.data.site("tray_center").xpos.copy()

            data = self.collect_episode(
                max_steps=max_steps,
                episode_idx=episode_idx,
            )
            episode_progress.set_postfix(
                phase=data["final_phase"].name,
                steps=data["steps"],
            )

            DataCollector.save_episode(
                out_dir=out_dir,
                episode_idx=episode_idx,
                observations=data["observations"],
                actions=data["actions"],
                cube_init_pos=cube_init_pos,
                tray_init_pos=tray_init_pos,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scripted pick-place demonstrations for BC."
    )
    parser.add_argument("--episodes", type=int, default=1)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--out-dir", type=str, default=f"data/demos/{timestamp}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument(
        "--randomize-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = SimEnv()
    
    collector = DataCollector(sim=sim)
    collector.collect_episodes(
        episodes=args.episodes,
        out_dir=args.out_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        randomize_scene=args.randomize_scene,
    )

    # print(f"Collected {successes}/{args.episodes} successful episodes; saved {saved}.")


if __name__ == "__main__":
    main()
