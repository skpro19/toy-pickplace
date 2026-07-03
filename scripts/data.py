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
        out_dir: str,
        episode_idx: int,
        observations: list[object],
        actions: list[object]) -> None:
        """Persist one episode to disk."""

        out_dir =  Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_path = out_dir / f"pick_place_{episode_idx:06d}.npz"

        np.savez_compressed(
            episode_path,
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
        )

    def collect_episode(
        self,
        *,
        max_steps: int) -> dict[str, list[np.ndarray]]:
        """Run one scripted expert episode and return trajectory buffers."""
        initial_cube_z = float(self.data.body("cube").xpos[2])
        controller = PickPlaceController(self.model, self.data)

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []

        for _ in tqdm(range(0,max_steps)):
            observations.append(self.sim.build_observation())

            controller.control()
            actions.append(self.sim.build_action())

            mujoco.mj_step(self.model, self.data)

            controller.max_cube_z = max(
                controller.max_cube_z,
                float(self.data.body("cube").xpos[2]),
            )
            controller.update_phase()

            if controller.phase == Phase.DONE:
                break

        return {"observations": observations, "actions": actions}

    def collect_episodes(
        self,
        *,
        episodes: int,
        out_dir: Path,
        seed: int,
        max_steps: int) -> None:
        """Collect and optionally save multiple scripted expert episodes."""

        for episode_idx in range(episodes):
            self.sim.reset_episode()

            data = self.collect_episode(
                max_steps=max_steps,
            )

            DataCollector.save_episode(
                out_dir=out_dir,
                episode_idx=episode_idx,
                observations=data["observations"],
                actions=data["actions"],
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
    )

    # print(f"Collected {successes}/{args.episodes} successful episodes; saved {saved}.")


if __name__ == "__main__":
    main()
