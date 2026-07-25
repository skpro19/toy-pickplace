"""Collect scripted pick-place demonstrations for behaviour cloning.

Examples:
    # Collect proprioceptive trajectories at the default 60 Hz capture rate.
    uv run scripts/data.py --episodes 100 --out-dir data/expert/rand-100

    # Collect aligned proprioception, actions, and 64x64 RGB image observations.
    uv run scripts/data.py --episodes 100 --out-dir data/expert/rand-100-img \
        --capture-hz 60 --save-images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
from tqdm import tqdm
from datetime import datetime

from constant import DEFAULT_CAPTURE_HZ


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
        img_obs: list[np.ndarray] | None = None,
    ) -> None:
        """Persist one episode to disk."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_path = out_dir / f"pick_place_{episode_idx:06d}.npz"

        episode_arrays = {
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "cube_init_pos": np.asarray(cube_init_pos, dtype=np.float32),
            "tray_init_pos": np.asarray(tray_init_pos, dtype=np.float32),
        }
        if img_obs is not None:
            if not (len(img_obs) == len(observations) == len(actions)):
                raise ValueError(
                    "img_obs, observations, and actions must have the same length "
                    f"(got {len(img_obs)}, {len(observations)}, {len(actions)})"
                )
            episode_arrays["img_obs"] = np.asarray(img_obs, dtype=np.uint8)

        np.savez_compressed(episode_path, **episode_arrays)

    def collect_episode(
        self,
        *,
        max_steps: int,
        episode_idx: int,
        capture_hz: float = DEFAULT_CAPTURE_HZ,
        save_images: bool = False,
    ) -> dict[str, object]:
        """Run one scripted expert episode and return trajectory buffers."""
        if capture_hz <= 0.0:
            raise ValueError(f"capture_hz must be positive, got {capture_hz}")
        sim_hz = 1.0 / self.model.opt.timestep
        if capture_hz > sim_hz:
            raise ValueError(
                f"capture_hz ({capture_hz}) cannot exceed simulation rate ({sim_hz})"
            )

        initial_cube_z = float(self.data.body("cube").xpos[2])
        controller = PickPlaceController(self.model, self.data)

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        img_obs: list[np.ndarray] = []
        capture_period = 1.0 / capture_hz
        next_capture_time = float(self.data.time)

        step_count = 0
        progress = tqdm(
            range(max_steps),
            desc=f"episode {episode_idx:06d}",
            leave=False,
            unit="step",
        )
        for step_count, _ in enumerate(progress, start=1):
            capture_due = self.data.time + 1e-9 >= next_capture_time
            if capture_due:
                observations.append(self.sim.build_observation())
                if save_images:
                    img_obs.append(self.sim.build_image())

            controller.control()
            if capture_due:
                actions.append(self.sim.build_action())
                next_capture_time += capture_period

            mujoco.mj_step(self.model, self.data)

            controller.max_cube_z = max(
                controller.max_cube_z,
                float(self.data.body("cube").xpos[2]),
            )
            controller.update_phase()
            progress.set_postfix(phase=controller.phase.name)

            if controller.phase == Phase.DONE:
                break

        episode_data: dict[str, object] = {
            "observations": observations,
            "actions": actions,
            "final_phase": controller.phase,
            "steps": step_count,
        }
        if save_images:
            episode_data["img_obs"] = img_obs
        return episode_data

    def collect_episodes(
        self,
        *,
        episodes: int,
        out_dir: Path,
        max_steps: int,
        capture_hz: float = DEFAULT_CAPTURE_HZ,
        save_images: bool = False,
    ) -> None:
        """Collect and optionally save multiple scripted expert episodes."""

        episode_progress = tqdm(range(episodes), desc="episodes", unit="episode")
        for episode_idx in episode_progress:
            self.sim.reset_episode()
            cube_init_pos = self.data.body("cube").xpos.copy()
            tray_init_pos = self.data.site("tray_center").xpos.copy()

            data = self.collect_episode(
                max_steps=max_steps,
                episode_idx=episode_idx,
                capture_hz=capture_hz,
                save_images=save_images,
            )
            episode_progress.set_postfix(
                phase=data["final_phase"].name,
                steps=data["steps"],
            )

            save_kwargs = {
                "out_dir": out_dir,
                "episode_idx": episode_idx,
                "observations": data["observations"],
                "actions": data["actions"],
                "cube_init_pos": cube_init_pos,
                "tray_init_pos": tray_init_pos,
            }
            if save_images:
                save_kwargs["img_obs"] = data["img_obs"]
            DataCollector.save_episode(**save_kwargs)


def collect_expert_episodes(
    *,
    episodes: int,
    out_dir: Path,
    seed: int,
    max_steps: int,
    capture_hz: float = DEFAULT_CAPTURE_HZ,
    randomize_scene: bool = True,
    save_images: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_episodes = len(list(out_dir.glob("*.npz")))
    if existing_episodes >= episodes:
        return out_dir

    sim = SimEnv(randomize_scene=randomize_scene, seed=seed)
    try:
        collector = DataCollector(sim=sim)
        collector.collect_episodes(
            episodes=episodes,
            out_dir=out_dir,
            max_steps=max_steps,
            capture_hz=capture_hz,
            save_images=save_images,
        )
    finally:
        sim.close()
    return out_dir


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
        "--capture-hz",
        type=float,
        default=DEFAULT_CAPTURE_HZ,
        help="Rate for aligned obs, action, and image-observation samples.",
    )
    parser.add_argument(
        "--randomize-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Render top-camera RGB frames each step and store img_obs in the NPZ.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_expert_episodes(
        episodes=args.episodes,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        max_steps=args.max_steps,
        capture_hz=args.capture_hz,
        randomize_scene=args.randomize_scene,
        save_images=args.save_images,
    )

    # print(f"Collected {successes}/{args.episodes} successful episodes; saved {saved}.")


if __name__ == "__main__":
    main()
