"""Load and replay actions from .npz files in mujoco viewer."""
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from pathlib import Path

from sim import SimEnv


def replay_actions(*, npz_dir: str, episodes: int = 10, randomised: bool = True) -> None:
    sim = SimEnv()
    npz_path = Path(npz_dir)
    npz_files = sorted(npz_path.glob("*.npz"))

    if randomised:
        rng = np.random.default_rng()
        rng.shuffle(npz_files)

    selected = npz_files[:episodes]
    print(f"Replaying {len(selected)} / {len(npz_files)} episodes from {npz_path.resolve()}")

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.45, 0.0, 0.78)
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        for i, fpath in enumerate(selected):
            print(f"\nEpisode {i + 1}/{len(selected)}: {fpath.name}")
            sim.reset_episode()

            with np.load(fpath) as data:
                actions = data["actions"]
                print(f"  actions.shape => {actions.shape}")

                for action in actions:
                    sim.data.ctrl[: sim.model.nu] = action
                    mujoco.mj_step(sim.model, sim.data)
                    time.sleep(sim.model.opt.timestep * 1)
                    viewer.sync()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="data/test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--no-randomised",
        action="store_false",
        dest="randomised",
        default=True,
        help="Disable shuffling of file order (default: shuffled)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    replay_actions(npz_dir=args.npz, episodes=args.episodes, randomised=args.randomised)


if __name__ == "__main__":
    main()
