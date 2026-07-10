"""Load and replay actions from .npz files in mujoco viewer."""
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from pathlib import Path

from sim import SimEnv


def restore_episode_layout(
    *,
    sim: SimEnv,
    cube_init_pos: np.ndarray,
    tray_init_pos: np.ndarray,
) -> None:
    """Restore the randomized layout used when the episode was collected."""
    sim.data.qpos[sim.cube_qpos_addr : sim.cube_qpos_addr + 3] = cube_init_pos
    sim.data.qpos[sim.cube_qpos_addr + 3 : sim.cube_qpos_addr + 7] = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    tray_center_id = sim.model.site("tray_center").id
    sim.model.body_pos[sim.tray_body_id] = tray_init_pos - sim.model.site_pos[tray_center_id]
    mujoco.mj_forward(sim.model, sim.data)


def replay_actions(
    *,
    npz: str,
    episodes: int = 10,
    randomised: bool = True,
    action_key: str = "actions",
) -> None:
    sim = SimEnv()
    npz_path = Path(npz)
    if npz_path.is_file():
        npz_files = [npz_path]
    elif npz_path.is_dir():
        npz_files = sorted(npz_path.glob("*.npz"))
    else:
        raise FileNotFoundError(f"NPZ path not found: {npz_path}")

    if randomised:
        rng = np.random.default_rng()
        rng.shuffle(npz_files)

    selected = npz_files[:episodes]
    print(f"Replaying {len(selected)} / {len(npz_files)} episodes from {npz_path.resolve()}")

    quit_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested
        if chr(keycode).lower() == "q":
            quit_requested = True

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=key_callback,
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
            if quit_requested:
                break

            print(f"\nEpisode {i + 1}/{len(selected)}: {fpath.name}")
            sim.reset_episode()

            with np.load(fpath) as data:
                if action_key not in data.files:
                    raise KeyError(
                        f"{fpath} does not contain action key {action_key!r}. "
                        f"Available keys: {sorted(data.files)}"
                    )
                actions = data[action_key]
                if "cube_init_pos" in data.files and "tray_init_pos" in data.files:
                    restore_episode_layout(
                        sim=sim,
                        cube_init_pos=data["cube_init_pos"],
                        tray_init_pos=data["tray_init_pos"],
                    )
                else:
                    print("  layout metadata missing; replaying from reset layout")
                print(f"  {action_key}.shape => {actions.shape}")

                for action in actions:
                    if quit_requested or not viewer.is_running():
                        break
                    sim.data.ctrl[: sim.model.nu] = action
                    mujoco.mj_step(sim.model, sim.data)
                    time.sleep(sim.model.opt.timestep * 1)
                    viewer.sync()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="data/test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--action-key",
        type=str,
        default="actions",
        help="NPZ action array to replay, e.g. actions, executed_actions, policy_actions.",
    )
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
    replay_actions(
        npz=args.npz,
        episodes=args.episodes,
        randomised=args.randomised,
        action_key=args.action_key,
    )


if __name__ == "__main__":
    main()
