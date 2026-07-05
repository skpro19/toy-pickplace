import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBS_NAMES = [
    "arm_qpos_0",
    "arm_qpos_1",
    "arm_qpos_2",
    "arm_qpos_3",
    "arm_qpos_4",
    "arm_qpos_5",
    "arm_qpos_6",
    "gripper_qpos_0",
    "gripper_qpos_1",
    "cube_xpos_0",
    "cube_xpos_1",
    "cube_xpos_2",
    "cube_xquat_0",
    "cube_xquat_1",
    "cube_xquat_2",
    "cube_xquat_3",
    "tray_xpos_0",
    "tray_xpos_1",
    "tray_xpos_2",
    "tray_xmat_0",
    "tray_xmat_1",
    "tray_xmat_2",
    "tray_xmat_3",
    "tray_xmat_4",
    "tray_xmat_5",
    "tray_xmat_6",
    "tray_xmat_7",
    "tray_xmat_8",
    "grasp_xpos_0",
    "grasp_xpos_1",
    "grasp_xpos_2",
    "grasp_xmat_0",
    "grasp_xmat_1",
    "grasp_xmat_2",
    "grasp_xmat_3",
    "grasp_xmat_4",
    "grasp_xmat_5",
    "grasp_xmat_6",
    "grasp_xmat_7",
    "grasp_xmat_8",
]

OBS_GROUPS: dict[str, tuple[int, int]] = {
    "arm_qpos": (0, 7),
    "gripper_qpos": (7, 9),
    "cube_xpos": (9, 12),
    "cube_xquat": (12, 16),
    "tray_xpos": (16, 19),
    "tray_xmat": (19, 28),
    "grasp_xpos": (28, 31),
    "grasp_xmat": (31, 40),
}


def load_obs(*, npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Episode file not found: {npz_path}")

    with np.load(npz_path) as data:
        if "obs" not in data:
            raise KeyError(f"{npz_path} does not contain an 'obs' array")
        obs = data["obs"]

    if obs.ndim != 2:
        raise ValueError(f"obs must have shape (T, obs_dim), got {obs.shape}")
    if obs.shape[1] != len(OBS_NAMES):
        raise ValueError(f"expected obs_dim={len(OBS_NAMES)}, got {obs.shape[1]}")
    if not np.isfinite(obs).all():
        raise ValueError("obs contains NaN or Inf values")

    return obs


def print_stats(*, obs: np.ndarray) -> None:
    print(f"obs shape: {obs.shape}")
    print("\nper-dimension stats:")
    print(f"{'dim':>3} {'name':>20} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}")
    for dim, name in enumerate(OBS_NAMES):
        values = obs[:, dim]
        print(
            f"{dim:>3} {name:>20} "
            f"{values.min():>12.6f} {values.max():>12.6f} "
            f"{values.mean():>12.6f} {values.std():>12.6f}"
        )


def save_all_obs_timeseries(*, obs: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(obs.shape[0])
    n_groups = len(OBS_GROUPS)
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3 * n_groups), sharex=True)
    for ax, (group_name, (start, end)) in zip(axes, OBS_GROUPS.items()):
        for dim in range(start, end):
            ax.plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}")
        ax.set_ylabel(group_name)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(9, end - start), fontsize=7)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Observation Timeseries (by group)")
    fig.tight_layout()
    fig.savefig(out_dir / "obs_timeseries_all.png", dpi=160)
    plt.close(fig)


def save_arm_qpos_timeseries(*, obs: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(obs.shape[0])
    fig, ax = plt.subplots(figsize=(14, 6))
    start, end = OBS_GROUPS["arm_qpos"]
    for dim in range(start, end):
        ax.plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    ax.set_title("Arm Joint Position Timeseries")
    ax.set_xlabel("timestep")
    ax.set_ylabel("qpos (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "obs_timeseries_arm_qpos.png", dpi=160)
    plt.close(fig)


def save_gripper_qpos_timeseries(*, obs: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(obs.shape[0])
    fig, ax = plt.subplots(figsize=(14, 4))
    start, end = OBS_GROUPS["gripper_qpos"]
    for dim in range(start, end):
        ax.plot(timesteps, obs[:, dim], linewidth=1.5, label=f"{dim}: {OBS_NAMES[dim]}")
    ax.set_title("Gripper Position Timeseries")
    ax.set_xlabel("timestep")
    ax.set_ylabel("qpos")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "obs_timeseries_gripper_qpos.png", dpi=160)
    plt.close(fig)


def save_cube_timeseries(*, obs: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(obs.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    start, end = OBS_GROUPS["cube_xpos"]
    for dim in range(start, end):
        axes[0].plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    axes[0].set_title("Cube Position")
    axes[0].set_ylabel("position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    start, end = OBS_GROUPS["cube_xquat"]
    for dim in range(start, end):
        axes[1].plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    axes[1].set_title("Cube Orientation (quaternion)")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("quat")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Cube Observation Timeseries")
    fig.tight_layout()
    fig.savefig(out_dir / "obs_timeseries_cube.png", dpi=160)
    plt.close(fig)


def save_tray_timeseries(*, obs: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(obs.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    start, end = OBS_GROUPS["tray_xpos"]
    for dim in range(start, end):
        axes[0].plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    axes[0].set_title("Tray Position")
    axes[0].set_ylabel("position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    start, end = OBS_GROUPS["tray_xmat"]
    for dim in range(start, end):
        axes[1].plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    axes[1].set_title("Tray Rotation Matrix (flattened)")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=7)

    fig.suptitle("Tray Observation Timeseries")
    fig.tight_layout()
    fig.savefig(out_dir / "obs_timeseries_tray.png", dpi=160)
    plt.close(fig)


def save_grasp_timeseries(*, obs: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(obs.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    start, end = OBS_GROUPS["grasp_xpos"]
    for dim in range(start, end):
        axes[0].plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    axes[0].set_title("Grasp Site Position")
    axes[0].set_ylabel("position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    start, end = OBS_GROUPS["grasp_xmat"]
    for dim in range(start, end):
        axes[1].plot(timesteps, obs[:, dim], linewidth=1.0, label=f"{dim}: {OBS_NAMES[dim]}")
    axes[1].set_title("Grasp Site Rotation Matrix (flattened)")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=7)

    fig.suptitle("Grasp Site Observation Timeseries")
    fig.tight_layout()
    fig.savefig(out_dir / "obs_timeseries_grasp.png", dpi=160)
    plt.close(fig)


def save_obs_stats(*, obs: np.ndarray, out_dir: Path) -> None:
    dims = np.arange(obs.shape[1])
    mins = obs.min(axis=0)
    maxs = obs.max(axis=0)
    means = obs.mean(axis=0)
    stds = obs.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(dims - 0.2, mins, width=0.4, label="min")
    axes[0].bar(dims + 0.2, maxs, width=0.4, label="max")
    axes[0].set_title("Obs Min/Max By Dimension")
    axes[0].set_ylabel("value")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(dims - 0.2, means, width=0.4, label="mean")
    axes[1].bar(dims + 0.2, stds, width=0.4, label="std")
    axes[1].set_title("Obs Mean/Std By Dimension")
    axes[1].set_xlabel("observation dimension")
    axes[1].set_ylabel("value")
    axes[1].set_xticks(dims)
    axes[1].set_xticklabels(
        [f"{dim}\n{OBS_NAMES[dim]}" for dim in dims], fontsize=5
    )
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "obs_stats.png", dpi=160)
    plt.close(fig)


def save_obs_histograms(*, obs: np.ndarray, out_dir: Path) -> None:
    n_dims = obs.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    for dim in range(n_dims):
        ax = axes.flat[dim]
        ax.hist(obs[:, dim], bins=40)
        ax.set_title(f"{dim}: {OBS_NAMES[dim]}", fontsize=8)
        ax.grid(True, alpha=0.3)
    for dim in range(n_dims, len(axes.flat)):
        axes.flat[dim].set_visible(False)
    fig.suptitle("Observation Histograms")
    fig.tight_layout()
    fig.savefig(out_dir / "obs_histograms.png", dpi=160)
    plt.close(fig)


def save_gripper_qpos_histogram(*, obs: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    start, end = OBS_GROUPS["gripper_qpos"]
    for i, dim in enumerate(range(start, end)):
        axes[i].hist(obs[:, dim], bins=40)
        axes[i].set_title(f"{dim}: {OBS_NAMES[dim]}")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Gripper Position Histograms")
    fig.tight_layout()
    fig.savefig(out_dir / "obs_histogram_gripper_qpos.png", dpi=160)
    plt.close(fig)


def visualize_obs(*, npz_path: Path, out_dir: Path) -> None:
    obs = load_obs(npz_path=npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_stats(obs=obs)
    save_all_obs_timeseries(obs=obs, out_dir=out_dir)
    save_arm_qpos_timeseries(obs=obs, out_dir=out_dir)
    save_gripper_qpos_timeseries(obs=obs, out_dir=out_dir)
    save_cube_timeseries(obs=obs, out_dir=out_dir)
    save_tray_timeseries(obs=obs, out_dir=out_dir)
    save_grasp_timeseries(obs=obs, out_dir=out_dir)
    save_obs_stats(obs=obs, out_dir=out_dir)
    save_obs_histograms(obs=obs, out_dir=out_dir)
    save_gripper_qpos_histogram(obs=obs, out_dir=out_dir)

    print(f"\nsaved plots to: {out_dir}")


def default_output_dir(*, npz_path: Path, data_dir: Path, out_root: Path) -> Path:
    npz_path = npz_path.resolve()
    data_dir = data_dir.resolve()

    try:
        relative_path = npz_path.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError(f"{npz_path} is not inside data directory {data_dir}") from exc

    return out_root / relative_path.with_suffix("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize observation data from one episode .npz file."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to episode .npz file containing an obs array.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory used to compute the relative episode path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("plots/obs"),
        help="Root directory where obs plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = default_output_dir(
        npz_path=args.npz,
        data_dir=args.data_dir,
        out_root=args.out_root,
    )
    visualize_obs(npz_path=args.npz, out_dir=out_dir)


if __name__ == "__main__":
    main()
