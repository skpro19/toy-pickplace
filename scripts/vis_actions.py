import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = [
    "arm_0",
    "arm_1",
    "arm_2",
    "arm_3",
    "arm_4",
    "arm_5",
    "arm_6",
    "gripper",
]


def load_actions(*, npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Episode file not found: {npz_path}")

    with np.load(npz_path) as data:
        if "actions" not in data:
            raise KeyError(f"{npz_path} does not contain an 'actions' array")
        actions = data["actions"]

    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, action_dim), got {actions.shape}")
    if actions.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"expected action_dim={len(ACTION_NAMES)}, got {actions.shape[1]}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contains NaN or Inf values")

    return actions


def print_stats(*, actions: np.ndarray) -> None:
    print(f"actions shape: {actions.shape}")
    print("\nper-dimension stats:")
    print(f"{'dim':>3} {'name':>8} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}")
    for dim, name in enumerate(ACTION_NAMES):
        values = actions[:, dim]
        print(
            f"{dim:>3} {name:>8} "
            f"{values.min():>12.6f} {values.max():>12.6f} "
            f"{values.mean():>12.6f} {values.std():>12.6f}"
        )


def save_all_actions_timeseries(*, actions: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(actions.shape[0])

    fig, axes = plt.subplots(8, 1, figsize=(14, 14), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(timesteps, actions[:, dim], linewidth=1.0)
        ax.set_ylabel(f"{dim}: {ACTION_NAMES[dim]}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Action Timeseries")
    fig.tight_layout()
    fig.savefig(out_dir / "action_timeseries_all.png", dpi=160)
    plt.close(fig)


def save_arm_actions_timeseries(*, actions: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(actions.shape[0])

    fig, ax = plt.subplots(figsize=(14, 6))
    for dim in range(7):
        ax.plot(timesteps, actions[:, dim], linewidth=1.0, label=f"{dim}: {ACTION_NAMES[dim]}")
    ax.set_title("Arm Action Timeseries")
    ax.set_xlabel("timestep")
    ax.set_ylabel("control")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "action_timeseries_arm.png", dpi=160)
    plt.close(fig)


def save_gripper_timeseries(*, actions: np.ndarray, out_dir: Path) -> None:
    timesteps = np.arange(actions.shape[0])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(timesteps, actions[:, 7], where="post", linewidth=1.5)
    ax.set_title("Gripper Action Timeseries")
    ax.set_xlabel("timestep")
    ax.set_ylabel("gripper control")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "action_timeseries_gripper.png", dpi=160)
    plt.close(fig)


def save_action_stats(*, actions: np.ndarray, out_dir: Path) -> None:
    dims = np.arange(actions.shape[1])
    mins = actions.min(axis=0)
    maxs = actions.max(axis=0)
    means = actions.mean(axis=0)
    stds = actions.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(dims - 0.2, mins, width=0.4, label="min")
    axes[0].bar(dims + 0.2, maxs, width=0.4, label="max")
    axes[0].set_title("Action Min/Max By Dimension")
    axes[0].set_ylabel("value")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(dims - 0.2, means, width=0.4, label="mean")
    axes[1].bar(dims + 0.2, stds, width=0.4, label="std")
    axes[1].set_title("Action Mean/Std By Dimension")
    axes[1].set_xlabel("action dimension")
    axes[1].set_ylabel("value")
    axes[1].set_xticks(dims)
    axes[1].set_xticklabels([f"{dim}\n{name}" for dim, name in enumerate(ACTION_NAMES)])
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "action_stats.png", dpi=160)
    plt.close(fig)


def save_action_histograms(*, actions: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for dim, ax in enumerate(axes.flat):
        ax.hist(actions[:, dim], bins=40)
        ax.set_title(f"{dim}: {ACTION_NAMES[dim]}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Action Histograms")
    fig.tight_layout()
    fig.savefig(out_dir / "action_histograms.png", dpi=160)
    plt.close(fig)


def visualize_actions(*, npz_path: Path, out_dir: Path) -> None:
    actions = load_actions(npz_path=npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_stats(actions=actions)
    save_all_actions_timeseries(actions=actions, out_dir=out_dir)
    save_arm_actions_timeseries(actions=actions, out_dir=out_dir)
    save_gripper_timeseries(actions=actions, out_dir=out_dir)
    save_action_stats(actions=actions, out_dir=out_dir)
    save_action_histograms(actions=actions, out_dir=out_dir)

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
    parser = argparse.ArgumentParser(description="Visualize action data from one episode .npz file.")
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to episode .npz file containing an actions array.",
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
        default=Path("plots/actions"),
        help="Root directory where action plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = default_output_dir(
        npz_path=args.npz,
        data_dir=args.data_dir,
        out_root=args.out_root,
    )
    visualize_actions(npz_path=args.npz, out_dir=out_dir)


if __name__ == "__main__":
    main()
