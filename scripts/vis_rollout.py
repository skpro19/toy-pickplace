import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARM_JOINT_DIMS = 7
JOINT_NAMES = [f"joint_{idx}" for idx in range(ARM_JOINT_DIMS)]


def load_metadata(*, rollout_dir: Path) -> dict[str, object]:
    metadata_path = rollout_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_out_dir(*, rollout_dir: Path, out_root: Path) -> Path:
    return out_root / rollout_dir.name


def load_episode_obs(*, episode_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(episode_path) as data:
        missing_keys = {"rollout_obs", "train_obs"} - set(data.files)
        if missing_keys:
            raise KeyError(f"{episode_path} is missing keys: {sorted(missing_keys)}")
        rollout_obs = np.asarray(data["rollout_obs"], dtype=np.float32)
        train_obs = np.asarray(data["train_obs"], dtype=np.float32)

    for name, obs in (("rollout_obs", rollout_obs), ("train_obs", train_obs)):
        if obs.ndim != 2:
            raise ValueError(f"{episode_path} {name} must have shape (T, obs_dim), got {obs.shape}")
        if obs.shape[1] < ARM_JOINT_DIMS:
            raise ValueError(
                f"{episode_path} {name} must have at least {ARM_JOINT_DIMS} dims, got {obs.shape[1]}"
            )
        if not np.isfinite(obs).all():
            raise ValueError(f"{episode_path} {name} contains NaN or Inf values")

    return rollout_obs, train_obs


def save_arm_timeseries_overlay(
    *,
    rollout_obs: np.ndarray,
    train_obs: np.ndarray,
    out_dir: Path,
) -> None:
    rollout_steps = np.arange(rollout_obs.shape[0])
    train_steps = np.arange(train_obs.shape[0])

    fig, axes = plt.subplots(ARM_JOINT_DIMS, 1, figsize=(14, 18), sharex=True)
    for joint_idx, ax in enumerate(axes):
        ax.plot(
            train_steps,
            train_obs[:, joint_idx],
            linewidth=1.1,
            alpha=0.85,
            label="train",
        )
        ax.plot(
            rollout_steps,
            rollout_obs[:, joint_idx],
            linewidth=1.1,
            alpha=0.85,
            label="rollout",
        )
        ax.set_ylabel(JOINT_NAMES[joint_idx])
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend()

    axes[-1].set_xlabel("timestep")
    fig.suptitle("Arm Joint Observation Overlay")
    fig.tight_layout()
    fig.savefig(out_dir / "arm_qpos_overlay.png", dpi=160)
    plt.close(fig)


def save_episode_plots(*, episode_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_obs, train_obs = load_episode_obs(episode_path=episode_path)
    if rollout_obs.shape[0] == 0 or train_obs.shape[0] == 0:
        return

    save_arm_timeseries_overlay(
        rollout_obs=rollout_obs,
        train_obs=train_obs,
        out_dir=out_dir,
    )


def load_all_arm_obs(*, episode_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    rollout_values = []
    train_values = []
    for episode_path in episode_paths:
        rollout_obs, train_obs = load_episode_obs(episode_path=episode_path)
        if rollout_obs.shape[0] > 0:
            rollout_values.append(rollout_obs[:, :ARM_JOINT_DIMS])
        if train_obs.shape[0] > 0:
            train_values.append(train_obs[:, :ARM_JOINT_DIMS])

    if not rollout_values:
        raise ValueError("No rollout observations found in episode logs.")
    if not train_values:
        raise ValueError("No train observations found in episode logs.")

    return np.concatenate(rollout_values, axis=0), np.concatenate(train_values, axis=0)


def save_arm_distribution_overlay(
    *,
    rollout_arm_obs: np.ndarray,
    train_arm_obs: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for joint_idx, ax in enumerate(axes.flat):
        if joint_idx >= ARM_JOINT_DIMS:
            ax.set_visible(False)
            continue
        ax.hist(
            train_arm_obs[:, joint_idx],
            bins=60,
            alpha=0.55,
            density=True,
            label="train",
        )
        ax.hist(
            rollout_arm_obs[:, joint_idx],
            bins=60,
            alpha=0.55,
            density=True,
            label="rollout",
        )
        ax.set_title(JOINT_NAMES[joint_idx])
        ax.set_xlabel("qpos")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend()

    fig.suptitle("Arm Joint Observation Distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "arm_qpos_distribution.png", dpi=160)
    plt.close(fig)


def save_summary_plots(*, episode_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_arm_obs, train_arm_obs = load_all_arm_obs(episode_paths=episode_paths)
    save_arm_distribution_overlay(
        rollout_arm_obs=rollout_arm_obs,
        train_arm_obs=train_arm_obs,
        out_dir=out_dir,
    )


def visualize_rollout(*, rollout_dir: Path, out_root: Path) -> None:
    if not rollout_dir.exists():
        raise FileNotFoundError(f"Rollout directory not found: {rollout_dir}")

    metadata = load_metadata(rollout_dir=rollout_dir)
    out_dir = default_out_dir(rollout_dir=rollout_dir, out_root=out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if metadata:
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    episode_paths = sorted(rollout_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.npz logs found in {rollout_dir}")

    for episode_path in episode_paths:
        save_episode_plots(
            episode_path=episode_path,
            out_dir=out_dir / episode_path.stem,
        )
    save_summary_plots(episode_paths=episode_paths, out_dir=out_dir / "summary")
    print(f"Saved rollout observation plots to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize train vs rollout observation logs.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("plots/rollouts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_rollout(
        rollout_dir=args.rollout_dir,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
