import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from constant import MAX_ARM_DELTA


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


def validate_timeseries(
    *,
    episode_path: Path,
    name: str,
    values: np.ndarray,
    min_dims: int,
) -> None:
    if values.ndim != 2:
        raise ValueError(f"{episode_path} {name} must have shape (T, dim), got {values.shape}")
    if values.shape[1] < min_dims:
        raise ValueError(
            f"{episode_path} {name} must have at least {min_dims} dims, got {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{episode_path} {name} contains NaN or Inf values")


def load_episode_arrays(*, episode_path: Path) -> dict[str, np.ndarray]:
    with np.load(episode_path) as data:
        missing_keys = {"rollout_obs", "train_obs"} - set(data.files)
        if missing_keys:
            raise KeyError(f"{episode_path} is missing keys: {sorted(missing_keys)}")
        arrays = {name: np.asarray(data[name], dtype=np.float32) for name in data.files}

    for name in ("rollout_obs", "train_obs"):
        validate_timeseries(
            episode_path=episode_path,
            name=name,
            values=arrays[name],
            min_dims=ARM_JOINT_DIMS,
        )

    for name in ("rollout_actions", "train_actions"):
        if name in arrays:
            validate_timeseries(
                episode_path=episode_path,
                name=name,
                values=arrays[name],
                min_dims=ARM_JOINT_DIMS,
            )

    for name in ("rollout_action_deltas", "train_action_deltas"):
        if name in arrays:
            validate_timeseries(
                episode_path=episode_path,
                name=name,
                values=arrays[name],
                min_dims=ARM_JOINT_DIMS,
            )

    return arrays


def save_arm_timeseries(
    *,
    values: np.ndarray,
    title: str,
    ylabel: str,
    filename: str,
    out_dir: Path,
) -> None:
    steps = np.arange(values.shape[0])

    fig, axes = plt.subplots(ARM_JOINT_DIMS, 1, figsize=(14, 18), sharex=True)
    for joint_idx, ax in enumerate(axes):
        ax.plot(
            steps,
            values[:, joint_idx],
            linewidth=1.1,
            alpha=0.9,
        )
        ax.set_ylabel(JOINT_NAMES[joint_idx])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.supylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def save_arm_timeseries_overlay(
    *,
    rollout_values: np.ndarray,
    train_values: np.ndarray,
    title: str,
    ylabel: str,
    filename: str,
    out_dir: Path,
) -> None:
    rollout_steps = np.arange(rollout_values.shape[0])
    train_steps = np.arange(train_values.shape[0])

    fig, axes = plt.subplots(ARM_JOINT_DIMS, 1, figsize=(14, 18), sharex=True)
    for joint_idx, ax in enumerate(axes):
        ax.plot(
            train_steps,
            train_values[:, joint_idx],
            linewidth=1.1,
            alpha=0.85,
            label="train",
        )
        ax.plot(
            rollout_steps,
            rollout_values[:, joint_idx],
            linewidth=1.1,
            alpha=0.85,
            label="rollout",
        )
        ax.set_ylabel(JOINT_NAMES[joint_idx])
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend()

    axes[-1].set_xlabel("timestep")
    fig.supylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def has_keys(*, arrays: dict[str, np.ndarray], keys: tuple[str, ...]) -> bool:
    return all(key in arrays for key in keys)


def save_delta_threshold_overlay(
    *,
    rollout_deltas: np.ndarray,
    train_deltas: np.ndarray,
    filename: str,
    out_dir: Path,
) -> None:
    rollout_abs_max = np.max(np.abs(rollout_deltas[:, :ARM_JOINT_DIMS]), axis=1)
    train_abs = np.abs(train_deltas[:, :ARM_JOINT_DIMS]).reshape(-1)
    p95 = float(np.percentile(train_abs, 95))
    p99 = float(np.percentile(train_abs, 99))
    train_max = float(np.max(train_abs))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        np.arange(rollout_abs_max.shape[0]),
        rollout_abs_max,
        linewidth=1.2,
        label="rollout max |delta|",
    )
    ax.axhline(p95, color="tab:orange", linestyle="--", linewidth=1.1, label="train p95")
    ax.axhline(p99, color="tab:red", linestyle="--", linewidth=1.1, label="train p99")
    ax.axhline(train_max, color="black", linestyle=":", linewidth=1.1, label="train max")
    ax.set_title("Rollout Delta Magnitude vs Train Thresholds")
    ax.set_xlabel("timestep")
    ax.set_ylabel("max abs joint delta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def first_exceedance_step(*, rollout_abs: np.ndarray, thresholds: np.ndarray) -> int:
    exceeds = np.any(rollout_abs > thresholds, axis=1)
    if not np.any(exceeds):
        return -1
    return int(np.argmax(exceeds))


def save_episode_plots(*, episode_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_episode_arrays(episode_path=episode_path)
    if arrays["rollout_obs"].shape[0] == 0 or arrays["train_obs"].shape[0] == 0:
        return

    save_arm_timeseries_overlay(
        rollout_values=arrays["rollout_obs"],
        train_values=arrays["train_obs"],
        title="Arm Joint Observation Overlay",
        ylabel="qpos",
        filename="arm_qpos_overlay.png",
        out_dir=out_dir,
    )
    if has_keys(arrays=arrays, keys=("rollout_actions", "train_actions")):
        save_arm_timeseries_overlay(
            rollout_values=arrays["rollout_actions"],
            train_values=arrays["train_actions"],
            title="Arm Joint Action Overlay",
            ylabel="ctrl",
            filename="arm_actions_overlay.png",
            out_dir=out_dir,
        )
    if has_keys(arrays=arrays, keys=("rollout_action_deltas", "train_action_deltas")):
        save_arm_timeseries_overlay(
            rollout_values=arrays["rollout_action_deltas"],
            train_values=arrays["train_action_deltas"],
            title="Arm Joint Action Delta Overlay",
            ylabel="ctrl - qpos",
            filename="arm_action_deltas_overlay.png",
            out_dir=out_dir,
        )
        save_delta_threshold_overlay(
            rollout_deltas=arrays["rollout_action_deltas"],
            train_deltas=arrays["train_action_deltas"],
            filename="arm_action_delta_thresholds.png",
            out_dir=out_dir,
        )
    if "joints_pred_raw" in arrays:
        save_arm_timeseries(
            values=arrays["joints_pred_raw"],
            title="Raw Joint Model Outputs",
            ylabel="raw model output",
            filename="joints_pred_raw.png",
            out_dir=out_dir,
        )
    if "joints_pred_unnorm" in arrays:
        save_arm_timeseries(
            values=arrays["joints_pred_unnorm"],
            title="Unnormalized Joint Model Outputs",
            ylabel="delta or absolute ctrl",
            filename="joints_pred_unnorm.png",
            out_dir=out_dir,
        )


def load_all_arm_values(
    *,
    episode_paths: list[Path],
    rollout_key: str,
    train_key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    rollout_values = []
    train_values = []
    for episode_path in episode_paths:
        arrays = load_episode_arrays(episode_path=episode_path)
        if not has_keys(arrays=arrays, keys=(rollout_key, train_key)):
            return None
        if arrays[rollout_key].shape[0] > 0:
            rollout_values.append(arrays[rollout_key][:, :ARM_JOINT_DIMS])
        if arrays[train_key].shape[0] > 0:
            train_values.append(arrays[train_key][:, :ARM_JOINT_DIMS])

    if not rollout_values or not train_values:
        return None

    return np.concatenate(rollout_values, axis=0), np.concatenate(train_values, axis=0)


def save_arm_distribution_overlay(
    *,
    rollout_values: np.ndarray,
    train_values: np.ndarray,
    title: str,
    xlabel: str,
    filename: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for joint_idx, ax in enumerate(axes.flat):
        if joint_idx >= ARM_JOINT_DIMS:
            ax.set_visible(False)
            continue
        train_joint = train_values[:, joint_idx]
        ax.hist(
            train_joint,
            bins=60,
            alpha=0.55,
            density=True,
            label="train",
        )
        ax.hist(
            rollout_values[:, joint_idx],
            bins=60,
            alpha=0.55,
            density=True,
            label="rollout",
        )
        p05 = float(np.percentile(train_joint, 5))
        p95 = float(np.percentile(train_joint, 95))
        train_min = float(np.min(train_joint))
        train_max = float(np.max(train_joint))
        ax.axvline(train_min, color="black", linestyle=":", linewidth=0.9, label="train min/max" if joint_idx == 0 else None)
        ax.axvline(train_max, color="black", linestyle=":", linewidth=0.9)
        ax.axvline(p05, color="tab:orange", linestyle="--", linewidth=0.9, label="train p5/p95" if joint_idx == 0 else None)
        ax.axvline(p95, color="tab:orange", linestyle="--", linewidth=0.9)
        ax.set_title(JOINT_NAMES[joint_idx])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def save_arm_distribution_single(
    *,
    values: np.ndarray,
    reference_values: np.ndarray | None = None,
    title: str,
    xlabel: str,
    filename: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for joint_idx, ax in enumerate(axes.flat):
        if joint_idx >= ARM_JOINT_DIMS:
            ax.set_visible(False)
            continue
        ax.hist(
            values[:, joint_idx],
            bins=60,
            alpha=0.75,
            density=True,
        )
        if reference_values is not None:
            ref_abs = np.abs(reference_values[:, joint_idx])
            p95 = float(np.percentile(ref_abs, 95))
            p99 = float(np.percentile(ref_abs, 99))
            ax.axvline(-p95, color="tab:orange", linestyle="--", linewidth=0.9, label="ref abs p95" if joint_idx == 0 else None)
            ax.axvline(p95, color="tab:orange", linestyle="--", linewidth=0.9)
            ax.axvline(-p99, color="tab:red", linestyle="--", linewidth=0.9, label="ref abs p99" if joint_idx == 0 else None)
            ax.axvline(p99, color="tab:red", linestyle="--", linewidth=0.9)
            ax.axvline(-MAX_ARM_DELTA, color="black", linestyle=":", linewidth=0.9, label="MAX_ARM_DELTA" if joint_idx == 0 else None)
            ax.axvline(MAX_ARM_DELTA, color="black", linestyle=":", linewidth=0.9)
        ax.set_title(JOINT_NAMES[joint_idx])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if joint_idx == 0 and reference_values is not None:
            ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def save_model_vs_applied_delta_distribution(
    *,
    train_deltas: np.ndarray,
    pred_deltas: np.ndarray,
    applied_deltas: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for joint_idx, ax in enumerate(axes.flat):
        if joint_idx >= ARM_JOINT_DIMS:
            ax.set_visible(False)
            continue
        ax.hist(
            train_deltas[:, joint_idx],
            bins=60,
            alpha=0.45,
            density=True,
            label="train",
        )
        ax.hist(
            pred_deltas[:, joint_idx],
            bins=60,
            alpha=0.45,
            density=True,
            label="model unclipped",
        )
        ax.hist(
            applied_deltas[:, joint_idx],
            bins=60,
            alpha=0.45,
            density=True,
            label="applied",
        )
        train_abs = np.abs(train_deltas[:, joint_idx])
        p95 = float(np.percentile(train_abs, 95))
        p99 = float(np.percentile(train_abs, 99))
        ax.axvline(-p95, color="tab:orange", linestyle="--", linewidth=0.9, label="train abs p95" if joint_idx == 0 else None)
        ax.axvline(p95, color="tab:orange", linestyle="--", linewidth=0.9)
        ax.axvline(-p99, color="tab:red", linestyle="--", linewidth=0.9, label="train abs p99" if joint_idx == 0 else None)
        ax.axvline(p99, color="tab:red", linestyle="--", linewidth=0.9)
        ax.axvline(-MAX_ARM_DELTA, color="black", linestyle=":", linewidth=1.0)
        ax.axvline(MAX_ARM_DELTA, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(JOINT_NAMES[joint_idx])
        ax.set_xlabel("joint delta")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Train vs Model vs Applied Joint Delta Distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "model_vs_applied_delta_distribution.png", dpi=160)
    plt.close(fig)


def save_max_abs_pred_delta_distribution(
    *,
    pred_deltas: np.ndarray,
    train_deltas: np.ndarray | None = None,
    out_dir: Path,
) -> None:
    max_abs = np.max(np.abs(pred_deltas[:, :ARM_JOINT_DIMS]), axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_abs, bins=80, alpha=0.75, density=True)
    if train_deltas is not None:
        train_max_abs = np.max(np.abs(train_deltas[:, :ARM_JOINT_DIMS]), axis=1)
        ax.axvline(float(np.percentile(train_max_abs, 95)), color="tab:orange", linestyle="--", linewidth=1.1, label="train p95")
        ax.axvline(float(np.percentile(train_max_abs, 99)), color="tab:red", linestyle="--", linewidth=1.1, label="train p99")
        ax.axvline(float(np.max(train_max_abs)), color="black", linestyle=":", linewidth=1.1, label="train max")
    ax.axvline(MAX_ARM_DELTA, color="tab:red", linestyle="--", linewidth=1.2, label="MAX_ARM_DELTA")
    ax.set_title("Max Absolute Predicted Delta Distribution")
    ax.set_xlabel("max abs predicted delta across joints")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "max_abs_pred_delta_distribution.png", dpi=160)
    plt.close(fig)


def save_clipping_fraction_by_joint(
    *,
    pred_deltas: np.ndarray,
    out_dir: Path,
) -> None:
    clipped = np.abs(pred_deltas[:, :ARM_JOINT_DIMS]) > MAX_ARM_DELTA
    fractions = np.mean(clipped, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(ARM_JOINT_DIMS), fractions)
    ax.set_title("Clipping Fraction By Joint")
    ax.set_xlabel("joint")
    ax.set_ylabel("fraction of timesteps clipped")
    ax.set_xticks(np.arange(ARM_JOINT_DIMS))
    ax.set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clipping_fraction_by_joint.png", dpi=160)
    plt.close(fig)


def save_clipping_summary(*, pred_deltas: np.ndarray, out_dir: Path) -> None:
    clipped = np.abs(pred_deltas[:, :ARM_JOINT_DIMS]) > MAX_ARM_DELTA
    rows = []
    for joint_idx in range(ARM_JOINT_DIMS):
        rows.append(
            {
                "joint": JOINT_NAMES[joint_idx],
                "clipped_count": int(np.sum(clipped[:, joint_idx])),
                "total_steps": int(clipped.shape[0]),
                "clipped_fraction": float(np.mean(clipped[:, joint_idx])),
                "max_abs_pred_delta": float(np.max(np.abs(pred_deltas[:, joint_idx]))),
            }
        )

    any_clipped = np.any(clipped, axis=1)
    rows.append(
        {
            "joint": "any_joint",
            "clipped_count": int(np.sum(any_clipped)),
            "total_steps": int(clipped.shape[0]),
            "clipped_fraction": float(np.mean(any_clipped)),
            "max_abs_pred_delta": float(np.max(np.abs(pred_deltas[:, :ARM_JOINT_DIMS]))),
        }
    )

    with (out_dir / "clipping_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_first_ood_summary(*, episode_paths: list[Path], out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for episode_path in episode_paths:
        arrays = load_episode_arrays(episode_path=episode_path)
        if not has_keys(arrays=arrays, keys=("rollout_action_deltas", "train_action_deltas")):
            continue

        rollout_abs = np.abs(arrays["rollout_action_deltas"][:, :ARM_JOINT_DIMS])
        train_abs = np.abs(arrays["train_action_deltas"][:, :ARM_JOINT_DIMS])
        train_p95 = np.percentile(train_abs, 95, axis=0)
        train_p99 = np.percentile(train_abs, 99, axis=0)
        train_max = np.max(train_abs, axis=0)

        rows.append(
            {
                "episode": episode_path.stem,
                "steps": int(rollout_abs.shape[0]),
                "first_step_gt_train_p95": first_exceedance_step(
                    rollout_abs=rollout_abs,
                    thresholds=train_p95,
                ),
                "first_step_gt_train_p99": first_exceedance_step(
                    rollout_abs=rollout_abs,
                    thresholds=train_p99,
                ),
                "first_step_gt_train_max": first_exceedance_step(
                    rollout_abs=rollout_abs,
                    thresholds=train_max,
                ),
                "max_rollout_abs_delta": float(np.max(rollout_abs)),
                "train_p95_abs_delta_max_joint": float(np.max(train_p95)),
                "train_p99_abs_delta_max_joint": float(np.max(train_p99)),
                "train_max_abs_delta": float(np.max(train_max)),
            }
        )

    if not rows:
        return

    summary_path = out_dir / "first_ood_timestep.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plots(*, episode_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_values = load_all_arm_values(
        episode_paths=episode_paths,
        rollout_key="rollout_obs",
        train_key="train_obs",
    )
    if obs_values is None:
        raise ValueError("No observation values found in episode logs.")
    save_arm_distribution_overlay(
        rollout_values=obs_values[0],
        train_values=obs_values[1],
        title="Arm Joint Observation Distributions",
        xlabel="qpos",
        filename="arm_qpos_distribution.png",
        out_dir=out_dir,
    )

    action_values = load_all_arm_values(
        episode_paths=episode_paths,
        rollout_key="rollout_actions",
        train_key="train_actions",
    )
    if action_values is not None:
        save_arm_distribution_overlay(
            rollout_values=action_values[0],
            train_values=action_values[1],
            title="Arm Joint Action Distributions",
            xlabel="ctrl",
            filename="arm_actions_distribution.png",
            out_dir=out_dir,
        )

    delta_values = load_all_arm_values(
        episode_paths=episode_paths,
        rollout_key="rollout_action_deltas",
        train_key="train_action_deltas",
    )
    if delta_values is not None:
        save_arm_distribution_overlay(
            rollout_values=delta_values[0],
            train_values=delta_values[1],
            title="Arm Joint Action Delta Distributions",
            xlabel="ctrl - qpos",
            filename="arm_action_deltas_distribution.png",
            out_dir=out_dir,
        )

    pred_raw_values = []
    pred_unnorm_values = []
    for episode_path in episode_paths:
        arrays = load_episode_arrays(episode_path=episode_path)
        if "joints_pred_raw" in arrays and arrays["joints_pred_raw"].shape[0] > 0:
            pred_raw_values.append(arrays["joints_pred_raw"][:, :ARM_JOINT_DIMS])
        if "joints_pred_unnorm" in arrays and arrays["joints_pred_unnorm"].shape[0] > 0:
            pred_unnorm_values.append(arrays["joints_pred_unnorm"][:, :ARM_JOINT_DIMS])

    pred_raw = np.concatenate(pred_raw_values, axis=0) if pred_raw_values else None
    pred_unnorm = (
        np.concatenate(pred_unnorm_values, axis=0) if pred_unnorm_values else None
    )

    if pred_raw is not None:
        save_arm_distribution_single(
            values=pred_raw,
            reference_values=None,
            title="Raw Joint Model Output Distributions",
            xlabel="raw model output",
            filename="joints_pred_raw_distribution.png",
            out_dir=out_dir,
        )

    if pred_unnorm is not None:
        save_arm_distribution_single(
            values=pred_unnorm,
            reference_values=delta_values[1] if delta_values is not None else None,
            title="Unnormalized Joint Model Output Distributions",
            xlabel="delta or absolute ctrl",
            filename="joints_pred_unnorm_distribution.png",
            out_dir=out_dir,
        )
        save_max_abs_pred_delta_distribution(
            pred_deltas=pred_unnorm,
            train_deltas=delta_values[1] if delta_values is not None else None,
            out_dir=out_dir,
        )
        save_clipping_fraction_by_joint(pred_deltas=pred_unnorm, out_dir=out_dir)
        save_clipping_summary(pred_deltas=pred_unnorm, out_dir=out_dir)

    if delta_values is not None and pred_unnorm is not None:
        save_model_vs_applied_delta_distribution(
            train_deltas=delta_values[1],
            pred_deltas=pred_unnorm,
            applied_deltas=delta_values[0],
            out_dir=out_dir,
        )
    save_first_ood_summary(episode_paths=episode_paths, out_dir=out_dir)


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
    print(f"Saved rollout plots to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize train vs rollout logs.")
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
