import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRIPPER_ACTION_DIM = 7


def list_npz_files(*, npz_dir: Path) -> list[Path]:
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ folder not found: {npz_dir}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"Expected a folder, got: {npz_dir}")

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

    return npz_paths


def load_folder_stats(*, npz_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    frame_counts = []
    gripper_actions = []

    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            if "actions" not in data:
                raise KeyError(f"{npz_path} does not contain an 'actions' array")
            actions = data["actions"]

        if actions.ndim != 2:
            raise ValueError(f"actions must have shape (T, action_dim), got {actions.shape}")
        if actions.shape[1] <= GRIPPER_ACTION_DIM:
            raise ValueError(
                f"expected gripper action at dim {GRIPPER_ACTION_DIM}, got action_dim={actions.shape[1]}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"{npz_path} actions contains NaN or Inf values")

        frame_counts.append(actions.shape[0])
        gripper_actions.append(actions[:, GRIPPER_ACTION_DIM])

    return np.asarray(frame_counts), np.concatenate(gripper_actions)


def print_stats(*, npz_paths: list[Path], frame_counts: np.ndarray, gripper_actions: np.ndarray) -> None:
    print(f"num files: {len(npz_paths)}")
    print(
        "frame counts: "
        f"min={frame_counts.min()} max={frame_counts.max()} "
        f"mean={frame_counts.mean():.2f} std={frame_counts.std():.2f}"
    )
    print(
        "gripper actions: "
        f"min={gripper_actions.min():.6f} max={gripper_actions.max():.6f} "
        f"mean={gripper_actions.mean():.6f} std={gripper_actions.std():.6f}"
    )


def save_frame_count_histogram(*, frame_counts: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(frame_counts, bins=min(40, max(1, len(frame_counts))))
    ax.set_title("Frames Per NPZ File")
    ax.set_xlabel("number of frames")
    ax.set_ylabel("file count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "frame_count_histogram.png", dpi=160)
    plt.close(fig)


def save_gripper_action_histogram(*, gripper_actions: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gripper_actions, bins=40)
    ax.set_title("Gripper Action Values")
    ax.set_xlabel("gripper action")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gripper_action_histogram.png", dpi=160)
    plt.close(fig)


def visualize_npz_folder(*, npz_dir: Path, out_dir: Path) -> None:
    npz_paths = list_npz_files(npz_dir=npz_dir)
    frame_counts, gripper_actions = load_folder_stats(npz_paths=npz_paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_stats(
        npz_paths=npz_paths,
        frame_counts=frame_counts,
        gripper_actions=gripper_actions,
    )
    save_frame_count_histogram(frame_counts=frame_counts, out_dir=out_dir)
    save_gripper_action_histogram(gripper_actions=gripper_actions, out_dir=out_dir)

    print(f"\nsaved plots to: {out_dir}")


def default_output_dir(*, npz_dir: Path, data_dir: Path, out_root: Path) -> Path:
    npz_dir = npz_dir.resolve()
    data_dir = data_dir.resolve()

    try:
        relative_path = npz_dir.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError(f"{npz_dir} is not inside data directory {data_dir}") from exc

    return out_root / relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize summary stats for a folder of .npz files.")
    parser.add_argument(
        "--npz",
        "--npz-dir",
        dest="npz_dir",
        type=Path,
        required=True,
        help="Path to a folder containing episode .npz files with actions arrays.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory used to compute the relative folder path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("plots/npz-dataset"),
        help="Root directory where dataset plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = default_output_dir(
        npz_dir=args.npz_dir,
        data_dir=args.data_dir,
        out_root=args.out_root,
    )
    visualize_npz_folder(npz_dir=args.npz_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
