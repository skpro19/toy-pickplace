"""Dump frame grids from image episodes to verify alignment and coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from constant import IMAGE_HEIGHT, IMAGE_WIDTH


MIN_COLOR_PIXELS = 3


def list_npz_files(*, npz_dir: Path) -> list[Path]:
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ folder not found: {npz_dir}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"Expected a folder, got: {npz_dir}")

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

    return npz_paths


def count_red_pixels(*, image: np.ndarray) -> int:
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    return int(np.sum((red >= 100) & (green <= 90) & (blue <= 90)))


def count_blue_pixels(*, image: np.ndarray) -> int:
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    return int(np.sum((blue >= 100) & (red <= 90) & (green <= 110)))


def sample_frame_indices(*, num_frames: int, frames_per_episode: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError(f"episode must contain at least one frame, got {num_frames}")
    if frames_per_episode <= 0:
        raise ValueError(f"frames_per_episode must be positive, got {frames_per_episode}")

    count = min(frames_per_episode, num_frames)
    if count == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, num_frames - 1, num=count, dtype=int)


def validate_episode_arrays(
    *,
    npz_path: Path,
    img_obs: np.ndarray,
    obs: np.ndarray,
    actions: np.ndarray,
) -> None:
    expected_image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    if img_obs.ndim != 4:
        raise ValueError(
            f"{npz_path}: img_obs must have shape (T, H, W, 3), got {img_obs.shape}"
        )
    if img_obs.shape[1:] != expected_image_shape:
        raise ValueError(
            f"{npz_path}: expected image shape {expected_image_shape}, got {img_obs.shape[1:]}"
        )
    if img_obs.dtype != np.uint8:
        raise ValueError(f"{npz_path}: expected img_obs dtype uint8, got {img_obs.dtype}")
    if obs.shape[0] != img_obs.shape[0]:
        raise ValueError(
            f"{npz_path}: obs length {obs.shape[0]} != img_obs length {img_obs.shape[0]}"
        )
    if actions.shape[0] != img_obs.shape[0]:
        raise ValueError(
            f"{npz_path}: actions length {actions.shape[0]} != img_obs length {img_obs.shape[0]}"
        )


def save_episode_grid(
    *,
    npz_path: Path,
    img_obs: np.ndarray,
    frame_indices: np.ndarray,
    out_path: Path,
) -> None:
    rows = len(frame_indices)
    fig, axes = plt.subplots(rows, 1, figsize=(3, rows * 3))
    if rows == 1:
        axes = [axes]

    for axis, frame_idx in zip(axes, frame_indices, strict=True):
        frame = img_obs[frame_idx]
        axis.imshow(frame)
        axis.set_title(
            f"t={frame_idx}  red={count_red_pixels(image=frame)}  "
            f"blue={count_blue_pixels(image=frame)}"
        )
        axis.axis("off")

    fig.suptitle(npz_path.name, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize_episode(*, npz_path: Path, img_obs: np.ndarray, frame_indices: np.ndarray) -> None:
    red_counts = [count_red_pixels(image=img_obs[idx]) for idx in frame_indices]
    blue_counts = [count_blue_pixels(image=img_obs[idx]) for idx in frame_indices]
    print(
        f"{npz_path.name}: T={img_obs.shape[0]}, "
        f"sampled_frames={frame_indices.tolist()}, "
        f"red_pixels={red_counts}, blue_pixels={blue_counts}"
    )


def visualize_image_episodes(
    *,
    npz_dir: Path,
    out_dir: Path,
    episodes: int,
    frames_per_episode: int,
) -> None:
    npz_paths = list_npz_files(npz_dir=npz_dir)
    selected = npz_paths[:episodes]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"visualizing {len(selected)} / {len(npz_paths)} episodes from {npz_dir.resolve()}")
    for npz_path in selected:
        with np.load(npz_path) as data:
            if "img_obs" not in data:
                raise KeyError(f"{npz_path} does not contain an 'img_obs' array")
            img_obs = data["img_obs"]
            obs = data["obs"]
            actions = data["actions"]

        validate_episode_arrays(
            npz_path=npz_path,
            img_obs=img_obs,
            obs=obs,
            actions=actions,
        )
        frame_indices = sample_frame_indices(
            num_frames=img_obs.shape[0],
            frames_per_episode=frames_per_episode,
        )
        summarize_episode(npz_path=npz_path, img_obs=img_obs, frame_indices=frame_indices)
        save_episode_grid(
            npz_path=npz_path,
            img_obs=img_obs,
            frame_indices=frame_indices,
            out_path=out_dir / f"{npz_path.stem}_grid.png",
        )

    print(f"\nsaved grids to: {out_dir.resolve()}")


def default_output_dir(*, npz_dir: Path, data_dir: Path, out_root: Path) -> Path:
    npz_dir = npz_dir.resolve()
    data_dir = data_dir.resolve()

    try:
        relative_path = npz_dir.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError(f"{npz_dir} is not inside data directory {data_dir}") from exc

    return out_root / relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump evenly spaced frame grids from image episode NPZ files."
    )
    parser.add_argument(
        "--npz",
        "--npz-dir",
        dest="npz_dir",
        type=Path,
        default=Path("data/expert/rand-100-img"),
        help="Folder containing episode .npz files with img_obs arrays.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize from the start of the folder.",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=8,
        help="Number of evenly spaced frames to show per episode grid.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory used to compute the relative output folder path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("plots/vis_images"),
        help="Root directory where frame grids will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")

    out_dir = default_output_dir(
        npz_dir=args.npz_dir,
        data_dir=args.data_dir,
        out_root=args.out_root,
    )
    visualize_image_episodes(
        npz_dir=args.npz_dir,
        out_dir=out_dir,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
    )


if __name__ == "__main__":
    main()
