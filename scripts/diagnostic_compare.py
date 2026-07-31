"""Compare two diagnostic collection directories for cross-host reproducibility.

Loads per-episode NPZ files from two directories and reports the first
float64 state or uint8 image divergence between them.

Usage:
    uv run python scripts/diagnostic_compare.py \
        data/diagnostic/seed0-hostA \
        data/diagnostic/seed0-hostB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_host_info(path: Path) -> dict:
    info_path = path / "host_info.json"
    if info_path.is_file():
        return json.loads(info_path.read_text())
    return {}


def _load_npz_keys(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _array_diff_summary(a: np.ndarray, b: np.ndarray) -> str:
    abs_diff_max = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    abs_diff_count = int(np.count_nonzero(a != b))
    return f"max_abs_diff={abs_diff_max:.6e}, differing_elements={abs_diff_count}/{a.size}"


def _compare_arrays(
    *,
    label: str,
    a: np.ndarray,
    b: np.ndarray,
    episode_idx: int,
    frame_idx: int,
    prefix: str = "",
) -> bool:
    """Return True if arrays match exactly."""
    if a.shape != b.shape:
        print(
            f"{prefix}DIFF: episode {episode_idx:06d} frame {frame_idx}: "
            f"{label} shape mismatch: {a.shape} vs {b.shape}"
        )
        return False
    if not np.array_equal(a, b):
        diff = _array_diff_summary(a, b)
        print(
            f"{prefix}DIFF: episode {episode_idx:06d} frame {frame_idx}: "
            f"{label} differ: {diff}"
        )
        return False
    return True


def compare_diagnostic_runs(
    *,
    dir_a: Path,
    dir_b: Path,
    report_all_diffs: bool = False,
    max_episodes: int | None = None,
) -> int:
    """Compare two diagnostic directories. Returns number of differing episodes."""

    info_a = _load_host_info(dir_a)
    info_b = _load_host_info(dir_b)

    print("=== Host A ===")
    print(f"  hostname:   {info_a.get('hostname', '?')}")
    print(f"  cpu:        {info_a.get('cpu_model', '?')}")
    print(f"  gpu_uuid:   {info_a.get('gpu_uuid', '?')}")
    print(f"  gpu_vbios:  {info_a.get('gpu_vbios', '?')}")
    print()
    print("=== Host B ===")
    print(f"  hostname:   {info_b.get('hostname', '?')}")
    print(f"  cpu:        {info_b.get('cpu_model', '?')}")
    print(f"  gpu_uuid:   {info_b.get('gpu_uuid', '?')}")
    print(f"  gpu_vbios:  {info_b.get('gpu_vbios', '?')}")
    print()

    npz_files_a = sorted(dir_a.glob("pick_place_*.npz"))
    npz_files_b = sorted(dir_b.glob("pick_place_*.npz"))

    if not npz_files_a:
        print(f"ERROR: No NPZ files found in {dir_a}")
        return -1
    if not npz_files_b:
        print(f"ERROR: No NPZ files found in {dir_b}")
        return -1

    print(f"NPZ files: {len(npz_files_a)} (A) vs {len(npz_files_b)} (B)")

    float64_keys = ["qpos", "qvel", "ctrl", "body_xpos", "body_xmat", "sim_time"]
    float32_keys = ["obs", "actions", "cube_init_pos", "tray_init_pos"]
    uint8_keys = ["img_obs"]
    int_keys = ["expert_phase"]

    episodes_with_diff: set[int] = set()
    diff_count = 0
    total_frames = 0

    for a_path in npz_files_a:
        episode_idx = int(a_path.stem.split("_")[-1])
        b_path = dir_b / a_path.name

        if not b_path.is_file():
            print(f"WARNING: episode {episode_idx:06d} missing in B, skipping")
            continue

        try:
            data_a = _load_npz_keys(a_path)
            data_b = _load_npz_keys(b_path)
        except Exception as exc:
            print(f"ERROR loading episode {episode_idx:06d}: {exc}")
            continue

        steps_a = int(data_a.get("obs", np.empty((0,))).shape[0])
        steps_b = int(data_b.get("obs", np.empty((0,))).shape[0])

        if steps_a != steps_b:
            print(
                f"DIFF: episode {episode_idx:06d} step count: {steps_a} (A) vs {steps_b} (B)"
            )
            episodes_with_diff.add(episode_idx)
            diff_count += 1
            if not report_all_diffs:
                print(f"  (stopping comparison of this episode)")
            continue

        total_frames += steps_a
        episode_diff = False

        for frame_idx in range(steps_a):
            prefix = "  " if episode_diff else ""

            for key in float64_keys:
                if key not in data_a or key not in data_b:
                    continue
                a_arr = data_a[key][frame_idx]
                b_arr = data_b[key][frame_idx]
                if not _compare_arrays(
                    label=key,
                    a=a_arr,
                    b=b_arr,
                    episode_idx=episode_idx,
                    frame_idx=frame_idx,
                    prefix=prefix,
                ):
                    episode_diff = True

            for key in uint8_keys:
                if key not in data_a or key not in data_b:
                    continue
                a_arr = data_a[key][frame_idx]
                b_arr = data_b[key][frame_idx]
                if not _compare_arrays(
                    label=key,
                    a=a_arr,
                    b=b_arr,
                    episode_idx=episode_idx,
                    frame_idx=frame_idx,
                    prefix=prefix,
                ):
                    episode_diff = True

            for key in float32_keys:
                if key not in data_a or key not in data_b:
                    continue
                a_arr = data_a[key]
                b_arr = data_b[key]
                if a_arr.ndim <= 1:
                    if not _compare_arrays(
                        label=key,
                        a=a_arr,
                        b=b_arr,
                        episode_idx=episode_idx,
                        frame_idx=frame_idx,
                        prefix=prefix,
                    ):
                        episode_diff = True

            for key in int_keys:
                if key not in data_a or key not in data_b:
                    continue
                a_arr = data_a[key][frame_idx]
                b_arr = data_b[key][frame_idx]
                if not _compare_arrays(
                    label=key,
                    a=a_arr,
                    b=b_arr,
                    episode_idx=episode_idx,
                    frame_idx=frame_idx,
                    prefix=prefix,
                ):
                    episode_diff = True

            if episode_diff and not report_all_diffs:
                break

        if episode_diff:
            episodes_with_diff.add(episode_idx)
            diff_count += 1
        else:
            print(f"  episode {episode_idx:06d}: EXACT ({steps_a} frames)")

    print()
    print("=== Summary ===")
    print(f"Total frames compared: {total_frames}")
    print(f"Episodes with diffs:   {diff_count}/{len(npz_files_a)}")
    if episodes_with_diff:
        sorted_eps = sorted(episodes_with_diff)
        print(f"Differing episodes:    {sorted_eps}")

    return diff_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two diagnostic collection directories."
    )
    parser.add_argument("dir_a", type=str, help="First diagnostic directory")
    parser.add_argument("dir_b", type=str, help="Second diagnostic directory")
    parser.add_argument(
        "--report-all-diffs",
        action="store_true",
        help="Report all frame differences (default: stop at first diff per episode)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only compare up to this many episodes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()

    if not dir_a.is_dir():
        print(f"ERROR: {dir_a} is not a directory")
        sys.exit(1)
    if not dir_b.is_dir():
        print(f"ERROR: {dir_b} is not a directory")
        sys.exit(1)

    n = compare_diagnostic_runs(
        dir_a=dir_a,
        dir_b=dir_b,
        report_all_diffs=args.report_all_diffs,
        max_episodes=args.max_episodes,
    )
    if n > 0:
        print(f"\n{len(dir_a.name)} episodes differ between the two runs.")
        sys.exit(1)
    elif n == 0:
        print("\nAll episodes are EXACT matches between the two runs.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
