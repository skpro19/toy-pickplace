"""Exhaustively compare two diagnostic collection directories for cross-host reproducibility.

Loads per-episode NPZ files from two directories and reports every divergence in
every saved field across every frame and every episode. Comparison never stops
after the first difference: a later state divergence is always reported even if
an earlier image frame already differed.

Fields:

- per-frame arrays: ``qpos``, ``qvel``, ``ctrl``, ``body_xpos``, ``body_xmat``,
  ``sim_time``, ``obs``, ``actions``, ``img_obs``, ``expert_phase``;
- per-episode arrays: ``cube_init_pos``, ``tray_init_pos``.

Usage:
    uv run python scripts/diagnostic_compare.py \
        data/diagnostic/seed0-hostA data/diagnostic/seed0-hostB \
        --report-json data/diagnostic/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PER_FRAME_KEYS = (
    "qpos",
    "qvel",
    "ctrl",
    "body_xpos",
    "body_xmat",
    "sim_time",
    "obs",
    "actions",
    "img_obs",
    "expert_phase",
)
PER_EPISODE_KEYS = (
    "cube_init_pos",
    "tray_init_pos",
)
EXPECTED_KEYS = frozenset(PER_FRAME_KEYS + PER_EPISODE_KEYS)

SCHEMA_VERSION = 2
MAX_PIXEL_DIFFS_IN_REPORT = 50


def _load_host_info(path: Path) -> dict:
    info_path = path / "host_info.json"
    if info_path.is_file():
        return json.loads(info_path.read_text())
    return {}


def _to_python(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.dtype.kind in ("i", "u", "b"):
        return float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _first_diff_details(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Return the first differing element as {index, value_a, value_b}."""
    coords = np.argwhere(a != b)
    if coords.shape[0] == 0:
        return None
    first = tuple(int(i) for i in coords[0])
    return {
        "index": list(first),
        "value_a": _to_python(a[first]),
        "value_b": _to_python(b[first]),
    }


def _empty_field_report(*, kind: str) -> dict:
    return {
        "kind": kind,
        "present_a": True,
        "present_b": True,
        "dtype_a": None,
        "dtype_b": None,
        "shape_a": None,
        "shape_b": None,
        "exact": True,
        "differing_episodes": 0,
        "differing_episode_indices": [],
        "differing_frames": 0,
        "differing_frame_indices": [],
        "differing_elements": 0,
        "total_elements": 0,
        "max_abs_diff": 0.0,
        "first_diff": None,
        "frames_in_a": 0,
        "frames_in_b": 0,
    }


def _aggregate_per_frame_field(
    *,
    key: str,
    reports: list[dict],
    compared_episodes: int,
    total_frames: int,
) -> dict:
    """Merge per-episode frame-wise results into one deterministic field report."""
    differing_episodes = [r["episode"] for r in reports if not r["exact"]]
    differing_frames = sorted(
        frame
        for r in reports
        for frame in r["differing_frames"]
    )
    first = next(
        (r for r in reports if r["first_diff"] is not None),
        None,
    )
    return {
        "kind": "per_frame",
        "present_a": True,
        "present_b": True,
        "dtype_a": reports[0]["dtype_a"] if reports else None,
        "dtype_b": reports[0]["dtype_b"] if reports else None,
        "shape_a": reports[0]["shape_a"] if reports else None,
        "shape_b": reports[0]["shape_b"] if reports else None,
        "exact": not differing_episodes,
        "differing_episodes": len(differing_episodes),
        "differing_episode_indices": differing_episodes,
        "differing_frames": len(differing_frames),
        "differing_frame_indices": differing_frames,
        "differing_elements": sum(r["differing_elements"] for r in reports),
        "total_elements": sum(r["total_elements"] for r in reports),
        "max_abs_diff": max((r["max_abs_diff"] for r in reports), default=0.0),
        "first_diff": first["first_diff"] if first else None,
        "frames_in_a": sum(r["frames_in_a"] for r in reports),
        "frames_in_b": sum(r["frames_in_b"] for r in reports),
        "any_shape_mismatch": any(r.get("shape_mismatch", False) for r in reports),
        "any_dtype_mismatch": any(r.get("dtype_mismatch", False) for r in reports),
        "compared_episodes": compared_episodes,
        "total_frames": total_frames,
    }


def _aggregate_per_episode_field(
    *,
    key: str,
    reports: list[dict],
    compared_episodes: int,
) -> dict:
    differing_episodes = [r["episode"] for r in reports if not r["exact"]]
    first = next(
        (r for r in reports if r["first_diff"] is not None),
        None,
    )
    return {
        "kind": "per_episode",
        "present_a": True,
        "present_b": True,
        "dtype_a": reports[0]["dtype_a"] if reports else None,
        "dtype_b": reports[0]["dtype_b"] if reports else None,
        "shape_a": reports[0]["shape_a"] if reports else None,
        "shape_b": reports[0]["shape_b"] if reports else None,
        "exact": not differing_episodes,
        "differing_episodes": len(differing_episodes),
        "differing_episode_indices": differing_episodes,
        "differing_frames": 0,
        "differing_frame_indices": [],
        "differing_elements": sum(r["differing_elements"] for r in reports),
        "total_elements": sum(r["total_elements"] for r in reports),
        "max_abs_diff": max((r["max_abs_diff"] for r in reports), default=0.0),
        "first_diff": first["first_diff"] if first else None,
        "any_shape_mismatch": any(r.get("shape_mismatch", False) for r in reports),
        "any_dtype_mismatch": any(r.get("dtype_mismatch", False) for r in reports),
        "compared_episodes": compared_episodes,
    }


def _pixel_diff_rows(
    *,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    episode_idx: int,
    frame_idx: int,
    limit: int,
) -> list[dict]:
    coords = np.argwhere(arr_a != arr_b)[:limit]
    rows = []
    for coord in coords:
        h, w, channel = (int(c) for c in coord)
        rows.append(
            {
                "episode": episode_idx,
                "frame": frame_idx,
                "h": h,
                "w": w,
                "channel": channel,
                "value_a": int(arr_a[h, w, channel]),
                "value_b": int(arr_b[h, w, channel]),
            }
        )
    return rows


def _compare_episode_arrays(
    *,
    data_a: dict,
    data_b: dict,
    episode_idx: int,
    validation: dict,
) -> dict:
    """Compare one episode's arrays. Returns per-field results for this episode."""
    per_frame_reports: dict[str, dict] = {}
    per_episode_reports: dict[str, dict] = {}

    key_a = set(data_a)
    key_b = set(data_b)
    if key_a != key_b:
        validation["key_mismatches"].append(
            {
                "episode": episode_idx,
                "missing_in_a": sorted(key_b - key_a),
                "missing_in_b": sorted(key_a - key_b),
                "extra_in_a": sorted(key_a - key_b - EXPECTED_KEYS),
                "extra_in_b": sorted(key_b - key_a - EXPECTED_KEYS),
            }
        )

    for key in sorted(EXPECTED_KEYS & key_a & key_b):
        arr_a = data_a[key]
        arr_b = data_b[key]
        dtype_mismatch = arr_a.dtype != arr_b.dtype
        shape_mismatch = arr_a.shape != arr_b.shape
        if dtype_mismatch:
            validation["dtype_mismatches"].append(
                {
                    "episode": episode_idx,
                    "key": key,
                    "dtype_a": str(arr_a.dtype),
                    "dtype_b": str(arr_b.dtype),
                }
            )
        if shape_mismatch:
            validation["shape_mismatches"].append(
                {
                    "episode": episode_idx,
                    "key": key,
                    "shape_a": list(arr_a.shape),
                    "shape_b": list(arr_b.shape),
                }
            )
        if key in PER_FRAME_KEYS:
            report = _compare_per_frame(
                key=key,
                arr_a=arr_a,
                arr_b=arr_b,
                episode_idx=episode_idx,
                validation=validation,
            )
        else:
            report = _compare_per_episode(
                key=key,
                arr_a=arr_a,
                arr_b=arr_b,
                episode_idx=episode_idx,
                validation=validation,
            )
        if dtype_mismatch or shape_mismatch:
            report["exact"] = False
            report["dtype_mismatch"] = dtype_mismatch
            report["shape_mismatch"] = shape_mismatch
        if key in PER_FRAME_KEYS:
            per_frame_reports[key] = report
        else:
            per_episode_reports[key] = report
    return {
        "per_frame": per_frame_reports,
        "per_episode": per_episode_reports,
    }


def _compare_per_frame(
    *,
    key: str,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    episode_idx: int,
    validation: dict,
) -> dict:
    report: dict = {
        "episode": episode_idx,
        "exact": True,
        "differing_frames": [],
        "differing_elements": 0,
        "total_elements": int(arr_a.size),
        "max_abs_diff": 0.0,
        "first_diff": None,
        "dtype_a": str(arr_a.dtype),
        "dtype_b": str(arr_b.dtype),
        "shape_a": list(arr_a.shape),
        "shape_b": list(arr_b.shape),
        "frames_in_a": int(arr_a.shape[0]),
        "frames_in_b": int(arr_b.shape[0]),
    }
    frames_a = int(arr_a.shape[0])
    frames_b = int(arr_b.shape[0])
    shared_frames = min(frames_a, frames_b)

    if key == "img_obs":
        report["pixel_diffs"] = []

    for frame in range(shared_frames):
        frame_a = arr_a[frame]
        frame_b = arr_b[frame]
        if np.array_equal(frame_a, frame_b):
            continue
        report["exact"] = False
        report["differing_frames"].append(frame)
        report["differing_elements"] += int(np.count_nonzero(frame_a != frame_b))
        report["max_abs_diff"] = max(
            report["max_abs_diff"],
            _max_abs_diff(frame_a, frame_b),
        )
        if report["first_diff"] is None:
            details = _first_diff_details(frame_a, frame_b)
            details["frame"] = frame
            details["episode"] = episode_idx
            report["first_diff"] = details
        if key == "img_obs":
            remaining_pixel_rows = (
                MAX_PIXEL_DIFFS_IN_REPORT - len(report["pixel_diffs"])
            )
            if remaining_pixel_rows > 0:
                report["pixel_diffs"].extend(
                    _pixel_diff_rows(
                        arr_a=frame_a,
                        arr_b=frame_b,
                        episode_idx=episode_idx,
                        frame_idx=frame,
                        limit=remaining_pixel_rows,
                    )
                )
    return report


def _compare_per_episode(
    *,
    key: str,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    episode_idx: int,
    validation: dict,
) -> dict:
    report: dict = {
        "episode": episode_idx,
        "exact": np.array_equal(arr_a, arr_b),
        "differing_elements": int(np.count_nonzero(arr_a != arr_b)),
        "total_elements": int(arr_a.size),
        "max_abs_diff": _max_abs_diff(arr_a, arr_b) if not np.array_equal(arr_a, arr_b) else 0.0,
        "first_diff": (
            _first_diff_details(arr_a, arr_b)
            if not np.array_equal(arr_a, arr_b)
            else None
        ),
        "dtype_a": str(arr_a.dtype),
        "dtype_b": str(arr_b.dtype),
        "shape_a": list(arr_a.shape),
        "shape_b": list(arr_b.shape),
    }
    return report


def _validate_episode_lengths(
    *,
    data: dict,
    episode_idx: int,
    validation: dict,
) -> int:
    """Return the canonical episode frame count, validating internal consistency."""
    frame_lengths = {
        key: int(data[key].shape[0])
        for key in PER_FRAME_KEYS
        if key in data and data[key].ndim > 0
    }
    if not frame_lengths:
        return 0
    canonical = max(frame_lengths.values())
    for key, length in sorted(frame_lengths.items()):
        if length != canonical:
            validation["internal_length_inconsistencies"].append(
                {
                    "episode": episode_idx,
                    "key": key,
                    "length": length,
                    "canonical_length": canonical,
                }
            )
    return canonical


def compare_diagnostic_runs(
    *,
    dir_a: Path,
    dir_b: Path,
    max_episodes: int | None = None,
) -> dict:
    """Exhaustively compare two diagnostic directories.

    Returns a deterministic, machine-readable report dict. Every field of every
    frame of every episode is compared; comparison continues past image
    differences so later state divergence is always captured.
    """
    info_a = _load_host_info(dir_a)
    info_b = _load_host_info(dir_b)

    names_a = sorted(path.name for path in dir_a.glob("pick_place_*.npz"))
    names_b = sorted(path.name for path in dir_b.glob("pick_place_*.npz"))
    full_count_a = len(names_a)
    full_count_b = len(names_b)

    if max_episodes is not None:
        names_a = names_a[:max_episodes]
        names_b = names_b[:max_episodes]

    missing_in_a = sorted(set(names_b) - set(names_a))
    missing_in_b = sorted(set(names_a) - set(names_b))
    compared_names = sorted(set(names_a) & set(names_b))
    compared_episodes = len(compared_names)

    validation: dict = {
        "missing_episodes_in_a": missing_in_a,
        "missing_episodes_in_b": missing_in_b,
        "episode_length_mismatches": [],
        "key_mismatches": [],
        "dtype_mismatches": [],
        "shape_mismatches": [],
        "internal_length_inconsistencies": [],
        "unexpected_keys": [],
    }

    fields = {key: _empty_field_report(kind="per_frame") for key in PER_FRAME_KEYS}
    fields.update({key: _empty_field_report(kind="per_episode") for key in PER_EPISODE_KEYS})
    for key in fields:
        fields[key]["compared_episodes"] = 0
        fields[key]["total_frames"] = 0

    per_frame_storage: dict[str, list[dict]] = {key: [] for key in PER_FRAME_KEYS}
    per_episode_storage: dict[str, list[dict]] = {key: [] for key in PER_EPISODE_KEYS}
    missing_key_presence: dict[str, dict[str, list[int]]] = {
        key: {"missing_in_a": [], "missing_in_b": []} for key in EXPECTED_KEYS
    }
    episodes_differing: list[int] = []
    episodes_exact_indices: list[int] = []
    total_frames_compared = 0

    episode_length_map_a: dict[str, int] = {}
    episode_length_map_b: dict[str, int] = {}

    for name in compared_names:
        a_path = dir_a / name
        b_path = dir_b / name
        episode_idx = int(a_path.stem.split("_")[-1])

        with np.load(a_path, allow_pickle=False) as loaded_a, np.load(
            b_path, allow_pickle=False
        ) as loaded_b:
            data_a = {key: loaded_a[key] for key in loaded_a.files}
            data_b = {key: loaded_b[key] for key in loaded_b.files}

        unexpected_a = sorted(set(data_a) - EXPECTED_KEYS)
        unexpected_b = sorted(set(data_b) - EXPECTED_KEYS)
        for key in sorted(set(unexpected_a) | set(unexpected_b)):
            validation["unexpected_keys"].append(
                {
                    "episode": episode_idx,
                    "key": key,
                    "in_a": key in data_a,
                    "in_b": key in data_b,
                }
            )

        length_a = _validate_episode_lengths(
            data=data_a,
            episode_idx=episode_idx,
            validation=validation,
        )
        length_b = _validate_episode_lengths(
            data=data_b,
            episode_idx=episode_idx,
            validation=validation,
        )
        episode_length_map_a[name] = length_a
        episode_length_map_b[name] = length_b
        if length_a != length_b:
            validation["episode_length_mismatches"].append(
                {
                    "episode": episode_idx,
                    "len_a": length_a,
                    "len_b": length_b,
                }
            )

        total_frames_compared += min(length_a, length_b)

        for key in EXPECTED_KEYS:
            if key not in data_a:
                missing_key_presence[key]["missing_in_a"].append(episode_idx)
            if key not in data_b:
                missing_key_presence[key]["missing_in_b"].append(episode_idx)

        results = _compare_episode_arrays(
            data_a=data_a,
            data_b=data_b,
            episode_idx=episode_idx,
            validation=validation,
        )

        episode_exact = True
        for key, report in results["per_frame"].items():
            per_frame_storage[key].append(report)
            if not report["exact"]:
                episode_exact = False
        for key, report in results["per_episode"].items():
            per_episode_storage[key].append(report)
            if not report["exact"]:
                episode_exact = False

        validation_touches_episode = any(
            item["episode"] == episode_idx
            for item in (
                validation["key_mismatches"]
                + validation["dtype_mismatches"]
                + validation["shape_mismatches"]
            )
        )
        if validation_touches_episode:
            episode_exact = False

        has_length_mismatch = length_a != length_b
        if not episode_exact or has_length_mismatch:
            episodes_differing.append(episode_idx)
        else:
            episodes_exact_indices.append(episode_idx)

    missing_episode_names = sorted(set(missing_in_a) | set(missing_in_b))
    for name in missing_episode_names:
        episodes_differing.append(int(Path(name).stem.split("_")[-1]))

    for key in PER_FRAME_KEYS:
        if per_frame_storage[key]:
            fields[key] = _aggregate_per_frame_field(
                key=key,
                reports=per_frame_storage[key],
                compared_episodes=compared_episodes,
                total_frames=total_frames_compared,
            )
    for key in PER_EPISODE_KEYS:
        if per_episode_storage[key]:
            fields[key] = _aggregate_per_episode_field(
                key=key,
                reports=per_episode_storage[key],
                compared_episodes=compared_episodes,
            )

    for key, presence in missing_key_presence.items():
        missing_in_a = presence["missing_in_a"]
        missing_in_b = presence["missing_in_b"]
        if not missing_in_a and not missing_in_b:
            continue
        fields[key]["exact"] = False
        fields[key]["missing_in_a"] = missing_in_a
        fields[key]["missing_in_b"] = missing_in_b
        fields[key]["differing_episodes"] += len(set(missing_in_a) | set(missing_in_b))
        fields[key]["differing_episode_indices"] = sorted(
            set(fields[key]["differing_episode_indices"])
            | set(missing_in_a)
            | set(missing_in_b)
        )

    image_field = fields["img_obs"]
    pixel_rows = sorted(
        (
            row
            for report in per_frame_storage["img_obs"]
            for row in report.get("pixel_diffs", [])
        ),
        key=lambda row: (row["episode"], row["frame"], row["h"], row["w"], row["channel"]),
    )
    consecutive_differing_frames = False
    differing_frame_sets: dict[int, list[int]] = {}
    for report in per_frame_storage["img_obs"]:
        frames = report["differing_frames"]
        differing_frame_sets[report["episode"]] = frames
        if any(right - left == 1 for left, right in zip(frames, frames[1:])):
            consecutive_differing_frames = True

    images = {
        "differing_episodes": image_field["differing_episodes"],
        "differing_episode_indices": image_field["differing_episode_indices"],
        "differing_frames": image_field["differing_frames"],
        "differing_frame_indices": image_field["differing_frame_indices"],
        "total_frames_compared": image_field["total_frames"],
        "differing_pixels": image_field["differing_elements"],
        "max_abs_diff": image_field["max_abs_diff"],
        "first_diff": image_field["first_diff"],
        "pixel_diffs": pixel_rows[:MAX_PIXEL_DIFFS_IN_REPORT],
        "pixel_diffs_truncated": len(pixel_rows) > MAX_PIXEL_DIFFS_IN_REPORT,
        "consecutive_differing_frames_present": consecutive_differing_frames,
        "differing_frames_by_episode": {
            str(episode): frames
            for episode, frames in sorted(differing_frame_sets.items())
            if frames
        },
    }

    fields_exact = [key for key, report in fields.items() if report["exact"]]
    fields_differing = [key for key, report in fields.items() if not report["exact"]]
    exact = (
        not missing_in_a
        and not missing_in_b
        and not validation["episode_length_mismatches"]
        and not validation["key_mismatches"]
        and not validation["dtype_mismatches"]
        and not validation["shape_mismatches"]
        and not episodes_differing
    )

    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/diagnostic_compare.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hosts": {"a": info_a, "b": info_b},
        "directories": {"a": str(dir_a), "b": str(dir_b)},
        "comparison": {
            "max_episodes": max_episodes,
            "episode_count_a": full_count_a,
            "episode_count_b": full_count_b,
            "episodes_compared": compared_episodes,
            "total_frames_compared": total_frames_compared,
        },
        "validation": validation,
        "fields": fields,
        "images": images,
        "exact": exact,
        "summary": {
            "episodes_exact": len(episodes_exact_indices),
            "episodes_differing": len(set(episodes_differing)),
            "differing_episode_indices": sorted(set(episodes_differing)),
            "episodes_compared": compared_episodes,
            "fields_exact": fields_exact,
            "fields_differing": fields_differing,
            "exact": exact,
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively compare two diagnostic collection directories."
    )
    parser.add_argument("dir_a", type=str, help="First diagnostic directory")
    parser.add_argument("dir_b", type=str, help="Second diagnostic directory")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only compare up to this many episodes (sorted filename order)",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Write the machine-readable JSON report to this path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode status lines",
    )
    return parser.parse_args()


def _print_report(report: dict, *, verbose: bool) -> None:
    hosts = report["hosts"]
    for label in ("a", "b"):
        info = hosts[label]
        print(f"=== Host {label.upper()} ===")
        print(f"  hostname:  {info.get('hostname', '?')}")
        print(f"  cpu:       {info.get('cpu_model', '?')}")
        print(f"  gpu_uuid:  {info.get('gpu_uuid', '?')}")
        print(f"  gpu_vbios: {info.get('gpu_vbios', '?')}")
        print()
    comparison = report["comparison"]
    print(
        "Episodes: "
        f"{comparison['episode_count_a']} (A) vs {comparison['episode_count_b']} (B); "
        f"compared {comparison['episodes_compared']}; "
        f"frames compared {comparison['total_frames_compared']}"
    )
    if comparison["max_episodes"] is not None:
        print(f"  (limited by --max-episodes {comparison['max_episodes']})")

    validation = report["validation"]
    issues = [
        f"missing in A: {len(validation['missing_episodes_in_a'])}",
        f"missing in B: {len(validation['missing_episodes_in_b'])}",
        f"episode-length mismatches: {len(validation['episode_length_mismatches'])}",
        f"key mismatches: {len(validation['key_mismatches'])}",
        f"dtype mismatches: {len(validation['dtype_mismatches'])}",
        f"shape mismatches: {len(validation['shape_mismatches'])}",
    ]
    active_issues = [issue for issue in issues if not issue.endswith(": 0")]
    if active_issues:
        print("Validation issues: " + "; ".join(active_issues))
    else:
        print("Validation: OK (files, keys, dtypes, shapes, lengths all match)")

    print()
    print("Fields:")
    for key in sorted(report["fields"]):
        field = report["fields"][key]
        if field["exact"]:
            print(f"  {key:12s} exact")
        else:
            print(
                f"  {key:12s} DIFF episodes={field['differing_episodes']} "
                f"frames={field['differing_frames']} "
                f"elements={field['differing_elements']} "
                f"maxdiff={field['max_abs_diff']:.6g}"
            )

    summary = report["summary"]
    print()
    print(
        f"Summary: {summary['episodes_differing']}/{summary['episodes_compared']} "
        "episodes differ"
    )
    if summary["fields_differing"]:
        print(f"  differing fields: {', '.join(summary['fields_differing'])}")
    else:
        print("  all fields exact")
    print(f"Overall: {'EXACT' if report['exact'] else 'NOT EXACT'}")

    if verbose:
        print()
        for episode in report["summary"]["differing_episode_indices"]:
            print(f"  episode {episode:06d}: DIFF")


def main() -> None:
    args = parse_args()
    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()

    if not dir_a.is_dir():
        print(f"ERROR: {dir_a} is not a directory", file=sys.stderr)
        sys.exit(2)
    if not dir_b.is_dir():
        print(f"ERROR: {dir_b} is not a directory", file=sys.stderr)
        sys.exit(2)
    if args.max_episodes is not None and args.max_episodes < 1:
        print("ERROR: --max-episodes must be a positive integer", file=sys.stderr)
        sys.exit(2)

    report = compare_diagnostic_runs(
        dir_a=dir_a,
        dir_b=dir_b,
        max_episodes=args.max_episodes,
    )
    _print_report(report, verbose=args.verbose)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_name(f"{report_path.name}.tmp")
        temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(report_path)
        print(f"JSON report written to {report_path}")

    if report["exact"]:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
