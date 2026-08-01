"""Focused tests for the exhaustive cross-host diagnostic comparator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostic_compare import (  # noqa: E402
    MAX_PIXEL_DIFFS_IN_REPORT,
    compare_diagnostic_runs,
)


def make_episode(*, path: Path, seed_episode: int, n_frames: int = 10) -> None:
    rng = np.random.default_rng(seed_episode)
    obs = rng.normal(size=(n_frames, 45)).astype(np.float32)
    actions = np.empty((n_frames, 8), dtype=np.float32)
    actions[:, :7] = obs[:, :7]
    actions[:, 7] = np.arange(n_frames) % 256
    np.savez_compressed(
        path,
        obs=obs,
        actions=actions,
        img_obs=rng.integers(0, 256, size=(n_frames, 64, 64, 3), dtype=np.uint8),
        qpos=rng.normal(size=(n_frames, 16)),
        qvel=rng.normal(size=(n_frames, 15)),
        ctrl=rng.normal(size=(n_frames, 8)),
        body_xpos=rng.normal(size=(n_frames, 16, 3)),
        body_xmat=rng.normal(size=(n_frames, 16, 9)),
        sim_time=np.arange(n_frames, dtype=np.float64) / 60.0,
        expert_phase=np.full(n_frames, 1, dtype=np.int8),
        cube_init_pos=rng.normal(size=(3,)),
        tray_init_pos=rng.normal(size=(3,)),
    )


def build_dirs(*, root: Path, n_episodes: int, seed: int) -> tuple[Path, Path]:
    dir_a = root / "a"
    dir_b = root / "b"
    for data_dir in (dir_a, dir_b):
        data_dir.mkdir(parents=True)
        (data_dir / "host_info.json").write_text(
            json.dumps(
                {"hostname": data_dir.name, "gpu_uuid": f"GPU-{data_dir.name}"}
            )
        )
    for episode_idx in range(n_episodes):
        make_episode(path=dir_a / f"pick_place_{episode_idx:06d}.npz", seed_episode=seed + episode_idx)
        make_episode(path=dir_b / f"pick_place_{episode_idx:06d}.npz", seed_episode=seed + episode_idx)
    return dir_a, dir_b


def run_cli(*, dir_a: Path, dir_b: Path, report_json: Path, max_episodes: int | None = None) -> tuple[int, dict]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "diagnostic_compare.py"),
        str(dir_a),
        str(dir_b),
        "--report-json",
        str(report_json),
    ]
    if max_episodes is not None:
        command.extend(["--max-episodes", str(max_episodes)])
    result = subprocess.run(command, capture_output=True, text=True)
    report = json.loads(report_json.read_text()) if report_json.exists() else {}
    return result.returncode, report


def test_exact_equality() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=3, seed=5)
        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["exact"] is True
        assert report["comparison"]["episodes_compared"] == 3
        assert report["summary"]["episodes_differing"] == 0
        assert report["summary"]["episodes_exact"] == 3
        assert report["summary"]["fields_differing"] == []
        assert all(field["exact"] for field in report["fields"].values())
        assert report["images"]["differing_pixels"] == 0


def test_image_only_difference() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=3, seed=5)
        b_ep1 = dir_b / "pick_place_000001.npz"
        with np.load(b_ep1) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["img_obs"][3, 10, 20, 2] = (int(arrays["img_obs"][3, 10, 20, 2]) + 1) % 256
        np.savez_compressed(b_ep1, **arrays)

        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["exact"] is False
        assert report["summary"]["fields_differing"] == ["img_obs"]
        assert report["summary"]["differing_episode_indices"] == [1]
        assert report["fields"]["obs"]["exact"] is True
        assert report["fields"]["actions"]["exact"] is True
        assert report["fields"]["qpos"]["exact"] is True
        image_field = report["fields"]["img_obs"]
        assert image_field["differing_frames"] == 1
        assert image_field["first_diff"]["frame"] == 3
        assert image_field["first_diff"]["episode"] == 1
        assert image_field["first_diff"]["index"] == [10, 20, 2]
        assert report["images"]["differing_pixels"] == 1
        assert len(report["images"]["pixel_diffs"]) == 1
        pixel = report["images"]["pixel_diffs"][0]
        assert pixel["episode"] == 1
        assert pixel["frame"] == 3
        assert pixel["h"] == 10
        assert pixel["w"] == 20
        assert pixel["channel"] == 2
        assert abs(pixel["value_a"] - pixel["value_b"]) == 1


def test_late_state_difference_after_image_difference() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=2, seed=5)
        b_ep0 = dir_b / "pick_place_000000.npz"
        with np.load(b_ep0) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["img_obs"][0, 5, 5, 0] = (int(arrays["img_obs"][0, 5, 5, 0]) + 1) % 256
        arrays["qpos"][5, 3] += 1e-9
        np.savez_compressed(b_ep0, **arrays)

        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["exact"] is False
        assert report["summary"]["differing_episode_indices"] == [0]
        assert set(report["summary"]["fields_differing"]) == {"img_obs", "qpos"}
        image_field = report["fields"]["img_obs"]
        qpos_field = report["fields"]["qpos"]
        assert image_field["first_diff"]["frame"] == 0
        assert qpos_field["first_diff"]["frame"] == 5
        assert qpos_field["differing_frames"] == 1
        assert qpos_field["differing_frame_indices"] == [5]
        assert qpos_field["exact"] is False
        assert report["fields"]["obs"]["exact"] is True


def test_obs_and_actions_differences() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=2, seed=5)
        b_ep0 = dir_b / "pick_place_000000.npz"
        with np.load(b_ep0) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["obs"][2, 10] += 1e-3
        arrays["actions"][4, 7] = (int(arrays["actions"][4, 7]) + 1) % 256
        np.savez_compressed(b_ep0, **arrays)

        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["exact"] is False
        assert set(report["summary"]["fields_differing"]) == {"obs", "actions"}
        assert report["fields"]["obs"]["differing_frames"] == 1
        assert report["fields"]["obs"]["differing_frame_indices"] == [2]
        assert report["fields"]["actions"]["differing_frames"] == 1
        assert report["fields"]["actions"]["differing_frame_indices"] == [4]
        assert report["fields"]["qpos"]["exact"] is True
        assert report["fields"]["img_obs"]["exact"] is True


def test_shape_key_dtype_mismatches() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=4, seed=5)

        b_ep1 = dir_b / "pick_place_000001.npz"
        with np.load(b_ep1) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["obs"] = np.asarray(arrays["obs"][:8], dtype=np.float32)
        np.savez_compressed(b_ep1, **arrays)

        b_ep2 = dir_b / "pick_place_000002.npz"
        with np.load(b_ep2) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        del arrays["qvel"]
        np.savez_compressed(b_ep2, **arrays)

        b_ep3 = dir_b / "pick_place_000003.npz"
        with np.load(b_ep3) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["expert_phase"] = arrays["expert_phase"].astype(np.int32)
        np.savez_compressed(b_ep3, **arrays)

        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["exact"] is False
        assert set(report["summary"]["differing_episode_indices"]) == {1, 2, 3}
        assert report["validation"]["episode_length_mismatches"] == []
        assert report["validation"]["shape_mismatches"] == [
            {
                "episode": 1,
                "key": "obs",
                "shape_a": [10, 45],
                "shape_b": [8, 45],
            }
        ]
        assert report["validation"]["internal_length_inconsistencies"] == [
            {
                "episode": 1,
                "key": "obs",
                "length": 8,
                "canonical_length": 10,
            }
        ]
        assert report["validation"]["key_mismatches"] == [
            {
                "episode": 2,
                "missing_in_a": [],
                "missing_in_b": ["qvel"],
                "extra_in_a": [],
                "extra_in_b": [],
            }
        ]
        assert report["validation"]["dtype_mismatches"] == [
            {
                "episode": 3,
                "key": "expert_phase",
                "dtype_a": "int8",
                "dtype_b": "int32",
            }
        ]
        assert report["fields"]["qvel"]["exact"] is False
        assert report["fields"]["qvel"]["missing_in_b"] == [2]
        assert report["fields"]["obs"]["exact"] is False


def test_max_episodes() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=3, seed=5)
        b_ep2 = dir_b / "pick_place_000002.npz"
        with np.load(b_ep2) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["obs"][0, 0] += 1.0
        np.savez_compressed(b_ep2, **arrays)

        limited = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b, max_episodes=2)
        assert limited["comparison"]["episodes_compared"] == 2
        assert limited["comparison"]["episode_count_a"] == 3
        assert limited["comparison"]["max_episodes"] == 2
        assert limited["exact"] is True
        assert limited["summary"]["episodes_differing"] == 0

        full = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert full["exact"] is False
        assert full["summary"]["differing_episode_indices"] == [2]


def test_json_report_semantics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=2, seed=5)
        b_ep1 = dir_b / "pick_place_000001.npz"
        with np.load(b_ep1) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["img_obs"][1, 0, 0, 1] = (int(arrays["img_obs"][1, 0, 0, 1]) + 1) % 256
        np.savez_compressed(b_ep1, **arrays)

        report_json = Path(temp_dir) / "report.json"
        returncode, report = run_cli(
            dir_a=dir_a,
            dir_b=dir_b,
            report_json=report_json,
            max_episodes=1,
        )
        assert returncode == 0
        assert report["schema_version"] == 2
        assert report["tool"] == "scripts/diagnostic_compare.py"
        assert report["comparison"]["max_episodes"] == 1
        assert report["comparison"]["episodes_compared"] == 1
        assert report["hosts"]["a"]["hostname"] == "a"
        assert report["exact"] is True
        assert set(report["fields"]) == {
            "qpos", "qvel", "ctrl", "body_xpos", "body_xmat", "sim_time",
            "obs", "actions", "img_obs", "expert_phase",
            "cube_init_pos", "tray_init_pos",
        }
        assert report["images"]["differing_pixels"] == 0
        assert report["summary"]["fields_exact"]

        report_json2 = Path(temp_dir) / "report2.json"
        returncode2, report2 = run_cli(
            dir_a=dir_a,
            dir_b=dir_b,
            report_json=report_json2,
        )
        assert returncode2 == 1
        assert report2["exact"] is False
        assert report2["summary"]["differing_episode_indices"] == [1]
        assert report2["fields"]["img_obs"]["exact"] is False
        assert report2["fields"]["obs"]["exact"] is True
        assert report2["images"]["differing_pixels"] == 1
        assert report2["images"]["pixel_diffs"][0]["frame"] == 1
        assert report2["summary"]["episodes_exact"] == 1


def test_sorted_pixel_report() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=1, seed=5)
        b_ep0 = dir_b / "pick_place_000000.npz"
        with np.load(b_ep0) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["img_obs"][2, 7, 3, 0] = (int(arrays["img_obs"][2, 7, 3, 0]) + 1) % 256
        arrays["img_obs"][1, 4, 9, 1] = (int(arrays["img_obs"][1, 4, 9, 1]) + 1) % 256
        arrays["img_obs"][2, 7, 3, 0] = (int(arrays["img_obs"][2, 7, 3, 0]) + 1) % 256
        np.savez_compressed(b_ep0, **arrays)

        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["images"]["differing_pixels"] == 2
        assert report["images"]["differing_frames"] == 2
        diffs = report["images"]["pixel_diffs"]
        keys = [(row["frame"], row["h"], row["w"], row["channel"]) for row in diffs]
        assert keys == sorted(keys)
        assert report["images"]["first_diff"]["frame"] == 1
        assert report["images"]["first_diff"]["index"] == [4, 9, 1]


def test_pixel_report_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_a, dir_b = build_dirs(root=Path(temp_dir), n_episodes=2, seed=5)
        for episode_idx in range(2):
            b_path = dir_b / f"pick_place_{episode_idx:06d}.npz"
            with np.load(b_path) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            arrays["img_obs"] ^= np.uint8(255)
            np.savez_compressed(b_path, **arrays)

        report = compare_diagnostic_runs(dir_a=dir_a, dir_b=dir_b)
        assert report["images"]["differing_pixels"] == 2 * 10 * 64 * 64 * 3
        assert len(report["images"]["pixel_diffs"]) == MAX_PIXEL_DIFFS_IN_REPORT
        assert report["images"]["pixel_diffs_truncated"] is True


def main() -> None:
    test_exact_equality()
    test_image_only_difference()
    test_late_state_difference_after_image_difference()
    test_obs_and_actions_differences()
    test_shape_key_dtype_mismatches()
    test_max_episodes()
    test_json_report_semantics()
    test_sorted_pixel_report()
    test_pixel_report_is_bounded()
    print("Diagnostic comparator smoke tests passed.")


if __name__ == "__main__":
    main()
