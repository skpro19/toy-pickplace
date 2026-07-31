import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
os.environ.setdefault("MUJOCO_GL", "egl")

from scripts.data import collect_expert_episodes  # noqa: E402
from scripts.eval import EVAL_METRIC_VERSION  # noqa: E402
from scripts.reproducibility import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
    build_environment_manifest,
    configure_reproducibility,
    write_environment_manifest,
)
from scripts.train import train  # noqa: E402
from scripts.train_core import engine  # noqa: E402


def assert_semantically_equal(*, left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_semantically_equal(left=left[key], right=right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_semantically_equal(left=left_item, right=right_item)
        return
    assert left == right


def write_training_episode(*, path: Path, include_images: bool) -> None:
    rng = np.random.default_rng(17)
    observations = rng.normal(size=(8, 45)).astype(np.float32)
    actions = np.empty((8, 8), dtype=np.float32)
    actions[:, :7] = observations[:, :7] + rng.normal(
        scale=0.05,
        size=(8, 7),
    ).astype(np.float32)
    actions[:, 7] = np.asarray([0, 255] * 4, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "obs": observations,
        "actions": actions,
    }
    if include_images:
        arrays["img_obs"] = rng.integers(
            0,
            256,
            size=(8, 64, 64, 3),
            dtype=np.uint8,
        )
    np.savez_compressed(path, **arrays)


def fixed_evaluation(**_: object) -> dict[str, object]:
    return {
        "eval_metric_version": EVAL_METRIC_VERSION,
        "mean_score": 0.5,
        "placement_success_rate": 0.25,
        "grasp_rate": 0.75,
        "lift_rate": 0.625,
        "tray_reach_rate": 0.5,
        "lowered_to_tray_rate": 0.375,
        "released_over_tray_rate": 0.25,
    }


def assert_npz_directories_equal(*, left: Path, right: Path) -> None:
    left_files = sorted(left.glob("*.npz"))
    right_files = sorted(right.glob("*.npz"))
    assert [path.name for path in left_files] == [path.name for path in right_files]
    assert left_files
    for left_path, right_path in zip(left_files, right_files, strict=True):
        with np.load(left_path) as left_data, np.load(right_path) as right_data:
            assert left_data.files == right_data.files
            for key in left_data.files:
                assert np.array_equal(left_data[key], right_data[key])


def run_training_trial(
    *,
    root: Path,
    trial_name: str,
    data_dir: Path,
    arch: str,
    seed: int,
) -> dict[str, object]:
    configure_reproducibility(seed=seed)
    checkpoint_dir = root / "checkpoints" / trial_name
    log_dir = root / "runs" / trial_name
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    train(
        arch=arch,
        num_epochs=1,
        batch_size=4,
        npz_folders=[data_dir],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        eval_capture_hz=60.0,
        sample_seed=seed,
        eval_interval=1,
        eval_seed=seed,
        eval_episodes=1,
        eval_max_steps=1,
        dataloader_workers=0,
        early_stop_patience=0,
    )
    return torch.load(
        checkpoint_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )


def main() -> None:
    seed = 420
    configure_reproducibility(seed=seed)
    first_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    configure_reproducibility(seed=seed)
    second_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert first_draws[0] == second_draws[0]
    assert first_draws[1] == second_draws[1]
    assert torch.equal(first_draws[2], second_draws[2])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        manifest_path = root / "environment-manifest.json"
        secret_sentinel = "must-not-appear-in-environment-manifest"
        original_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_sentinel
        try:
            manifest = build_environment_manifest(
                run_name="determinism-smoke",
                arch="vision_mlp",
                project_root=PROJECT_ROOT,
            )
        finally:
            if original_secret is None:
                del os.environ["AWS_SECRET_ACCESS_KEY"]
            else:
                os.environ["AWS_SECRET_ACCESS_KEY"] = original_secret
        write_environment_manifest(path=manifest_path, manifest=manifest)
        saved_manifest = json.loads(manifest_path.read_text())
        assert manifest_path.read_bytes().endswith(b"\n")
        assert saved_manifest["schema_version"] == (
            ENVIRONMENT_MANIFEST_SCHEMA_VERSION
        )
        assert saved_manifest["run_name"] == "determinism-smoke"
        assert saved_manifest["arch"] == "vision_mlp"
        assert saved_manifest["software"]["uv_lock_sha256"] is not None
        assert len(saved_manifest["software"]["uv_lock_sha256"]) == 64
        assert saved_manifest["software"]["packages"]["torch"]
        assert saved_manifest["reproducibility"]["deterministic_algorithms"]
        assert not saved_manifest["reproducibility"]["cuda_matmul_allow_tf32"]
        manifest_text = manifest_path.read_text()
        for secret_name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "VAST_API_KEY",
        ):
            assert secret_name not in manifest_text
        assert secret_sentinel not in manifest_text

        first_expert_dir = root / "expert-first"
        second_expert_dir = root / "expert-second"
        collect_expert_episodes(
            episodes=1,
            out_dir=first_expert_dir,
            seed=seed,
            max_steps=5,
            capture_hz=60.0,
            save_images=True,
        )
        collect_expert_episodes(
            episodes=1,
            out_dir=second_expert_dir,
            seed=seed,
            max_steps=5,
            capture_hz=60.0,
            save_images=True,
        )
        assert_npz_directories_equal(
            left=first_expert_dir,
            right=second_expert_dir,
        )

        data_dir = root / "data"
        data_dir.mkdir()
        arch = "vision_mlp" if torch.cuda.is_available() else "mlp"
        write_training_episode(
            path=data_dir / "episode_000.npz",
            include_images=arch == "vision_mlp",
        )
        original_evaluate_checkpoint = engine.evaluate_checkpoint
        engine.evaluate_checkpoint = fixed_evaluation
        try:
            first_checkpoint = run_training_trial(
                root=root,
                trial_name="first",
                data_dir=data_dir,
                arch=arch,
                seed=seed,
            )
            second_checkpoint = run_training_trial(
                root=root,
                trial_name="second",
                data_dir=data_dir,
                arch=arch,
                seed=seed,
            )
        finally:
            engine.evaluate_checkpoint = original_evaluate_checkpoint
        assert_semantically_equal(left=first_checkpoint, right=second_checkpoint)

    print("Reproducibility smoke test passed.")


if __name__ == "__main__":
    main()
