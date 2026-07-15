from pathlib import Path
import sys
import tempfile

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import PickPlaceDataset  # noqa: E402
from train import prepare_dataset  # noqa: E402


def save_episode(
    *,
    path: Path,
    length: int,
    value: float,
    execute_expert: list[bool] | None = None,
) -> None:
    arrays = {
        "obs": np.full((length, 45), value, dtype=np.float32),
        "actions": np.full((length, 8), value, dtype=np.float32),
    }
    if execute_expert is not None:
        arrays["execute_expert"] = np.asarray(execute_expert, dtype=np.bool_)
    np.savez_compressed(path, **arrays)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        expert_dir = root / "expert"
        dagger_dir = root / "dagger"
        expert_dir.mkdir()
        dagger_dir.mkdir()

        save_episode(path=expert_dir / "episode_0.npz", length=2, value=0.0)
        save_episode(path=expert_dir / "episode_1.npz", length=8, value=10.0)
        save_episode(
            path=dagger_dir / "episode_0.npz",
            length=4,
            value=20.0,
            execute_expert=[True, True, True, False],
        )

        dataset = PickPlaceDataset(
            data_dirs=[expert_dir, dagger_dir],
            sample_ratios=[0.6, 0.4],
            dagger_intervention_ratio=0.8,
        )
        assert np.isclose(dataset.sample_weights[:2].sum(), 0.3)
        assert np.isclose(dataset.sample_weights[2:10].sum(), 0.3)
        assert np.isclose(dataset.sample_weights[10:].sum(), 0.4)
        assert np.isclose(dataset.sample_weights[10:13].sum(), 0.32)
        assert np.isclose(dataset.sample_weights[13:].sum(), 0.08)
        assert np.isclose(dataset.sample_weights.sum(), 1.0)
        assert dataset.samples_per_epoch == 10
        assert dataset.source_frame_counts.tolist() == [10, 4]
        assert np.allclose(dataset.source_ratios, [0.6, 0.4])

        _, norm_stats = prepare_dataset(
            npz_folders=[expert_dir, dagger_dir],
            sample_ratios=[0.6, 0.4],
            dagger_intervention_ratio=0.8,
            action_space="absolute",
            normalize=True,
        )
        expected_mean = 11.0
        assert np.allclose(norm_stats["arm_obs_mean"], expected_mean)
        assert np.allclose(norm_stats["arm_actions_mean"], expected_mean)

    print("Episode sampling smoke test passed.")


if __name__ == "__main__":
    main()
