from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.dataset import PickPlaceDataset  # noqa: E402
from train_core.recipes.mlp import prepare_dataset  # noqa: E402


EPSILON = 1e-6


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


def build_eager_targets(
    *,
    dataset: PickPlaceDataset,
    action_space: str,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    obs_targets = dataset.obs.copy()
    action_targets = np.empty_like(dataset.actions)
    if action_space == "joint_delta":
        action_targets[:, :7] = dataset.actions[:, :7] - dataset.obs[:, :7]
    else:
        action_targets[:, :7] = dataset.actions[:, :7]
    action_targets[:, 7] = dataset.actions[:, 7] / 255.0

    arm_actions_mean = np.average(
        action_targets[:, :7],
        axis=0,
        weights=dataset.sample_weights,
    )[None, :].astype(np.float32)
    arm_obs_mean = np.average(
        obs_targets,
        axis=0,
        weights=dataset.sample_weights,
    )[None, :].astype(np.float32)
    expected_stats = {
        "arm_actions_mean": arm_actions_mean,
        "arm_actions_std": np.sqrt(
            np.average(
                np.square(action_targets[:, :7] - arm_actions_mean),
                axis=0,
                weights=dataset.sample_weights,
            )
        )[None, :].astype(np.float32),
        "arm_obs_mean": arm_obs_mean,
        "arm_obs_std": np.sqrt(
            np.average(
                np.square(obs_targets - arm_obs_mean),
                axis=0,
                weights=dataset.sample_weights,
            )
        )[None, :].astype(np.float32),
    }
    for name, expected in expected_stats.items():
        actual = dataset.norm_stats[name]
        assert actual.dtype == np.float32
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected)

    if normalize:
        action_targets[:, :7] -= expected_stats["arm_actions_mean"]
        action_targets[:, :7] /= expected_stats["arm_actions_std"] + EPSILON
        obs_targets -= expected_stats["arm_obs_mean"]
        obs_targets /= expected_stats["arm_obs_std"] + EPSILON

    return obs_targets, action_targets


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

        target_dir = root / "targets"
        target_dir.mkdir()
        target_obs = np.arange(3 * 45, dtype=np.float32).reshape(3, 45) / 10.0
        target_actions = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 255.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 127.5],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            target_dir / "episode_0.npz",
            obs=target_obs,
            actions=target_actions,
        )

        sampling_dataset = PickPlaceDataset(
            data_dirs=[expert_dir, dagger_dir],
            sample_ratios=[0.6, 0.4],
            dagger_intervention_ratio=0.8,
            action_space="absolute",
            normalize=False,
        )
        assert np.isclose(sampling_dataset.sample_weights[:2].sum(), 0.3)
        assert np.isclose(sampling_dataset.sample_weights[2:10].sum(), 0.3)
        assert np.isclose(sampling_dataset.sample_weights[10:].sum(), 0.4)
        assert np.isclose(sampling_dataset.sample_weights[10:13].sum(), 0.32)
        assert np.isclose(sampling_dataset.sample_weights[13:].sum(), 0.08)
        assert np.isclose(sampling_dataset.sample_weights.sum(), 1.0)
        assert sampling_dataset.samples_per_epoch == 10
        assert sampling_dataset.source_frame_counts.tolist() == [10, 4]
        assert np.allclose(sampling_dataset.source_ratios, [0.6, 0.4])

        for action_space in ("absolute", "joint_delta"):
            for normalize in (False, True):
                dataset = PickPlaceDataset(
                    data_dirs=target_dir,
                    action_space=action_space,
                    normalize=normalize,
                )
                raw_obs = dataset.obs.copy()
                raw_actions = dataset.actions.copy()
                expected_obs, expected_actions = build_eager_targets(
                    dataset=dataset,
                    action_space=action_space,
                    normalize=normalize,
                )
                for idx in range(len(dataset)):
                    obs, action = dataset[idx]
                    assert obs.dtype == torch.float32
                    assert action.dtype == torch.float32
                    torch.testing.assert_close(obs, torch.from_numpy(expected_obs[idx]))
                    torch.testing.assert_close(
                        action,
                        torch.from_numpy(expected_actions[idx]),
                    )

                first_sample = dataset[0]
                repeated_sample = dataset[0]
                torch.testing.assert_close(first_sample[0], repeated_sample[0])
                torch.testing.assert_close(first_sample[1], repeated_sample[1])
                np.testing.assert_array_equal(dataset.obs, raw_obs)
                np.testing.assert_array_equal(dataset.actions, raw_actions)
                assert not hasattr(dataset, "obs_targets")
                assert not hasattr(dataset, "action_targets")

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

        try:
            PickPlaceDataset(
                data_dirs=root / "does-not-exist",
                action_space="invalid",
                normalize=False,
            )
        except ValueError as error:
            assert str(error) == "Unknown action_space: invalid"
        else:
            raise AssertionError("invalid action_space was not rejected")

    print("Episode sampling smoke test passed.")


if __name__ == "__main__":
    main()
