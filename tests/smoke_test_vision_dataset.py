from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import PickPlaceVisionDataset  # noqa: E402


def save_episode(
    *,
    path: Path,
    obs: np.ndarray,
    actions: np.ndarray,
    img_obs: np.ndarray | None,
) -> None:
    arrays = {"obs": obs, "actions": actions}
    if img_obs is not None:
        arrays["img_obs"] = img_obs
    np.savez_compressed(path, **arrays)


def save_valid_episode(*, path: Path, marker: int) -> None:
    obs = np.full((1, 45), marker, dtype=np.float32)
    actions = np.full((1, 8), marker + 1, dtype=np.float32)
    actions[:, 7] = 255.0
    img_obs = np.full((1, 64, 64, 3), marker, dtype=np.uint8)
    save_episode(path=path, obs=obs, actions=actions, img_obs=img_obs)


def assert_dataset_error(
    *,
    data_dir: Path,
    error_type: type[Exception],
    message: str,
) -> None:
    try:
        PickPlaceVisionDataset(
            data_dirs=data_dir,
            action_space="absolute",
            normalize=False,
        )
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__} containing {message!r}")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_a = root / "source-a"
        source_b = root / "source-b"
        source_a.mkdir()
        source_b.mkdir()

        # Save out of order to verify sorting within each source directory.
        save_valid_episode(path=source_a / "episode_1.npz", marker=20)
        save_valid_episode(path=source_a / "episode_0.npz", marker=10)
        save_valid_episode(path=source_b / "episode_0.npz", marker=30)

        dataset = PickPlaceVisionDataset(
            data_dirs=[source_a, source_b],
            action_space="absolute",
            normalize=False,
        )
        raw_images = dataset.img_obs.copy()

        assert len(dataset) == 3
        for idx, marker in enumerate((10, 20, 30)):
            obs, actions, image = dataset[idx]
            assert obs.shape == (9,)
            assert actions.shape == (8,)
            assert image.shape == (3, 64, 64)
            assert obs.dtype == torch.float32
            assert actions.dtype == torch.float32
            assert image.dtype == torch.float32
            torch.testing.assert_close(obs, torch.full((9,), float(marker)))
            torch.testing.assert_close(
                actions[:7],
                torch.full((7,), float(marker + 1)),
            )
            assert actions[7].item() == 1.0
            torch.testing.assert_close(
                image[:, 0, 0],
                torch.full((3,), marker / 255.0),
            )

        np.testing.assert_array_equal(dataset.img_obs, raw_images)
        assert dataset.img_obs.dtype == np.uint8

        normalized_dataset = PickPlaceVisionDataset(
            data_dirs=source_a,
            action_space="absolute",
            normalize=True,
        )
        _, _, normalized_image = normalized_dataset[0]
        torch.testing.assert_close(
            normalized_image,
            dataset[0][2],
        )

        missing_dir = root / "missing"
        missing_dir.mkdir()
        save_episode(
            path=missing_dir / "episode.npz",
            obs=np.zeros((1, 45), dtype=np.float32),
            actions=np.zeros((1, 8), dtype=np.float32),
            img_obs=None,
        )
        assert_dataset_error(
            data_dir=missing_dir,
            error_type=KeyError,
            message="does not contain an 'img_obs' array",
        )

        wrong_shape_dir = root / "wrong-shape"
        wrong_shape_dir.mkdir()
        save_episode(
            path=wrong_shape_dir / "episode.npz",
            obs=np.zeros((1, 45), dtype=np.float32),
            actions=np.zeros((1, 8), dtype=np.float32),
            img_obs=np.zeros((1, 63, 64, 3), dtype=np.uint8),
        )
        assert_dataset_error(
            data_dir=wrong_shape_dir,
            error_type=ValueError,
            message="img_obs must have shape (1, 64, 64, 3)",
        )

        wrong_dtype_dir = root / "wrong-dtype"
        wrong_dtype_dir.mkdir()
        save_episode(
            path=wrong_dtype_dir / "episode.npz",
            obs=np.zeros((1, 45), dtype=np.float32),
            actions=np.zeros((1, 8), dtype=np.float32),
            img_obs=np.zeros((1, 64, 64, 3), dtype=np.float32),
        )
        assert_dataset_error(
            data_dir=wrong_dtype_dir,
            error_type=ValueError,
            message="img_obs must have dtype uint8",
        )

        misaligned_dir = root / "misaligned"
        misaligned_dir.mkdir()
        save_episode(
            path=misaligned_dir / "episode.npz",
            obs=np.zeros((1, 45), dtype=np.float32),
            actions=np.zeros((1, 8), dtype=np.float32),
            img_obs=np.zeros((2, 64, 64, 3), dtype=np.uint8),
        )
        assert_dataset_error(
            data_dir=misaligned_dir,
            error_type=ValueError,
            message="img_obs must have shape (1, 64, 64, 3)",
        )

    print("Vision dataset smoke test passed.")


if __name__ == "__main__":
    main()
