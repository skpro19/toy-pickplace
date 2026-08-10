from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.constant import ACTION_DIMS, IMAGE_HEIGHT, IMAGE_WIDTH, PROPRIO_DIMS  # noqa: E402
from scripts.dataset import PickPlaceACTDataset  # noqa: E402
from train_core.dataloader import build_weighted_dataloader  # noqa: E402

CHUNK_SIZE = 3
BATCH_SIZE = 4


def save_episode(
    *,
    path: Path,
    frame_joint_values: list[float],
    marker: int,
) -> None:
    timestep_count = len(frame_joint_values)
    obs = np.full((timestep_count, 45), float(marker), dtype=np.float32)
    actions = np.empty((timestep_count, ACTION_DIMS), dtype=np.float32)
    for step, joint_value in enumerate(frame_joint_values):
        actions[step, : ACTION_DIMS - 1] = joint_value
        actions[step, ACTION_DIMS - 1] = 255.0
    img_obs = np.full(
        (timestep_count, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        marker,
        dtype=np.uint8,
    )
    np.savez_compressed(
        path,
        obs=obs,
        actions=actions,
        img_obs=img_obs,
    )


def assert_joint_values(*, action: torch.Tensor, expected: float) -> None:
    torch.testing.assert_close(
        action[: ACTION_DIMS - 1],
        torch.full((ACTION_DIMS - 1,), expected),
    )
    assert action[ACTION_DIMS - 1].item() == 1.0


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        save_episode(
            path=data_dir / "episode_0.npz",
            frame_joint_values=[100.0, 101.0, 102.0, 103.0],
            marker=10,
        )
        save_episode(
            path=data_dir / "episode_1.npz",
            frame_joint_values=[200.0, 201.0],
            marker=20,
        )

        dataset = PickPlaceACTDataset(
            data_dirs=data_dir,
            chunk_size=CHUNK_SIZE,
            action_space="absolute",
            normalize=False,
        )

        assert len(dataset) == 6
        np.testing.assert_array_equal(dataset.episode_ends, np.array([4, 6], dtype=np.int64))

        obs, actions, image, is_pad = dataset[1]
        assert obs.shape == (PROPRIO_DIMS,)
        assert actions.shape == (CHUNK_SIZE, ACTION_DIMS)
        assert image.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert is_pad.shape == (CHUNK_SIZE,)
        assert is_pad.dtype == torch.bool
        assert not is_pad.any()
        assert_joint_values(action=actions[0], expected=101.0)
        assert_joint_values(action=actions[1], expected=102.0)
        assert_joint_values(action=actions[2], expected=103.0)

        _, tail_actions, _, tail_is_pad = dataset[2]
        assert tail_is_pad.tolist() == [False, False, True]
        assert_joint_values(action=tail_actions[0], expected=102.0)
        assert_joint_values(action=tail_actions[1], expected=103.0)
        assert_joint_values(action=tail_actions[2], expected=103.0)
        assert not torch.allclose(
            tail_actions[2, : ACTION_DIMS - 1],
            torch.full((ACTION_DIMS - 1,), 200.0),
        )

        dataloader = build_weighted_dataloader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            sample_seed=0,
            dataloader_workers=0,
            persistent_workers=False,
            device=torch.device("cpu"),
        )
        batch_obs, batch_actions, batch_image, batch_is_pad = next(iter(dataloader))
        assert batch_obs.shape == (BATCH_SIZE, PROPRIO_DIMS)
        assert batch_actions.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIMS)
        assert batch_image.shape == (BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert batch_is_pad.shape == (BATCH_SIZE, CHUNK_SIZE)
        assert batch_is_pad.dtype == torch.bool

    print("ACT dataset smoke test passed.")


if __name__ == "__main__":
    main()
