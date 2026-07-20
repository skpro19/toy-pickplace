from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import PickPlaceVisionDataset  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        episode_path = Path(temp_dir) / "episode_0.npz"
        img_obs = np.full((2, 64, 64, 3), 17, dtype=np.uint8)
        np.savez_compressed(
            episode_path,
            obs=np.full((2, 45), 1.0, dtype=np.float32),
            actions=np.full((2, 8), 2.0, dtype=np.float32),
            img_obs=img_obs,
        )

        dataset = PickPlaceVisionDataset(data_dirs=Path(temp_dir))
        obs, actions, loaded_img_obs = dataset[0]

        assert len(dataset) == 2
        assert obs.shape == (9,)
        assert actions.shape == (8,)
        assert loaded_img_obs.shape == (64, 64, 3)
        assert loaded_img_obs.dtype == torch.uint8
        assert int(loaded_img_obs[0, 0, 0]) == 17

    print("Vision dataset smoke test passed.")


if __name__ == "__main__":
    main()
