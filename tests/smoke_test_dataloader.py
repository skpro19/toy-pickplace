from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from train_core.dataloader import build_weighted_dataloader  # noqa: E402


class WeightedDataset(Dataset):
    def __init__(self) -> None:
        self.values = torch.arange(8)
        self.sample_weights = np.ones(8, dtype=np.float64)
        self.samples_per_epoch = 8

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


def main() -> None:
    dataset = WeightedDataset()
    dataloader = build_weighted_dataloader(
        dataset=dataset,
        batch_size=4,
        sample_seed=0,
        dataloader_workers=2,
        persistent_workers=True,
        device=torch.device("cpu"),
    )
    assert dataloader.num_workers == 2
    assert dataloader.persistent_workers

    try:
        build_weighted_dataloader(
            dataset=dataset,
            batch_size=4,
            sample_seed=0,
            dataloader_workers=0,
            persistent_workers=True,
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "dataloader_workers > 0" in str(error)
    else:
        raise AssertionError("persistent workers with zero workers were accepted")


if __name__ == "__main__":
    main()
