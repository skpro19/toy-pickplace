from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def build_weighted_dataloader(
    *,
    dataset: Dataset,
    batch_size: int,
    sample_seed: int,
    dataloader_workers: int,
    device: torch.device,
) -> DataLoader:
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=dataset.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(sample_seed),
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=dataloader_workers,
        pin_memory=device.type == "cuda" and dataloader_workers > 0,
    )


def print_dataset_sampling_summary(
    *,
    npz_folders: list[Path],
    dataset: Dataset,
) -> None:
    print(f"Weighted samples per epoch: {dataset.samples_per_epoch:,}")
    for data_dir, frame_count, source_ratio in zip(
        npz_folders,
        dataset.source_frame_counts,
        dataset.source_ratios,
    ):
        expected_samples = dataset.samples_per_epoch * float(source_ratio)
        print(
            f"  {expected_samples:,.1f} expected samples | "
            f"{int(frame_count):,} available frames | {data_dir}"
        )
