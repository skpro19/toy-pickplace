from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import Dataset

from eval import EvalSelectionMode
from scripts.dataset import NormStats


class EpochMetrics(TypedDict):
    loss: float
    joints_loss: float
    gripper_loss: float


class LossStep(TypedDict):
    loss: torch.Tensor
    joints_loss: float
    gripper_loss: float


class TrainConfig(TypedDict):
    num_epochs: int
    batch_size: int
    npz_folders: list[Path]
    checkpoint_dir: Path
    log_dir: Path
    normalize: bool
    action_space: str
    sample_ratios: list[float] | None
    dagger_intervention_ratio: float | None
    sample_seed: int
    eval_interval: int
    eval_seed: int
    eval_episodes: int
    eval_max_steps: int
    eval_workers: int
    eval_capture_hz: float
    dataloader_workers: int
    persistent_workers: bool
    early_stop_patience: int
    eval_selection_mode: EvalSelectionMode


class TrainingRecipe:
    architecture: str

    def build_dataset(
        self,
        *,
        npz_folders: list[Path],
        sample_ratios: list[float] | None,
        dagger_intervention_ratio: float | None,
        action_space: str,
        normalize: bool,
    ) -> tuple[Dataset, NormStats]:
        raise NotImplementedError

    def build_model(self, *, device: torch.device) -> nn.Module:
        raise NotImplementedError

    def build_optimizer(self, *, model: nn.Module) -> torch.optim.Optimizer:
        raise NotImplementedError

    def train_step(
        self,
        *,
        model: nn.Module,
        batch: object,
        device: torch.device,
        joint_loss_fn: nn.Module,
        gripper_loss_fn: nn.Module,
    ) -> LossStep:
        raise NotImplementedError
