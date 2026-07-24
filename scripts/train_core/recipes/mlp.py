from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from constant import ACTION_DIMS, OBS_DIMS
from models.mlp import MLP
from policy_runtimes.mlp import MLP_ARCHITECTURE
from scripts.dataset import NormStats, PickPlaceDataset
from train_core.loss import compute_dual_head_bc_loss
from train_core.types import LossStep, TrainingRecipe


class MlpRecipe(TrainingRecipe):
    architecture = MLP_ARCHITECTURE

    def build_dataset(
        self,
        *,
        npz_folders: list[Path],
        sample_ratios: list[float] | None,
        dagger_intervention_ratio: float | None,
        action_space: str,
        normalize: bool,
    ) -> tuple[Dataset, NormStats]:
        dataset = PickPlaceDataset(
            data_dirs=npz_folders,
            sample_ratios=sample_ratios,
            dagger_intervention_ratio=dagger_intervention_ratio,
            action_space=action_space,
            normalize=normalize,
        )
        return dataset, dataset.norm_stats

    def build_model(self, *, device: torch.device) -> nn.Module:
        return MLP(obs_dim=OBS_DIMS, action_dim=ACTION_DIMS).to(device)

    def build_optimizer(self, *, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_step(
        self,
        *,
        model: nn.Module,
        batch: object,
        device: torch.device,
        joint_loss_fn: nn.Module,
        gripper_loss_fn: nn.Module,
    ) -> LossStep:
        obs_target, action_target = batch
        obs_target = obs_target.to(device)
        action_target = action_target.to(device)

        joints_pred, gripper_pred = model(obs_target)
        return compute_dual_head_bc_loss(
            joints_pred=joints_pred,
            gripper_pred=gripper_pred,
            action_target=action_target,
            joint_loss_fn=joint_loss_fn,
            gripper_loss_fn=gripper_loss_fn,
        )


def prepare_dataset(
    *,
    npz_folders: list[Path],
    sample_ratios: list[float] | None,
    dagger_intervention_ratio: float | None,
    action_space: str,
    normalize: bool,
) -> tuple[PickPlaceDataset, NormStats]:
    return MlpRecipe().build_dataset(
        npz_folders=npz_folders,
        sample_ratios=sample_ratios,
        dagger_intervention_ratio=dagger_intervention_ratio,
        action_space=action_space,
        normalize=normalize,
    )
