from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from models.vision_mlp import VisionMLP
from policy_runtimes.vision_mlp import VISION_MLP_ARCHITECTURE
from scripts.dataset import NormStats, PickPlaceVisionDataset
from train_core.loss import compute_dual_head_bc_loss
from train_core.types import LossStep, TrainingRecipe


class VisionMlpRecipe(TrainingRecipe):
    architecture = VISION_MLP_ARCHITECTURE

    def build_dataset(
        self,
        *,
        npz_folders: list[Path],
        sample_ratios: list[float] | None,
        dagger_intervention_ratio: float | None,
        action_space: str,
        normalize: bool,
    ) -> tuple[Dataset, NormStats]:
        dataset = PickPlaceVisionDataset(
            data_dirs=npz_folders,
            sample_ratios=sample_ratios,
            dagger_intervention_ratio=dagger_intervention_ratio,
            action_space=action_space,
            normalize=normalize,
        )
        return dataset, dataset.norm_stats

    def build_model(self, *, device: torch.device, dropout: float) -> nn.Module:
        return VisionMLP(dropout=dropout).to(device)

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
        obs_proprio, action_target, obs_img = batch
        obs_proprio = obs_proprio.to(device)
        action_target = action_target.to(device)
        obs_img = obs_img.to(device)

        joints_pred, gripper_pred = model(obs_proprio, obs_img)
        return compute_dual_head_bc_loss(
            joints_pred=joints_pred,
            gripper_pred=gripper_pred,
            action_target=action_target,
            joint_loss_fn=joint_loss_fn,
            gripper_loss_fn=gripper_loss_fn,
        )
