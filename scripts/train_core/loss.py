import torch
from torch import nn

from constant import ACTION_DIMS, GRIPPER_LOSS_WEIGHT, JOINTS_LOSS_WEIGHT
from train_core.types import LossStep


def compute_dual_head_bc_loss(
    *,
    joints_pred: torch.Tensor,
    gripper_pred: torch.Tensor,
    action_target: torch.Tensor,
    joint_loss_fn: nn.Module,
    gripper_loss_fn: nn.Module,
) -> LossStep:
    joints_target = action_target[:, : ACTION_DIMS - 1]
    gripper_target = action_target[:, ACTION_DIMS - 1].unsqueeze(1)

    joints_loss = joint_loss_fn(joints_pred, joints_target)
    gripper_loss = gripper_loss_fn(gripper_pred, gripper_target)
    loss = JOINTS_LOSS_WEIGHT * joints_loss + GRIPPER_LOSS_WEIGHT * gripper_loss

    return {
        "loss": loss,
        "joints_loss": joints_loss.item(),
        "gripper_loss": gripper_loss.item(),
    }
