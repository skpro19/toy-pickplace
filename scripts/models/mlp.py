import torch
from torch import nn


class MLP(nn.Module):

    def __init__(
        self,
        *,
        obs_dim: int = 45,
        action_dim: int = 8,
    ):
        super().__init__()

        self.input_layer = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()

        self.h1 = nn.Linear(128, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, 128)

        self.joints_head = nn.Linear(128, action_dim - 1)
        self.gripper_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.h1(x)
        x = self.relu(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.relu(x)

        joints_logits = self.joints_head(x)
        gripper_logits = self.gripper_head(x)

        return (joints_logits, gripper_logits)
