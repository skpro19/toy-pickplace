from typing import TypedDict

import numpy as np
import torch

from sim import SimEnv


class NormDict(TypedDict):
    arm_actions_mean: torch.Tensor
    arm_actions_std: torch.Tensor
    arm_obs_mean: torch.Tensor
    arm_obs_std: torch.Tensor


class ActionStep(TypedDict):
    observation: np.ndarray
    obs_tensor: torch.Tensor
    policy_actions: torch.Tensor
    joints_pred: torch.Tensor
    joints_pred_unnorm: torch.Tensor
    action: np.ndarray
    img_obs: np.ndarray | None


class PolicyRuntime:
    architecture: str
    normalize: bool
    action_space: str

    def observe(self, *, sim: SimEnv) -> np.ndarray:
        raise NotImplementedError

    def act(
        self,
        *,
        sim: SimEnv,
        observation: np.ndarray | None = None,
    ) -> ActionStep:
        raise NotImplementedError
