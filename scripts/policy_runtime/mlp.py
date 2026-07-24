from pathlib import Path

import numpy as np
import torch

from constant import ACTION_DIMS, EPSILON, MAX_ARM_DELTA
from models.mlp import MLP
from policy_runtime.types import ActionStep, NormDict
from sim import SimEnv

MLP_ARCHITECTURE = "mlp"


class MlpRuntime:
    architecture = MLP_ARCHITECTURE

    def __init__(
        self,
        *,
        model: MLP,
        device: torch.device,
        normalize: bool,
        action_space: str,
        norm_dict: NormDict,
    ) -> None:
        self._model = model
        self._device = device
        self.normalize = normalize
        self.action_space = action_space
        self._norm_dict = norm_dict

    @classmethod
    def from_checkpoint(
        cls,
        *,
        model_path: Path,
        device: torch.device,
        checkpoint: dict | None = None,
    ) -> "MlpRuntime":
        if checkpoint is None:
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )
        model = MLP().to(device)
        model.load_state_dict(checkpoint["model_dict"])
        model.eval()

        norm_dict: NormDict = {
            key: torch.from_numpy(checkpoint[key]).to(device).squeeze(0)
            for key in (
                "arm_actions_mean",
                "arm_actions_std",
                "arm_obs_mean",
                "arm_obs_std",
            )
        }
        return cls(
            model=model,
            device=device,
            normalize=checkpoint["normalize"],
            action_space=checkpoint["action_space"],
            norm_dict=norm_dict,
        )

    def observe(self, *, sim: SimEnv) -> np.ndarray:
        return sim.build_observation()

    def act(
        self,
        *,
        sim: SimEnv,
        observation: np.ndarray | None = None,
    ) -> ActionStep:
        obs = observation if observation is not None else self.observe(sim=sim)
        obs_tensor = torch.from_numpy(obs).to(self._device).unsqueeze(0)
        obs_target = obs_tensor
        if self.normalize:
            obs_target = (obs_tensor - self._norm_dict["arm_obs_mean"]) / (
                self._norm_dict["arm_obs_std"] + EPSILON
            )

        joints_pred, gripper_pred = self._model(obs_target)
        joints_pred_unnorm = joints_pred
        if self.normalize:
            joints_pred_unnorm = (
                joints_pred * (self._norm_dict["arm_actions_std"] + EPSILON)
                + self._norm_dict["arm_actions_mean"]
            )

        joints_actions = joints_pred_unnorm
        if self.action_space == "joint_delta":
            joints_actions = torch.clamp(
                joints_actions,
                min=-MAX_ARM_DELTA,
                max=MAX_ARM_DELTA,
            )
            joints_actions = joints_actions + obs_tensor[:, : ACTION_DIMS - 1]

        gripper_prob = torch.sigmoid(gripper_pred)
        gripper_actions = torch.where(gripper_prob >= 0.5, 255.0, 0.0)
        policy_actions = torch.concat([joints_actions, gripper_actions], dim=1)
        action = policy_actions.squeeze(0).detach().cpu().numpy()
        return {
            "observation": obs,
            "obs_tensor": obs_tensor,
            "policy_actions": policy_actions,
            "joints_pred": joints_pred,
            "joints_pred_unnorm": joints_pred_unnorm,
            "action": action,
        }
