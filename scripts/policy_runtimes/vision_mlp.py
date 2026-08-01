from pathlib import Path

import numpy as np
import torch

from constant import ACTION_DIMS, EPSILON, MAX_ARM_DELTA, PROPRIO_DIMS
from models.vision_mlp import VisionMLP
from policy_runtimes.types import ActionStep, NormDict
from sim import SimEnv

VISION_MLP_ARCHITECTURE = "vision_mlp"


class VisionMlpRuntime:
    architecture = VISION_MLP_ARCHITECTURE

    def __init__(
        self,
        *,
        model: VisionMLP,
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
    ) -> "VisionMlpRuntime":
        if checkpoint is None:
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )
        if "dropout" not in checkpoint:
            raise ValueError(
                f"Checkpoint is missing required field 'dropout': {model_path}"
            )
        model = VisionMLP(dropout=float(checkpoint["dropout"])).to(device)
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

    def _prepare_proprio(self, *, obs: np.ndarray) -> torch.Tensor:
        proprio = np.asarray(obs[:PROPRIO_DIMS], dtype=np.float32)
        proprio_tensor = torch.from_numpy(proprio).to(self._device).unsqueeze(0)
        if self.normalize:
            proprio_tensor = (proprio_tensor - self._norm_dict["arm_obs_mean"]) / (
                self._norm_dict["arm_obs_std"] + EPSILON
            )
        return proprio_tensor

    def _prepare_image_tensor(self, *, img_obs: np.ndarray) -> torch.Tensor:
        img = np.asarray(img_obs, dtype=np.float32).copy()
        img /= 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def act(
        self,
        *,
        sim: SimEnv,
        observation: np.ndarray | None = None,
    ) -> ActionStep:
        obs = observation if observation is not None else self.observe(sim=sim)
        obs_tensor = torch.from_numpy(obs).to(self._device).unsqueeze(0)
        proprio_tensor = self._prepare_proprio(obs=obs)
        img_obs = np.asarray(sim.build_image(), dtype=np.uint8).copy()
        img_tensor = self._prepare_image_tensor(img_obs=img_obs)

        joints_pred, gripper_pred = self._model(proprio_tensor, img_tensor)
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
            "img_obs": img_obs,
        }
