from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from numpy.lib.npyio import NpzFile
import torch
from torch.utils.data import Dataset

from .constant import (
    ACTION_DIMS,
    EPSILON,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBS_DIMS,
    PROPRIO_DIMS,
)


class NormStats(TypedDict):
    arm_actions_mean: np.ndarray
    arm_actions_std: np.ndarray
    arm_obs_mean: np.ndarray
    arm_obs_std: np.ndarray


class EpisodeData(TypedDict):
    obs: np.ndarray
    actions: np.ndarray
    execute_expert: np.ndarray | None
    extra: dict[str, np.ndarray]



class PickPlaceDataset(Dataset):
    def __init__(
        self,
        *,
        data_dirs: str | Path | list[str | Path] = "data/demos/2026-07-02_14-04-52",
        sample_ratios: list[float] | None = None,
        dagger_intervention_ratio: float | None = None,
        action_space: Literal["joint_delta", "absolute"] = "joint_delta",
        normalize: bool = True,
    ) -> None:
        if action_space not in ("joint_delta", "absolute"):
            raise ValueError(f"Unknown action_space: {action_space}")

        self.action_space = action_space
        self.normalize = normalize

        parsed_data_dirs = self._parse_data_dirs(data_dirs=data_dirs)
        episodes_by_dir = self._load_episodes(data_dirs=parsed_data_dirs)

        self._build_sampling_metadata(
            episodes_by_dir=episodes_by_dir,
            sample_ratios=sample_ratios,
            dagger_intervention_ratio=dagger_intervention_ratio,
        )
        self._concatenate_episodes(episodes_by_dir=episodes_by_dir)
        self._concatenate_extra_episode_data(episodes_by_dir=episodes_by_dir)
        self.norm_stats = self._build_norm_stats()

    def _parse_data_dirs(
        self,
        *,
        data_dirs: str | Path | list[str | Path],) -> list[Path]:
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        return [Path(data_dir) for data_dir in data_dirs]

    def _load_episodes(
        self,
        *,
        data_dirs: list[Path],) -> list[list[EpisodeData]]:
        episodes_by_dir: list[list[EpisodeData]] = []
        for data_dir in data_dirs:
            episodes = [
                self._load_episode_data(file=file)
                for file in sorted(data_dir.glob("*.npz"))
            ]
            if not episodes:
                raise ValueError(f"{data_dir} contains no .npz files")

            mask_presence = {
                episode["execute_expert"] is not None for episode in episodes
            }
            if len(mask_presence) != 1:
                raise ValueError(
                    f"{data_dir} mixes episodes with and without execute_expert"
                )
            episodes_by_dir.append(episodes)

        return episodes_by_dir

    def _load_episode_data(self, *, file: Path) -> EpisodeData:
        with np.load(file) as data:
            obs = data["obs"]
            actions = data["actions"]
            execute_expert = (
                np.asarray(data["execute_expert"], dtype=np.bool_).copy()
                if "execute_expert" in data
                else None
            )
            extra = self._load_extra_episode_data(file=file, data=data)

        episode: EpisodeData = {
            "obs": obs,
            "actions": actions,
            "execute_expert": execute_expert,
            "extra": extra,
        }
        self._validate_episode(
            file=file,
            obs=obs,
            actions=actions,
            execute_expert=execute_expert,
        )
        self._validate_extra_episode_data(file=file, episode=episode)
        return episode

    def _load_extra_episode_data(
        self,
        *,
        file: Path,
        data: NpzFile,) -> dict[str, np.ndarray]:
        return {}

    def _validate_extra_episode_data(
        self,
        *,
        file: Path,
        episode: EpisodeData,) -> None:
        return None

    def _build_sampling_metadata(
        self,
        *,
        episodes_by_dir: list[list[EpisodeData]],
        sample_ratios: list[float] | None,
        dagger_intervention_ratio: float | None,) -> None:
        if dagger_intervention_ratio is not None and not (
            0.0 <= dagger_intervention_ratio <= 1.0
        ):
            raise ValueError(
                "dagger_intervention_ratio must be between 0 and 1, got "
                f"{dagger_intervention_ratio}"
            )

        episode_counts = np.asarray(
            [len(episodes) for episodes in episodes_by_dir],
            dtype=np.float64,
        )
        frame_counts = np.asarray(
            [
                sum(episode["obs"].shape[0] for episode in episodes)
                for episodes in episodes_by_dir
            ],
            dtype=np.int64,
        )
        if sample_ratios is not None:
            if len(sample_ratios) != len(episodes_by_dir):
                raise ValueError(
                    f"sample_ratios length ({len(sample_ratios)}) must match "
                    f"data_dirs length ({len(episodes_by_dir)})"
                )

            ratios = np.asarray(sample_ratios, dtype=np.float64)
            if np.any(ratios <= 0.0):
                raise ValueError(f"sample_ratios must be positive, got {sample_ratios}")
            ratios = ratios / ratios.sum()
        else:
            ratios = episode_counts / episode_counts.sum()

        weights_by_episode: list[np.ndarray] = []
        for source_ratio, episodes in zip(ratios, episodes_by_dir):
            has_intervention_masks = episodes[0]["execute_expert"] is not None
            episode_weight = source_ratio / len(episodes)
            base_weights = [
                np.full(
                    episode["obs"].shape[0],
                    episode_weight / episode["obs"].shape[0],
                    dtype=np.float64,
                )
                for episode in episodes
            ]
            intervention_weight = sum(
                float(np.sum(weights[execute_expert]))
                for weights, episode in zip(base_weights, episodes)
                if (execute_expert := episode["execute_expert"]) is not None
            )
            non_intervention_weight = source_ratio - intervention_weight
            has_intervention_frames = any(
                bool(np.any(execute_expert))
                for episode in episodes
                if (execute_expert := episode["execute_expert"]) is not None
            )
            has_non_intervention_frames = any(
                bool(np.any(~execute_expert))
                for episode in episodes
                if (execute_expert := episode["execute_expert"]) is not None
            )

            for base_weight, episode in zip(base_weights, episodes):
                execute_expert = episode["execute_expert"]
                if dagger_intervention_ratio is None or not has_intervention_masks:
                    weights = base_weight
                elif not has_intervention_frames or not has_non_intervention_frames:
                    weights = base_weight
                else:
                    assert execute_expert is not None
                    weights = np.where(
                        execute_expert,
                        base_weight
                        * source_ratio
                        * dagger_intervention_ratio
                        / intervention_weight,
                        base_weight
                        * source_ratio
                        * (1.0 - dagger_intervention_ratio)
                        / non_intervention_weight,
                    )
                weights_by_episode.append(weights)

        self.sample_weights = np.concatenate(weights_by_episode, axis=0)
        self.source_ratios = ratios.copy()
        self.source_frame_counts = frame_counts
        if sample_ratios is None:
            self.samples_per_epoch = int(frame_counts.sum())
        else:
            self.samples_per_epoch = int(np.floor(np.min(frame_counts / ratios)))

    def _concatenate_episodes(
        self,
        *,
        episodes_by_dir: list[list[EpisodeData]],) -> None:
        episodes = [episode for source in episodes_by_dir for episode in source]
        self.obs = np.concatenate(
            [episode["obs"] for episode in episodes],
            axis=0,
        )
        self.actions = np.concatenate(
            [episode["actions"] for episode in episodes],
            axis=0,
        )

    def _concatenate_extra_episode_data(
        self,
        *,
        episodes_by_dir: list[list[EpisodeData]],) -> None:
        return None

    def _select_obs(self, *, obs: np.ndarray) -> np.ndarray:
        return obs

    def _build_arm_action_targets(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,) -> np.ndarray:
        if self.action_space == "joint_delta":
            return actions[..., : ACTION_DIMS - 1] - obs[..., : ACTION_DIMS - 1]
        return actions[..., : ACTION_DIMS - 1]

    def _build_norm_stats(self) -> NormStats:
        obs_values = self._select_obs(obs=self.obs)
        arm_action_values = self._build_arm_action_targets(
            obs=self.obs,
            actions=self.actions,
        )
        arm_actions_mean = np.average(
            arm_action_values,
            axis=0,
            weights=self.sample_weights,
        )[None, :].astype(np.float32)
        arm_obs_mean = np.average(
            obs_values,
            axis=0,
            weights=self.sample_weights,
        )[None, :].astype(np.float32)
        return {
            "arm_actions_mean": arm_actions_mean,
            "arm_actions_std": np.sqrt(
                np.average(
                    np.square(arm_action_values - arm_actions_mean),
                    axis=0,
                    weights=self.sample_weights,
                )
            )[None, :].astype(np.float32),
            "arm_obs_mean": arm_obs_mean,
            "arm_obs_std": np.sqrt(
                np.average(
                    np.square(obs_values - arm_obs_mean),
                    axis=0,
                    weights=self.sample_weights,
                )
            )[None, :].astype(np.float32),
        }

    def _prepare_obs(self, *, idx: int) -> torch.Tensor:
        obs = np.asarray(
            self._select_obs(obs=self.obs[idx]),
            dtype=np.float32,
        ).copy()
        if self.normalize:
            obs -= self.norm_stats["arm_obs_mean"][0]
            obs /= self.norm_stats["arm_obs_std"][0] + EPSILON
        return torch.from_numpy(obs)

    def _prepare_action(self, *, idx: int) -> torch.Tensor:
        action = np.empty(ACTION_DIMS, dtype=np.float32)
        action[: ACTION_DIMS - 1] = self._build_arm_action_targets(
            obs=self.obs[idx],
            actions=self.actions[idx],
        )
        action[ACTION_DIMS - 1] = self.actions[idx, ACTION_DIMS - 1] / 255.0
        if self.normalize:
            action[: ACTION_DIMS - 1] -= self.norm_stats["arm_actions_mean"][0]
            action[: ACTION_DIMS - 1] /= (
                self.norm_stats["arm_actions_std"][0] + EPSILON
            )
        return torch.from_numpy(action)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._prepare_obs(idx=idx),
            self._prepare_action(idx=idx),
        )

    @staticmethod
    def _validate_episode(
        *,
        file: Path,
        obs: np.ndarray,
        actions: np.ndarray,
        execute_expert: np.ndarray | None,) -> None:
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"{file} obs/actions length mismatch: "
                f"{obs.shape[0]} vs {actions.shape[0]}"
            )
        if obs.ndim != 2:
            raise ValueError(f"{file} obs must have shape (T,obs_dim), got {obs.shape}")
        if actions.ndim != 2:
            raise ValueError(
                f"{file} actions must have shape (T,action_dim), got {actions.shape}"
            )
        if obs.shape[1] != OBS_DIMS:
            raise ValueError(f"{file} expected obs_dim={OBS_DIMS} got {obs.shape[1]}")
        if actions.shape[1] != ACTION_DIMS:
            raise ValueError(
                f"{file} expected action_dim={ACTION_DIMS} got {actions.shape[1]}"
            )
        if obs.shape[0] == 0:
            raise ValueError(f"{file} contains no timesteps")
        if execute_expert is not None and (
            execute_expert.ndim != 1 or execute_expert.shape[0] != obs.shape[0]
        ):
            raise ValueError(
                f"{file} execute_expert must have shape ({obs.shape[0]},), "
                f"got {execute_expert.shape}"
            )


class PickPlaceVisionDataset(PickPlaceDataset):
    def _load_extra_episode_data(
        self,
        *,
        file: Path,
        data: NpzFile,
        ) -> dict[str, np.ndarray]:
        if "img_obs" not in data:
            raise KeyError(f"{file} does not contain an 'img_obs' array")
        return {"img_obs": data["img_obs"]}


    def _validate_extra_episode_data(
        self,
        *,
        file: Path,
        episode: EpisodeData,) -> None:
        img_obs = episode["extra"]["img_obs"]
        expected_shape = (
            episode["obs"].shape[0],
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        )
        if img_obs.shape != expected_shape:
            raise ValueError(
                f"{file} img_obs must have shape {expected_shape}, got {img_obs.shape}"
            )
        if img_obs.dtype != np.uint8:
            raise ValueError(f"{file} img_obs must have dtype uint8, got {img_obs.dtype}")

    def _concatenate_extra_episode_data(
        self,
        *,
        episodes_by_dir: list[list[EpisodeData]],) -> None:

        episodes = [episode for source in episodes_by_dir for episode in source]
        self.img_obs = np.concatenate(
            [episode["extra"]["img_obs"] for episode in episodes],
            axis=0,
        )

    def _select_obs(self, *, obs: np.ndarray) -> np.ndarray:
        return obs[..., :PROPRIO_DIMS]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._prepare_obs(idx=idx),
            self._prepare_action(idx=idx),
            self._prepare_img_obs(idx=idx)
        )

    def _prepare_img_obs(self, *, idx: int) -> torch.Tensor:
        img_obs = np.asarray(self.img_obs[idx], dtype=np.float32).copy()
        img_obs /= 255.0
        img_obs = torch.from_numpy(img_obs).permute(2, 0, 1)
        return img_obs
