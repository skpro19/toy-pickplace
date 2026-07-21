from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, TypedDict
import numpy as np
import torch

from constant import ACTION_DIMS, EPSILON, OBS_DIMS


class NormStats(TypedDict):
    arm_actions_mean: np.ndarray
    arm_actions_std: np.ndarray
    arm_obs_mean: np.ndarray
    arm_obs_std: np.ndarray


class PickPlaceDataset(Dataset):
    def __init__(
        self,
        *,
        data_dirs: str | Path | list[str | Path] = "data/demos/2026-07-02_14-04-52",
        sample_ratios: list[float] | None = None,
        dagger_intervention_ratio: float | None = None,
        action_space: Literal["joint_delta", "absolute"],
        normalize: bool,
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        episodes_by_dir = []
        for data_dir in data_dirs:
            episodes = []
            for file in sorted(Path(data_dir).glob("*.npz")):
                with np.load(file) as data:
                    obs = data["obs"]
                    actions = data["actions"]
                    execute_expert = (
                        np.asarray(data["execute_expert"], dtype=np.bool_).copy()
                        if "execute_expert" in data
                        else None
                    )

                    self._validate_episode(
                        file=file,
                        obs=obs,
                        actions=actions,
                        execute_expert=execute_expert,
                    )

                    episodes.append((obs, actions, execute_expert))

            if not episodes:
                raise ValueError(f"{data_dir} contains no .npz files")
            mask_presence = {
                execute_expert is not None for _, _, execute_expert in episodes
            }
            if len(mask_presence) != 1:
                raise ValueError(
                    f"{data_dir} mixes episodes with and without execute_expert"
                )
            episodes_by_dir.append(episodes)

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
                sum(obs.shape[0] for obs, _, _ in episodes)
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

        obs_by_episode = []
        actions_by_episode = []
        weights_by_episode = []
        for source_ratio, episodes in zip(ratios, episodes_by_dir):
            has_intervention_masks = episodes[0][2] is not None
            episode_weight = source_ratio / len(episodes)
            base_weights = [
                np.full(
                    obs.shape[0], episode_weight / obs.shape[0], dtype=np.float64
                )
                for obs, _, _ in episodes
            ]
            intervention_weight = sum(
                float(np.sum(weights[execute_expert]))
                for weights, (_, _, execute_expert) in zip(base_weights, episodes)
                if execute_expert is not None
            )
            non_intervention_weight = source_ratio - intervention_weight
            has_intervention_frames = any(
                bool(np.any(execute_expert))
                for _, _, execute_expert in episodes
                if execute_expert is not None
            )
            has_non_intervention_frames = any(
                bool(np.any(~execute_expert))
                for _, _, execute_expert in episodes
                if execute_expert is not None
            )

            for base_weight, (obs, actions, execute_expert) in zip(
                base_weights, episodes
            ):
                obs_by_episode.append(obs)
                actions_by_episode.append(actions)
                if dagger_intervention_ratio is None or not has_intervention_masks:
                    weights = base_weight
                elif not has_intervention_frames or not has_non_intervention_frames:
                    weights = base_weight
                else:
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

        self.obs = np.concatenate(obs_by_episode, axis=0)
        self.actions = np.concatenate(actions_by_episode, axis=0)
        self.sample_weights = np.concatenate(weights_by_episode, axis=0)
        self.source_ratios = ratios.copy()
        self.source_frame_counts = frame_counts

        if sample_ratios is None:
            self.samples_per_epoch = self.obs.shape[0]
        else:
            total_samples = int(np.floor(np.min(frame_counts / ratios)))
            self.samples_per_epoch = total_samples

        self.obs_targets, self.action_targets, self.norm_stats = self._build_targets(
            action_space=action_space, normalize=normalize
        )

    def _build_targets(
        self, *, action_space: Literal["joint_delta", "absolute"], normalize: bool
    ) -> tuple[np.ndarray, np.ndarray, NormStats]:
        action_targets = np.empty_like(self.actions)
        if action_space == "joint_delta":
            arm_actions = self.actions[:, 0:ACTION_DIMS-1]
            arm_qpos = self.obs[:, 0:ACTION_DIMS-1]
            action_targets[:, 0:ACTION_DIMS-1] = arm_actions - arm_qpos
            action_targets[:, ACTION_DIMS-1] = (
                self.actions[:, ACTION_DIMS-1] / 255.0
            )
        elif action_space == "absolute":
            action_targets[:, 0:ACTION_DIMS-1] = self.actions[:, 0:ACTION_DIMS-1]
            action_targets[:, ACTION_DIMS-1] = (
                self.actions[:, ACTION_DIMS-1] / 255.0
            )
        else:
            raise ValueError(f"Unknown action_space: {action_space}")

        obs_targets = self.obs.copy()

        arm_action_targets = action_targets[:, 0:ACTION_DIMS-1]
        arm_actions_mean = np.average(
            arm_action_targets,
            axis=0,
            weights=self.sample_weights,
        )[None, :].astype(np.float32)
        arm_obs_mean = np.average(
            obs_targets,
            axis=0,
            weights=self.sample_weights,
        )[None, :].astype(np.float32)
        norm_stats: NormStats = {
            "arm_actions_mean": arm_actions_mean,
            "arm_actions_std": np.sqrt(
                np.average(
                    np.square(arm_action_targets - arm_actions_mean),
                    axis=0,
                    weights=self.sample_weights,
                )
            )[None, :].astype(np.float32),
            "arm_obs_mean": arm_obs_mean,
            "arm_obs_std": np.sqrt(
                np.average(
                    np.square(obs_targets - arm_obs_mean),
                    axis=0,
                    weights=self.sample_weights,
                )
            )[None, :].astype(np.float32),
        }

        if normalize:
            action_targets[:, 0:ACTION_DIMS-1] -= norm_stats["arm_actions_mean"]
            action_targets[:, 0:ACTION_DIMS-1] /= (
                norm_stats["arm_actions_std"] + EPSILON
            )
            obs_targets -= norm_stats["arm_obs_mean"]
            obs_targets /= norm_stats["arm_obs_std"] + EPSILON

        return obs_targets, action_targets, norm_stats

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs_targets[idx]),
            torch.from_numpy(self.action_targets[idx]),
        )

    @staticmethod
    def _validate_episode(
        *,
        file: Path,
        obs: np.ndarray,
        actions: np.ndarray,
        execute_expert: np.ndarray | None,
    ) -> None:
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
