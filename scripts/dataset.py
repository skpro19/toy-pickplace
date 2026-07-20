from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


class PickPlaceVisionDataset(Dataset):
    def __init__(
        self,
        *,
        data_dirs: str | Path | list[str | Path] = "data/demos/2026-07-02_14-04-52",
        sample_ratios: list[float] | None = None,
        dagger_intervention_ratio: float | None = None,
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
                    if "img_obs" not in data:
                        raise KeyError(f"{file} does not contain an 'img_obs' array")
                    img_obs = data["img_obs"]
                    execute_expert = (
                        np.asarray(data["execute_expert"], dtype=np.bool_).copy()
                        if "execute_expert" in data
                        else None
                    )

                    # error checking
                    # -----------------
                    if obs.shape[0] != actions.shape[0]:
                        raise ValueError(
                            f"{file} obs/actions length mismatch: "
                            f"{obs.shape[0]} vs {actions.shape[0]}"
                        )
                    if obs.ndim != 2:
                        raise ValueError(
                            f"{file} obs must have shape (T,obs_dim), got {obs.shape}"
                        )
                    if actions.ndim != 2:
                        raise ValueError(
                            f"{file} actions must have shape (T,action_dim), "
                            f"got {actions.shape}"
                        )
                    if obs.shape[1] != 45:
                        raise ValueError(f"{file} expected obs_dim=45 got {obs.shape[1]}")
                    if actions.shape[1] != 8:
                        raise ValueError(f"{file} expected action_dim=8 got {actions.shape[1]}")
                    if obs.shape[0] == 0:
                        raise ValueError(f"{file} contains no timesteps")
                    if img_obs.ndim != 4 or img_obs.shape[0] != obs.shape[0]:
                        raise ValueError(
                            f"{file} img_obs must have shape (T, H, W, C), got "
                            f"{img_obs.shape}"
                        )
                    if img_obs.dtype != np.uint8:
                        raise ValueError(
                            f"{file} img_obs must have dtype uint8, got {img_obs.dtype}"
                        )
                    if execute_expert is not None and (
                        execute_expert.ndim != 1
                        or execute_expert.shape[0] != obs.shape[0]
                    ):
                        raise ValueError(
                            f"{file} execute_expert must have shape ({obs.shape[0]},), "
                            f"got {execute_expert.shape}"
                        )
                    # -----------------

                    episodes.append((obs, actions, img_obs, execute_expert))

            if not episodes:
                raise ValueError(f"{data_dir} contains no .npz files")
            mask_presence = {
                execute_expert is not None for _, _, _, execute_expert in episodes
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
                sum(obs.shape[0] for obs, _, _, _ in episodes)
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
        img_obs_by_episode = []
        weights_by_episode = []
        for source_ratio, episodes in zip(ratios, episodes_by_dir):
            has_intervention_masks = episodes[0][3] is not None
            episode_weight = source_ratio / len(episodes)
            base_weights = [
                np.full(
                    obs.shape[0], episode_weight / obs.shape[0], dtype=np.float64
                )
                for obs, _, _, _ in episodes
            ]
            intervention_weight = sum(
                float(np.sum(weights[execute_expert]))
                for weights, (_, _, _, execute_expert) in zip(base_weights, episodes)
                if execute_expert is not None
            )
            non_intervention_weight = source_ratio - intervention_weight
            has_intervention_frames = any(
                bool(np.any(execute_expert))
                for _, _, _, execute_expert in episodes
                if execute_expert is not None
            )
            has_non_intervention_frames = any(
                bool(np.any(~execute_expert))
                for _, _, _, execute_expert in episodes
                if execute_expert is not None
            )

            for base_weight, (obs, actions, img_obs, execute_expert) in zip(
                base_weights, episodes
            ):
                obs_by_episode.append(obs)
                actions_by_episode.append(actions)
                img_obs_by_episode.append(img_obs)
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
        self.img_obs = np.concatenate(img_obs_by_episode, axis=0)
        self.sample_weights = np.concatenate(weights_by_episode, axis=0)
        self.source_ratios = ratios.copy()
        self.source_frame_counts = frame_counts

        if sample_ratios is None:
            self.samples_per_epoch = self.obs.shape[0]
        else:
            total_samples = int(np.floor(np.min(frame_counts / ratios)))
            self.samples_per_epoch = total_samples

        # values that would be used by the model
        self.action_targets = np.empty_like(self.actions)
        self.obs_targets = np.empty_like(self.obs)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs_targets[idx][:9]),
            torch.from_numpy(self.action_targets[idx]),
            torch.from_numpy(self.img_obs[idx])
        )


class PickPlaceDataset(Dataset):
    def __init__(
        self,
        *,
        data_dirs: str | Path | list[str | Path] = "data/demos/2026-07-02_14-04-52",
        sample_ratios: list[float] | None = None,
        dagger_intervention_ratio: float | None = None,
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

                    if obs.shape[0] != actions.shape[0]:
                        raise ValueError(
                            f"{file} obs/actions length mismatch: "
                            f"{obs.shape[0]} vs {actions.shape[0]}"
                        )
                    if obs.ndim != 2:
                        raise ValueError(
                            f"{file} obs must have shape (T,obs_dim), got {obs.shape}"
                        )
                    if actions.ndim != 2:
                        raise ValueError(
                            f"{file} actions must have shape (T,action_dim), "
                            f"got {actions.shape}"
                        )
                    if obs.shape[1] != 45:
                        raise ValueError(f"{file} expected obs_dim=45 got {obs.shape[1]}")
                    if actions.shape[1] != 8:
                        raise ValueError(f"{file} expected action_dim=8 got {actions.shape[1]}")
                    if obs.shape[0] == 0:
                        raise ValueError(f"{file} contains no timesteps")
                    if execute_expert is not None and (
                        execute_expert.ndim != 1
                        or execute_expert.shape[0] != obs.shape[0]
                    ):
                        raise ValueError(
                            f"{file} execute_expert must have shape ({obs.shape[0]},), "
                            f"got {execute_expert.shape}"
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

        # values that would be used by the model
        self.action_targets = np.empty_like(self.actions)
        self.obs_targets = np.empty_like(self.obs)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs_targets[idx]),
            torch.from_numpy(self.action_targets[idx]),
        )
