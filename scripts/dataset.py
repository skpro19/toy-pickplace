from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


class PickPlaceDataset(Dataset):
    def __init__(
        self,
        *,
        data_dirs: str | Path | list[str | Path] = "data/demos/2026-07-02_14-04-52",
        sample_ratios: list[float] | None = None,
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

                    episodes.append((obs, actions))

            if not episodes:
                raise ValueError(f"{data_dir} contains no .npz files")
            episodes_by_dir.append(episodes)

        episode_counts = np.asarray(
            [len(episodes) for episodes in episodes_by_dir],
            dtype=np.float64,
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
            episode_weight = source_ratio / len(episodes)
            for obs, actions in episodes:
                obs_by_episode.append(obs)
                actions_by_episode.append(actions)
                weights_by_episode.append(
                    np.full(
                        obs.shape[0],
                        episode_weight / obs.shape[0],
                        dtype=np.float64,
                    )
                )

        self.obs = np.concatenate(obs_by_episode, axis=0)
        self.actions = np.concatenate(actions_by_episode, axis=0)
        self.sample_weights = np.concatenate(weights_by_episode, axis=0)

        if sample_ratios is None:
            self.samples_per_epoch = self.obs.shape[0]
        else:
            counts = np.asarray(
                [
                    sum(obs.shape[0] for obs, _ in episodes)
                    for episodes in episodes_by_dir
                ],
                dtype=np.float64,
            )
            total_samples = int(np.floor(np.min(counts / ratios)))
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
