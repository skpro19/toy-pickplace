from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


class PickPlaceDataset(Dataset):
    def __init__(
        self,
        *,
        data_dirs: str | list[str] = "data/demos/2026-07-02_14-04-52",
        sample_ratios: list[float] | None = None,
        seed: int = 0,
    ):
        obs_by_dir = []
        actions_by_dir = []

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        for data_dir in data_dirs:
            dir_obs = []
            dir_actions = []
            for file in sorted(Path(data_dir).glob("*.npz")):
                with np.load(file) as data:
                    obs = data["obs"]
                    actions = data["actions"]

                    if obs.shape[0] != actions.shape[0]:
                        raise ValueError(f"{file} obs/actions length mismatch: {obs.shape[0]} vs {actions.shape[0]}")
                    if obs.ndim != 2:
                        raise ValueError(f"{file} obs must have shape (T,obs_dim), got {obs.shape}")
                    if actions.ndim != 2:
                        raise ValueError(f"{file} actions must have shape (T,action_dim), got {actions.shape}")
                    if obs.shape[1] != 45:
                        raise ValueError(f"{file} expected obs_dim=45 got {obs.shape[1]}")
                    if actions.shape[1] != 8:
                        raise ValueError(f"{file} expected action_dim=8 got {actions.shape[1]}")

                    dir_obs.append(obs)
                    dir_actions.append(actions)

            if not dir_obs:
                raise ValueError(f"{data_dir} contains no .npz files")

            obs_by_dir.append(np.concatenate(dir_obs, axis=0))
            actions_by_dir.append(np.concatenate(dir_actions, axis=0))

        if sample_ratios is None:
            self.obs = np.concatenate(obs_by_dir, axis=0)
            self.actions = np.concatenate(actions_by_dir, axis=0)
        else:
            if len(sample_ratios) != len(obs_by_dir):
                raise ValueError(
                    f"sample_ratios length ({len(sample_ratios)}) must match data_dirs length ({len(obs_by_dir)})"
                )

            ratios = np.asarray(sample_ratios, dtype=np.float64)
            if np.any(ratios <= 0.0):
                raise ValueError(f"sample_ratios must be positive, got {sample_ratios}")

            ratios = ratios / ratios.sum()
            counts = np.asarray([obs.shape[0] for obs in obs_by_dir], dtype=np.float64)
            total_samples = int(np.floor(np.min(counts / ratios)))
            sample_counts = np.floor(total_samples * ratios).astype(np.int64)

            rng = np.random.default_rng(seed)
            sampled_obs = []
            sampled_actions = []
            for obs, actions, sample_count in zip(obs_by_dir, actions_by_dir, sample_counts):
                indices = rng.choice(obs.shape[0], size=int(sample_count), replace=False)
                sampled_obs.append(obs[indices])
                sampled_actions.append(actions[indices])

            self.obs = np.concatenate(sampled_obs, axis=0)
            self.actions = np.concatenate(sampled_actions, axis=0)
         
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
