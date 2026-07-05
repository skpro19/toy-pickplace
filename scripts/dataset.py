from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


class PickPlaceDataset(Dataset):
    def __init__(self, *, data_dir: str = "data/demos/2026-07-02_14-04-52"):
        self.obs = []
        self.actions = []
        self.targets = []

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
                    raise ValueError(f"{file} expected obs_dim=40 got {obs.shape[1]}")
                if actions.shape[1] != 8:
                    raise ValueError(f"{file} expected action_dim=8 got {actions.shape[1]}")

                self.obs.append(obs)
                self.actions.append(actions)
                
        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        
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
