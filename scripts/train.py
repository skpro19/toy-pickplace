import torch 
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class MLP(nn.Module):

    def __init__(self): 
        pass

    def forward(self, x):
        pass

class PickPlaceDataset(Dataset):
    def __init__(self, *, data_dir: str):
        # self.files = []
        self.obs = []
        self.actions = []

        for file in sorted(Path(data_dir).glob("*.npz")):
            with np.load(file) as data:
                obs = data["obs"]
                actions = data["actions"]

                # validation checks
                if obs.shape[0] != actions.shape[0]:
                    raise ValueError(f"{file} obs/actions length mismatch: {obs.shape[0]} vs {actions.shape[0]}")
                if obs.ndims != 2: 
                    raise ValueError(f"{file} obs must have shape (T,obs_dim), got {obs.shape}")
                if actions.ndim != 2:
                    raise ValueError(f"{file} actions must have shape (T,action_dim), got {actions.shape}")
                if obs.shape[1] != 40:
                    raise ValueError(f"{file} expected obs_dim=40 got {obs.shape[1]}")
                if actions.shape[1] != 8:
                    raise ValueError(f"{file} expected action_dim=8 got {actions.shape[1]}")


                self.obs.append(obs)
                self.actions.append(actions)
                # break
        
        # print(f"self.obs.shape: {len(self.obs)}")
        # print(f"self.actions.shape: {len(self.actions)}")

        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        # print(f"self.obs.shape: {self.obs.shape}")
        # print(f"self.actions.shape: {self.actions.shape}")

    def __len__(self):
        return self.obs.shape[0]


    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]), 
            torch.from_numpy(self.actions[idx]),
        )
        

def train(): 
    pass


def main():
    train()

if __name__ == "__main__":
    main()