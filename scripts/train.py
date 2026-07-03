import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse


class MLP(nn.Module):

    def __init__(self, *, 
                obs_dim: int=40, 
                action_dim: int=8, 
                ): 
        super().__init__()

        # layers
        self.input_layer = nn.Linear(obs_dim, 128)
        # hidden layers
        self.h1 = nn.Linear(128, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim) 
        self.relu = nn.ReLU()
    

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        
        x = self.h1(x)
        x = self.relu(x)

        x = self.h2(x)
        x = self.relu(x)

        x = self.h3(x)
        x = self.relu(x)

        x = self.output_layer(x)
        return x

class PickPlaceDataset(Dataset):
    def __init__(self, *, data_dir: str):
        # self.files = []
        self.obs = []
        self.actions = []

        for file in sorted(Path(data_dir).glob("*.npz")):
            with np.load(file) as data:
                obs = data["obs"]
                actions = data["actions"]

                # print(f"obs.shape: {obs.shape}")
                # print(f"actions.shape: {actions.shape}")

                # validation checks
                if obs.shape[0] != actions.shape[0]:
                    raise ValueError(f"{file} obs/actions length mismatch: {obs.shape[0]} vs {actions.shape[0]}")
                if obs.ndim != 2: 
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
        

def train(*, num_epochs: int=10): 
    train_dataloader = DataLoader(
        dataset=PickPlaceDataset(data_dir="data/demos/2026-07-02_14-04-52"),
        batch_size=32,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loss_fn = nn.MSELoss()
    
    model = MLP(obs_dim=40, action_dim=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # epochs = 10

    writer = SummaryWriter()
   
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (obs, actions) in enumerate(train_dataloader):
            obs = obs.to(device)
            actions = actions.to(device)
            pred = model(obs)
            loss = loss_fn(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # if batch_idx % 20 == 0:
            #     print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item()}")

        avg_loss = epoch_loss / num_batches
        writer.add_scalar("Loss/train", avg_loss, epoch)

    writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=500)
    return parser.parse_args()

def main():

    args = parse_args()

    train(num_epochs=args.num_epochs)

if __name__ == "__main__":
    main()