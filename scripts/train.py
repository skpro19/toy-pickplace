import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import re

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
    def __init__(self, *, data_dir: str = "data/demos/2026-07-02_14-04-52"):
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
        

def next_run_name(
    *,
    base_name: str,
    checkpoint_root: str,
    log_root: str,
) -> str:
    existing_indices = []
    pattern = re.compile(rf"^(\d+)_({re.escape(base_name)})$")

    for root in (Path(checkpoint_root), Path(log_root)):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match is not None:
                existing_indices.append(int(match.group(1)))

    next_idx = max(existing_indices, default=0) + 1
    return f"{next_idx:03d}_{base_name}"


def make_run_dirs(
    *,
    base_name: str,
    checkpoint_root: str,
    log_root: str,
) -> tuple[str, Path, Path]:
    while True:
        run_name = next_run_name(
            base_name=base_name,
            checkpoint_root=checkpoint_root,
            log_root=log_root,
        )
        checkpoint_dir = Path(checkpoint_root) / run_name
        log_dir = Path(log_root) / run_name
        if checkpoint_dir.exists() or log_dir.exists():
            continue

        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        return run_name, checkpoint_dir, log_dir


def train(
    *, 
    num_epochs: int=10,
    run_name: str,
    checkpoint_dir: Path,
    log_dir: Path,
    normalize_actions:bool=True) -> None: 
    train_dataloader = DataLoader(
        # dataset=PickPlaceDataset(data_dir="data/demos/2026-07-02_14-04-52"),
        dataset=PickPlaceDataset(data_dir="data/test/"),
        batch_size=32,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

   

    loss_fn = nn.MSELoss()
    
    model = MLP(obs_dim=40, action_dim=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # epochs = 10

    writer = SummaryWriter(log_dir=str(log_dir))
   
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (obs, actions) in enumerate(train_dataloader):
            if batch_idx > 0:
                break
            # print(f"[before] obs.device=>{obs.device} type(obs)=>{type(obs)}")
            obs = obs.to(device)
            # print(f"[after] obs.device=>{obs.device} type(obs)=>{type(obs)}")
            actions = actions.to(device)

            if normalize_actions: 
                print(f"type(actions)=>{type(actions)} actions.shape=>{actions.shape}")
                arm_actions_mean = torch.mean(actions[:, 0:7], dim=0)
                arm_actions_std = torch.std(actions[: , 0:7], dim=0)

                print(f"arm_actions_mean.shape=>{arm_actions_mean.shape} arm_actions_std.shape=>{arm_actions_std.shape}")
                print(f"arm_actions_mean=>{arm_actions_mean}")
            
            # break

            # print(f"batch_idx=>{batch_idx} (obs)=>{type(obs)} obs.shape => {obs.shape}")
            
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

    # checkpointing
    model_path = checkpoint_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training params for simple MLP policy"
    )
    
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--base_name", type=str, default="mlp_action_norm")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--log_root", type=str, default="runs")
    
    return parser.parse_args()

def main():

    args = parse_args()

    run_name, checkpoint_dir, log_dir = make_run_dirs(
        base_name=args.base_name,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
    )

    train(
        num_epochs=args.epochs, 
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

if __name__ == "__main__":
    main()
