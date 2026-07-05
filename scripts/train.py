import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import re
import os

from mlp import MLP
from dataset import PickPlaceDataset

def next_run_name(
    *,
    base_name: str,
    checkpoint_root: str,
    log_root: str,) -> str:
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
    log_root: str) -> tuple[str, Path, Path]:
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
    npz_folder: str,
    checkpoint_dir: Path,
    log_dir: Path,
    normalize_actions:bool=True) -> None: 
    
    assert os.path.exists(npz_folder), f"npz_folder=>{npz_folder} does not exist"
    assert os.path.isdir(npz_folder), f"npz_folder=>{npz_folder} is not a directory"

    dataset_ = PickPlaceDataset(data_dir=npz_folder)
    train_dataloader = DataLoader(
        dataset=dataset_,
        batch_size=200,
        shuffle=True,
    )

    arm_actions_mean =  None
    arm_actions_std = None
    
    if normalize_actions:
        # print(f"[before normzalization] type(dataset_.targets)=>{type(dataset_.targets)}")
        # print(f"[before normzalization] dataset_.targets.shape=>{dataset_.targets.shape}")
        
        arm_actions_mean = np.mean(dataset_.actions[:, 0:7], axis=0, keepdims=True) # (1, 7)
        arm_actions_std = np.std(dataset_.actions[: , 0:7], axis=0, keepdims=True) # (1, 7)

        # print(f"arm_actions_mean.shape=>{arm_actions_mean.shape} arm_actions_std.shape=>{arm_actions_std.shape}")
        # print(f"arm_actions_mean=>{arm_actions_mean}")
        
        # normalizae 
        dataset_.targets[:, 0:7] = (dataset_.actions[:, 0:7] - arm_actions_mean) / (arm_actions_std + 1e-6)

        # print(f"actions.shape=>{actions.shape}")
        dataset_.targets[:,7] = dataset_.actions[:,7] / 255.0

        # print(f"[after normzalization] type(dataset_.actions)=>{type(dataset_.actions)}")
        # print(f"[after normzalization] dataset_.actions.shape=>{dataset_.actions.shape}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

   

    loss_fn = nn.MSELoss()
    
    model = MLP(obs_dim=45, action_dim=8).to(device)

    print(f"model created!")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # epochs = 10

    writer = SummaryWriter(log_dir=str(log_dir))
   
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (obs, targets) in enumerate(train_dataloader):
            # print(f"--------------------------------")
            # print(f"[batch_idx]=>{batch_idx}")
            # print(f"--------------------------------")

            obs = obs.to(device)
            targets = targets.to(device)
  
            pred = model(obs)
            loss = loss_fn(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
    
    writer.close()

    # checkpointing
    # model_path = checkpoint_dir / "model.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved model: {model_path}")
    checkpoint = {
        "model_dict" : model.state_dict(), 
        "normalize_actions": normalize_actions, 
        "arm_actions_mean" : arm_actions_mean, 
        "arm_actions_std" : arm_actions_std
    }
    
    model_path = checkpoint_dir / "model.pt"
    torch.save(checkpoint, model_path)
    print(f"Saved model: {model_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training params for simple MLP policy"
    )
    
    parser.add_argument("--epochs", type=int, default=1)

    # checkpoint and runs folder are created at `checkpoints/<idx>_<base_name>` 
    # and `runs/<idx>_<base_name>` respectively
    parser.add_argument("--base_name", type=str, default="mlp_action_norm")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--log_root", type=str, default="runs")
    parser.add_argument("--npz", type=str, required=True, help="npz folder path")
    return parser.parse_args()

def main():

    args = parse_args()

    # checkpoint and runs folder are created at `checkpoints/idx_base_name 
    # and `runs/idx_base_name respectively
    # idx is auto incremented
    run_name, checkpoint_dir, log_dir = make_run_dirs(
        base_name=args.base_name,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
    )

    train(
        num_epochs=args.epochs, 
        run_name=run_name,
        npz_folder=args.npz,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

if __name__ == "__main__":
    main()
