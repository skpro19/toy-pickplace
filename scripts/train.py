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

from constant import (
    OBS_DIMS, 
    ACTION_DIMS, 
    EPSILON,
    JOINTS_LOSS_WEIGHT,
    GRIPPER_LOSS_WEIGHT
    )

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
    npz_folder: str,
    num_epochs: int,
    # action_space: str,
    checkpoint_root: str,
    log_root: str) -> tuple[str, Path, Path]:
    n_episodes = len(list(Path(npz_folder).glob("*.npz")))
    # base_name = f"{base_name}_{action_space}_eps{n_episodes}_epochs{num_epochs}"
    base_name = f"{base_name}_eps{n_episodes}_epochs{num_epochs}"
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
    normalize: bool=True,
    action_space: str="joint_delta") -> None: 
    
    assert os.path.exists(npz_folder), f"npz_folder=>{npz_folder} does not exist"
    assert os.path.isdir(npz_folder), f"npz_folder=>{npz_folder} is not a directory"

    dataset_ = PickPlaceDataset(data_dir=npz_folder)
    train_dataloader = DataLoader(
        dataset=dataset_,
        batch_size=200,
        shuffle=True,
    )

    # normalization stats for checkpointing
    arm_actions_mean =  None
    arm_actions_std = None
    arm_obs_mean = None
    arm_obs_std = None

    # [action_targets]
    if action_space == "joint_delta":
        arm_actions = dataset_.actions[:, 0:ACTION_DIMS-1]
        arm_qpos    = dataset_.obs[:, 0:ACTION_DIMS-1]
        dataset_.action_targets[:, 0:ACTION_DIMS-1] = arm_actions - arm_qpos
        dataset_.action_targets[:, ACTION_DIMS-1] = dataset_.actions[:, ACTION_DIMS-1]/255.0
    elif action_space == "absolute":
        dataset_.action_targets[:, 0:ACTION_DIMS-1] = dataset_.actions[:, 0:ACTION_DIMS-1]
        dataset_.action_targets[:, ACTION_DIMS-1] = dataset_.actions[:, ACTION_DIMS-1]/255.0
    else:
        raise ValueError(f"Unknown action_space: {action_space}")
    
    # [obs targets]
    dataset_.obs_targets = dataset_.obs


    # [action/obs mean computation]
    arm_actions_mean = np.mean(dataset_.action_targets[:, 0:ACTION_DIMS-1], axis=0, keepdims=True)
    arm_actions_std = np.std(dataset_.action_targets[:, 0:ACTION_DIMS-1], axis=0, keepdims=True)

    arm_obs_mean = np.mean(dataset_.obs_targets, axis=0, keepdims=True)
    arm_obs_std = np.std(dataset_.obs_targets, axis=0, keepdims=True) # (1,45)
    
    if normalize:
        dataset_.action_targets[:, 0:ACTION_DIMS-1] -= arm_actions_mean
        dataset_.action_targets[:, 0:ACTION_DIMS-1] /= (arm_actions_std + EPSILON)

        dataset_.obs_targets -= arm_obs_mean
        dataset_.obs_targets /= (arm_obs_std + EPSILON)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

   

    # loss_fn = nn.MSELoss()
    joint_loss_fn = nn.MSELoss()
    gripper_loss_fn = nn.BCEWithLogitsLoss()

    model = MLP(obs_dim=OBS_DIMS, action_dim=ACTION_DIMS).to(device)

    print(f"model created!")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # epochs = 10

    writer = SummaryWriter(log_dir=str(log_dir))
   
    epoch_bar = tqdm(range(num_epochs), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        epoch_joints_loss = 0.0
        epoch_gripper_loss = 0.0
        num_batches = 0

        for batch_idx, (obs_target, action_target) in enumerate(train_dataloader):
            # print(f"BATCH_IDX=>{batch_idx}")
            # continue 
        
            obs_target = obs_target.to(device)
            action_target = action_target.to(device)

            # print(f"--------------------------------")
            # print(f"obs_target.shape=>{obs_target.shape}")
            # print(f"action_target.shape=>{action_target.shape}")
            # print(f"--------------------------------")

            # break  
            joints_target = action_target[:, :ACTION_DIMS-1]
            gripper_target = action_target[:, ACTION_DIMS-1].unsqueeze(1)

            # print(f"action_target.shape=>{action_target.shape}")
            # print(f"joints_target.shape=>{joints_target.shape}")
            # print(f"gripper_target.shape=>{gripper_target.shape}")

            # break
            # loss = loss_fn(pred, action_target)``

            # print(f"joints_target.shape=>{joints_target.shape} joints_pred.shape=>{joints_pred.shape}") 
            # print(f"gripper_target.shape=>{gripper_target.shape} gripper_pred.shape=>{gripper_pred.shape}")

            (joints_pred, gripper_pred) = model(obs_target)
                
            joints_loss = joint_loss_fn(joints_pred, joints_target)
            gripper_loss = gripper_loss_fn(gripper_pred, gripper_target)
            loss = JOINTS_LOSS_WEIGHT * joints_loss + GRIPPER_LOSS_WEIGHT * gripper_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_joints_loss += joints_loss.item()
            epoch_gripper_loss += gripper_loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        avg_joints_loss = epoch_joints_loss / num_batches
        avg_gripper_loss = epoch_gripper_loss / num_batches
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/joints", avg_joints_loss, epoch)
        writer.add_scalar("Loss/gripper", avg_gripper_loss, epoch)
        epoch_bar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.4f}")
        
    
    writer.close()

    # checkpointing
    # model_path = checkpoint_dir / "model.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved model: {model_path}")
    checkpoint = {
        "model_dict" : model.state_dict(), 
        "normalize": normalize,
        "action_space": action_space,
        "arm_actions_mean" : arm_actions_mean, 
        "arm_actions_std" : arm_actions_std, 
        "arm_obs_mean": arm_obs_mean, 
        "arm_obs_std": arm_obs_std
    }
    
    model_path = checkpoint_dir / "model.pt"
    torch.save(checkpoint, model_path)
    print(f"Saved model: {model_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training params for simple MLP policy"
    )
    
    
    # checkpoint and runs folder are created at `checkpoints/<idx>_<base_name>` 
    # and `runs/<idx>_<base_name>` respectively
    parser.add_argument("--base", type=str, default="action-delta")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_root", type=str, default="runs")
    parser.add_argument("--npz", type=str, required=True, help="npz folder path")
    parser.add_argument("--action_space", type=str, default="joint_delta", 
                        choices=["joint_delta", "absolute"])
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()

def main():

    args = parse_args()

    run_name, checkpoint_dir, log_dir = make_run_dirs(
        base_name=args.base,
        npz_folder=args.npz,
        num_epochs=args.epochs,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
        # action_space=args.action_space,
    )

    train(
        num_epochs=args.epochs, 
        run_name=run_name,
        npz_folder=args.npz,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        action_space=args.action_space,
        normalize=args.normalize,
    )

if __name__ == "__main__":
    main()
