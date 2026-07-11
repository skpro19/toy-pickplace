import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import TypedDict
import numpy as np
from tqdm import tqdm
import argparse
import re

from mlp import MLP
from dataset import PickPlaceDataset
from eval import score_ckpt

from constant import (
    OBS_DIMS, 
    ACTION_DIMS, 
    EPSILON,
    JOINTS_LOSS_WEIGHT,
    GRIPPER_LOSS_WEIGHT
    )


class NormStats(TypedDict):
    arm_actions_mean: np.ndarray
    arm_actions_std: np.ndarray
    arm_obs_mean: np.ndarray
    arm_obs_std: np.ndarray


class EpochMetrics(TypedDict):
    loss: float
    joints_loss: float
    gripper_loss: float


def next_run_name(
    *,
    base_name: str,
    checkpoint_root: Path,
    log_root: Path,) -> str:
    existing_indices = []
    pattern = re.compile(rf"^(\d+)_({re.escape(base_name)})$")

    for root in (checkpoint_root, log_root):
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
    npz_folders: list[Path],
    num_epochs: int,
    checkpoint_root: Path,
    log_root: Path) -> tuple[str, Path, Path]:
    n_episodes = sum(len(list(Path(d).glob("*.npz"))) for d in npz_folders)
    base_name = f"{base_name}_eps{n_episodes}_epochs{num_epochs}"
    while True:
        run_name = next_run_name(
            base_name=base_name,
            checkpoint_root=checkpoint_root,
            log_root=log_root,
        )
        checkpoint_dir = checkpoint_root / run_name
        log_dir = log_root / run_name
        if checkpoint_dir.exists() or log_dir.exists():
            continue

        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        return run_name, checkpoint_dir, log_dir


def save_checkpoint(
    *,
    model: nn.Module,
    model_path: Path,
    epoch_number: int,
    normalize: bool,
    action_space: str,
    norm_stats: NormStats,
    eval_score: float | None = None,
) -> Path:
    checkpoint = {
        "model_dict": model.state_dict(),
        "normalize": normalize,
        "action_space": action_space,
        "epoch": epoch_number,
    }
    if eval_score is not None:
        checkpoint["eval_score"] = eval_score
    checkpoint.update(norm_stats)
    torch.save(checkpoint, model_path)
    return model_path


def evaluate_checkpoint(
    *,
    model_path: Path,
    writer: SummaryWriter,
    epoch: int,
    seed: int,
    episodes: int,
    max_steps: int,
) -> float:
    score_dict = score_ckpt(
        ckpt_path=str(model_path),
        seed=seed,
        max_steps=max_steps,
        episodes=episodes,
    )
    mean_score = float(score_dict["mean_score"])
    writer.add_scalar("Eval/mean_score", mean_score, epoch)
    writer.flush()
    return mean_score


def prepare_dataset(
    *,
    npz_folders: list[Path],
    sample_ratios: list[float] | None,
    sample_seed: int,
    action_space: str,
    normalize: bool,
) -> tuple[PickPlaceDataset, NormStats]:
    dataset = PickPlaceDataset(
        data_dirs=npz_folders,
        sample_ratios=sample_ratios,
        seed=sample_seed,
    )

    if action_space == "joint_delta":
        arm_actions = dataset.actions[:, 0:ACTION_DIMS-1]
        arm_qpos = dataset.obs[:, 0:ACTION_DIMS-1]
        dataset.action_targets[:, 0:ACTION_DIMS-1] = arm_actions - arm_qpos
        dataset.action_targets[:, ACTION_DIMS-1] = (
            dataset.actions[:, ACTION_DIMS-1] / 255.0
        )
    elif action_space == "absolute":
        dataset.action_targets[:, 0:ACTION_DIMS-1] = dataset.actions[
            :, 0:ACTION_DIMS-1
        ]
        dataset.action_targets[:, ACTION_DIMS-1] = (
            dataset.actions[:, ACTION_DIMS-1] / 255.0
        )
    else:
        raise ValueError(f"Unknown action_space: {action_space}")

    dataset.obs_targets = dataset.obs.copy()

    norm_stats: NormStats = {
        "arm_actions_mean": np.mean(
            dataset.action_targets[:, 0:ACTION_DIMS-1], axis=0, keepdims=True
        ),
        "arm_actions_std": np.std(
            dataset.action_targets[:, 0:ACTION_DIMS-1], axis=0, keepdims=True
        ),
        "arm_obs_mean": np.mean(dataset.obs_targets, axis=0, keepdims=True),
        "arm_obs_std": np.std(dataset.obs_targets, axis=0, keepdims=True),
    }

    if normalize:
        dataset.action_targets[:, 0:ACTION_DIMS-1] -= norm_stats["arm_actions_mean"]
        dataset.action_targets[:, 0:ACTION_DIMS-1] /= (
            norm_stats["arm_actions_std"] + EPSILON
        )
        dataset.obs_targets -= norm_stats["arm_obs_mean"]
        dataset.obs_targets /= norm_stats["arm_obs_std"] + EPSILON

    return dataset, norm_stats


def train_epoch(
    *,
    model: MLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    joint_loss_fn: nn.Module,
    gripper_loss_fn: nn.Module,
) -> EpochMetrics:
    epoch_loss = 0.0
    epoch_joints_loss = 0.0
    epoch_gripper_loss = 0.0
    num_batches = 0

    for obs_target, action_target in dataloader:
        obs_target = obs_target.to(device)
        action_target = action_target.to(device)

        joints_target = action_target[:, :ACTION_DIMS-1]
        gripper_target = action_target[:, ACTION_DIMS-1].unsqueeze(1)

        joints_pred, gripper_pred = model(obs_target)

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

    return {
        "loss": epoch_loss / num_batches,
        "joints_loss": epoch_joints_loss / num_batches,
        "gripper_loss": epoch_gripper_loss / num_batches,
    }


def train(
    *,
    num_epochs: int=10,
    run_name: str,
    npz_folders: list[Path],
    checkpoint_dir: Path,
    log_dir: Path,
    normalize: bool=True,
    action_space: str="joint_delta",
    sample_ratios: list[float] | None = None,
    sample_seed: int = 0,
    eval_interval: int = 1,
    eval_seed: int = 0,
    eval_episodes: int = 100,
    eval_max_steps: int = 1400,
    early_stop_patience: int = 50,
) -> None:

    dataset, norm_stats = prepare_dataset(
        npz_folders=npz_folders,
        sample_ratios=sample_ratios,
        sample_seed=sample_seed,
        action_space=action_space,
        normalize=normalize,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

   

    joint_loss_fn = nn.MSELoss()
    gripper_loss_fn = nn.BCEWithLogitsLoss()

    model = MLP(obs_dim=OBS_DIMS, action_dim=ACTION_DIMS).to(device)

    print(f"model created!")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_score = float("-inf")
    best_epoch = 0
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        epoch_bar = tqdm(range(num_epochs), desc="epochs", unit="epoch")
        for epoch in epoch_bar:
            metrics = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                joint_loss_fn=joint_loss_fn,
                gripper_loss_fn=gripper_loss_fn,
            )
            writer.add_scalar("Loss/train", metrics["loss"], epoch)
            writer.add_scalar("Loss/joints", metrics["joints_loss"], epoch)
            writer.add_scalar("Loss/gripper", metrics["gripper_loss"], epoch)
            epoch_bar.set_postfix(epoch=epoch + 1, loss=f'{metrics["loss"]:.4f}')

            epoch_number = epoch + 1
            last_model_path = save_checkpoint(
                model=model,
                model_path=checkpoint_dir / "last.pt",
                epoch_number=epoch_number,
                normalize=normalize,
                action_space=action_space,
                norm_stats=norm_stats,
            )
            should_evaluate = epoch_number % eval_interval == 0 or epoch_number == num_epochs
            if should_evaluate:
                mean_score = evaluate_checkpoint(
                    model_path=last_model_path,
                    writer=writer,
                    epoch=epoch,
                    seed=eval_seed,
                    episodes=eval_episodes,
                    max_steps=eval_max_steps,
                )
                improved = mean_score > best_score
                if improved:
                    best_score = mean_score
                    best_epoch = epoch_number
                    save_checkpoint(
                        model=model,
                        model_path=checkpoint_dir / "best.pt",
                        epoch_number=epoch_number,
                        normalize=normalize,
                        action_space=action_space,
                        norm_stats=norm_stats,
                        eval_score=mean_score,
                    )

                writer.flush()
                epoch_bar.set_postfix(
                    epoch=epoch_number,
                    loss=f'{metrics["loss"]:.4f}',
                    eval_score=f"{mean_score:.4f}",
                )
                print(
                    f"Evaluated {last_model_path} (mean score: {mean_score:.4f}, "
                    f"best score: {best_score:.4f} at epoch {best_epoch})"
                )

                should_stop_early = (
                    early_stop_patience > 0
                    and epoch_number - best_epoch >= early_stop_patience
                )
                if should_stop_early:
                    print(
                        f"Stopping early after {early_stop_patience} epochs "
                        "without improvement"
                    )
                    break

    finally:
        writer.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training params for simple MLP policy"
    )
    
    
    parser.add_argument("--base", type=str, default="action-delta")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_root", type=Path, default=Path("runs"))
    parser.add_argument("--npz", type=Path, nargs="+", required=True, help="npz folder path(s)")
    parser.add_argument("--action_space", type=str, default="joint_delta", 
                        choices=["joint_delta", "absolute"])
    parser.add_argument(
        "--sample-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Optional per --npz directory timestep sampling ratios, e.g. 0.8 0.2",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used when --sample-ratios is set",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Save and evaluate every N epochs; the final epoch is always evaluated",
    )
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=1400)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop after this many epochs without eval improvement; 0 disables",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be at least 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.eval_max_steps < 1:
        parser.error("--eval-max-steps must be at least 1")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if args.sample_ratios is not None and len(args.sample_ratios) != len(args.npz):
        parser.error(
            f"--sample-ratios length ({len(args.sample_ratios)}) "
            f"must match --npz length ({len(args.npz)})"
        )
    for npz_folder in args.npz:
        if not npz_folder.is_dir():
            parser.error(f"--npz path is not a directory: {npz_folder}")

    return args

def main():

    args = parse_args()

    run_name, checkpoint_dir, log_dir = make_run_dirs(
        base_name=args.base,
        npz_folders=args.npz,
        num_epochs=args.epochs,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
    )

    train(
        num_epochs=args.epochs, 
        run_name=run_name,
        npz_folders=args.npz,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        action_space=args.action_space,
        normalize=args.normalize,
        sample_ratios=args.sample_ratios,
        sample_seed=args.sample_seed,
        eval_interval=args.eval_interval,
        eval_seed=args.eval_seed,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        early_stop_patience=args.early_stop_patience,
    )

if __name__ == "__main__":
    main()
