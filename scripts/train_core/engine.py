from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval import eval_selection_key
from train_core.checkpoint import save_checkpoint
from train_core.dataloader import build_weighted_dataloader, print_dataset_sampling_summary
from train_core.eval_hook import evaluate_checkpoint
from train_core.types import EpochMetrics, TrainConfig, TrainingRecipe


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    recipe: TrainingRecipe,
    device: torch.device,
    joint_loss_fn: nn.Module,
    gripper_loss_fn: nn.Module,
) -> EpochMetrics:
    epoch_loss = 0.0
    epoch_joints_loss = 0.0
    epoch_gripper_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        step = recipe.train_step(
            model=model,
            batch=batch,
            device=device,
            joint_loss_fn=joint_loss_fn,
            gripper_loss_fn=gripper_loss_fn,
        )

        optimizer.zero_grad()
        step["loss"].backward()
        optimizer.step()

        epoch_loss += step["loss"].item()
        epoch_joints_loss += step["joints_loss"]
        epoch_gripper_loss += step["gripper_loss"]
        num_batches += 1

    return {
        "loss": epoch_loss / num_batches,
        "joints_loss": epoch_joints_loss / num_batches,
        "gripper_loss": epoch_gripper_loss / num_batches,
    }


def run_training(
    *,
    config: TrainConfig,
    recipe: TrainingRecipe,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, norm_stats = recipe.build_dataset(
        npz_folders=config["npz_folders"],
        sample_ratios=config["sample_ratios"],
        dagger_intervention_ratio=config["dagger_intervention_ratio"],
        action_space=config["action_space"],
        normalize=config["normalize"],
    )
    dataloader = build_weighted_dataloader(
        dataset=dataset,
        batch_size=config["batch_size"],
        sample_seed=config["sample_seed"],
        dataloader_workers=config["dataloader_workers"],
        persistent_workers=config["persistent_workers"],
        device=device,
    )

    print_dataset_sampling_summary(
        npz_folders=config["npz_folders"],
        dataset=dataset,
    )

    print(f"Using device: {device}")
    print(f"TensorBoard log dir: {config['log_dir']}")
    print(f"Checkpoint dir: {config['checkpoint_dir']}")

    joint_loss_fn = nn.MSELoss()
    gripper_loss_fn = nn.BCEWithLogitsLoss()

    model = recipe.build_model(device=device)
    if config["init_checkpoint"] is not None:
        ckpt = torch.load(config["init_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_dict"])
        print(f"initialized model weights from {config['init_checkpoint']}")

    optimizer = recipe.build_optimizer(model=model)
    if config["init_checkpoint"] is not None and not config["fresh_optimizer_state"]:
        optimizer.load_state_dict(ckpt["optimizer_dict"])
        print(f"restored optimizer state from {config['init_checkpoint']}")

    best_score = float("-inf")
    best_placement_success_rate = float("-inf")
    best_selection_key: tuple[float, ...] | None = None
    best_epoch = 0
    writer = SummaryWriter(log_dir=str(config["log_dir"]))
    try:
        epoch_bar = tqdm(range(config["num_epochs"]), desc="epochs", unit="epoch")
        for epoch in epoch_bar:
            metrics = run_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                recipe=recipe,
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
                arch=recipe.architecture,
                model=model,
                model_path=config["checkpoint_dir"] / "last.pt",
                epoch_number=epoch_number,
                normalize=config["normalize"],
                action_space=config["action_space"],
                norm_stats=norm_stats,
                optimizer=optimizer,
            )
            should_evaluate = (
                epoch_number % config["eval_interval"] == 0
                or epoch_number == config["num_epochs"]
            )
            if should_evaluate:
                eval_result = evaluate_checkpoint(
                    model_path=last_model_path,
                    writer=writer,
                    epoch=epoch,
                    seed=config["eval_seed"],
                    episodes=config["eval_episodes"],
                    max_steps=config["eval_max_steps"],
                    workers=config["eval_workers"],
                    capture_hz=config["eval_capture_hz"],
                )
                mean_score = eval_result["mean_score"]
                placement_success_rate = eval_result["placement_success_rate"]
                selection_key = eval_selection_key(
                    selection_mode=config["eval_selection_mode"],
                    mean_score=mean_score,
                    placement_success_rate=placement_success_rate,
                )
                improved = (
                    best_selection_key is None or selection_key > best_selection_key
                )
                if improved:
                    best_selection_key = selection_key
                    best_score = mean_score
                    best_placement_success_rate = placement_success_rate
                    best_epoch = epoch_number
                    save_checkpoint(
                        arch=recipe.architecture,
                        model=model,
                        model_path=config["checkpoint_dir"] / "best.pt",
                        epoch_number=epoch_number,
                        normalize=config["normalize"],
                        action_space=config["action_space"],
                        norm_stats=norm_stats,
                        optimizer=optimizer,
                        eval_score=mean_score,
                        eval_metric_version=eval_result["eval_metric_version"],
                        eval_selection_mode=config["eval_selection_mode"],
                        eval_metrics={
                            "grasp_rate": eval_result["grasp_rate"],
                            "lift_rate": eval_result["lift_rate"],
                            "tray_reach_rate": eval_result["tray_reach_rate"],
                            "lowered_to_tray_rate": eval_result[
                                "lowered_to_tray_rate"
                            ],
                            "released_over_tray_rate": eval_result[
                                "released_over_tray_rate"
                            ],
                            "placement_success_rate": eval_result[
                                "placement_success_rate"
                            ],
                        },
                    )

                writer.flush()
                epoch_bar.set_postfix(
                    epoch=epoch_number,
                    loss=f'{metrics["loss"]:.4f}',
                    eval_score=f"{mean_score:.4f}",
                )
                print(
                    f"Evaluated {last_model_path} (mean score: {mean_score:.4f}, "
                    f"placement: {placement_success_rate:.4f}, "
                    f"best score: {best_score:.4f}, "
                    f"best placement: {best_placement_success_rate:.4f} "
                    f"at epoch {best_epoch}, mode: {config['eval_selection_mode']})"
                )

                should_stop_early = (
                    config["early_stop_patience"] > 0
                    and epoch_number - best_epoch >= config["early_stop_patience"]
                )
                if should_stop_early:
                    print(
                        f"Stopping early after {config['early_stop_patience']} epochs "
                        "without improvement"
                    )
                    break

    finally:
        writer.close()
    return config["checkpoint_dir"]
