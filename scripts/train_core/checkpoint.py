from pathlib import Path

import torch
from torch import nn

from eval import EvalSelectionMode
from scripts.dataset import NormStats


def save_checkpoint(
    *,
    arch: str,
    model: nn.Module,
    model_path: Path,
    epoch_number: int,
    dropout: float,
    normalize: bool,
    action_space: str,
    norm_stats: NormStats,
    optimizer: torch.optim.Optimizer | None = None,
    eval_score: float | None = None,
    eval_metrics: dict[str, float] | None = None,
    eval_metric_version: int | None = None,
    eval_selection_mode: EvalSelectionMode | None = None,
) -> Path:
    checkpoint = {
        "arch": arch,
        "model_dict": model.state_dict(),
        "dropout": dropout,
        "normalize": normalize,
        "action_space": action_space,
        "epoch": epoch_number,
    }
    if optimizer is not None:
        checkpoint["optimizer_dict"] = optimizer.state_dict()
    if eval_score is not None:
        checkpoint["eval_score"] = eval_score
    if eval_metrics is not None:
        checkpoint["eval_metrics"] = eval_metrics
    if eval_metric_version is not None:
        checkpoint["eval_metric_version"] = eval_metric_version
    if eval_selection_mode is not None:
        checkpoint["eval_selection_mode"] = eval_selection_mode
    checkpoint.update(norm_stats)
    torch.save(checkpoint, model_path)
    return model_path
