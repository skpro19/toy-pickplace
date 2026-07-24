from pathlib import Path

import torch

from policy_runtime.mlp import MLP_ARCHITECTURE, MlpRuntime
from policy_runtime.types import PolicyRuntime


def load_runtime(
    *,
    model_path: Path,
    device: torch.device,
) -> PolicyRuntime:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "arch" not in checkpoint:
        raise ValueError(
            f"Checkpoint is missing required field 'arch': {model_path}"
        )
    architecture = checkpoint["arch"]
    if architecture == MLP_ARCHITECTURE:
        return MlpRuntime.from_checkpoint(
            model_path=model_path,
            device=device,
            checkpoint=checkpoint,
        )
    raise ValueError(f"Unsupported policy architecture: {architecture!r}")
