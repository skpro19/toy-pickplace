from pathlib import Path

import torch

from policy_runtimes.mlp import MLP_ARCHITECTURE, MlpRuntime
from policy_runtimes.types import PolicyRuntime
from policy_runtimes.vision_mlp import VISION_MLP_ARCHITECTURE, VisionMlpRuntime


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
    if architecture == VISION_MLP_ARCHITECTURE:
        return VisionMlpRuntime.from_checkpoint(
            model_path=model_path,
            device=device,
            checkpoint=checkpoint,
        )
    raise ValueError(f"Unsupported policy architecture: {architecture!r}")
