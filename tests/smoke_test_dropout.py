import sys
from pathlib import Path
from typing import Callable

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from constant import PROPRIO_DIMS  # noqa: E402
from models.vision_mlp import VisionMLP  # noqa: E402

DROPOUT_LAYER_INDICES = (2, 5, 8)
BATCH_SIZE = 128
STEPS = 150
TOLERANCE = 0.02


def measure_zero_fractions(
    *,
    model: VisionMLP,
    device: torch.device,
) -> list[float]:
    dropped_counts = [0] * len(DROPOUT_LAYER_INDICES)
    candidate_counts = [0] * len(DROPOUT_LAYER_INDICES)

    def make_hook(position: int) -> Callable:
        def hook(module, args, output) -> None:
            inputs = args[0]
            dropped = ((inputs != 0) & (output == 0)).sum().item()
            candidates = (inputs != 0).sum().item()
            dropped_counts[position] += dropped
            candidate_counts[position] += candidates

        return hook

    handles = [
        model.backbone[index].register_forward_hook(make_hook(position))
        for position, index in enumerate(DROPOUT_LAYER_INDICES)
    ]
    try:
        with torch.no_grad():
            for _ in range(STEPS):
                proprio_obs = torch.randn(
                    BATCH_SIZE,
                    PROPRIO_DIMS,
                    device=device,
                )
                img_obs = torch.rand(BATCH_SIZE, 3, 64, 64, device=device)
                model(proprio_obs, img_obs)
    finally:
        for handle in handles:
            handle.remove()

    return [
        dropped_count / candidate_count
        for dropped_count, candidate_count in zip(dropped_counts, candidate_counts)
    ]


def assert_dropout_fraction(*, dropout: float, device: torch.device) -> None:
    model = VisionMLP(dropout=dropout).to(device)
    model.train()
    observed = measure_zero_fractions(model=model, device=device)
    for index, fraction in enumerate(observed):
        assert abs(fraction - dropout) <= TOLERANCE, (
            f"dropout={dropout}: layer {index} zeroed {fraction:.4f} "
            f"of activations, expected {dropout}"
        )
    formatted = ", ".join(f"{fraction:.4f}" for fraction in observed)
    print(f"dropout={dropout}: observed zero fractions = [{formatted}]")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dropout in (0.0, 0.3, 0.6):
        assert_dropout_fraction(dropout=dropout, device=device)
    print("Dropout smoke test passed.")


if __name__ == "__main__":
    main()
