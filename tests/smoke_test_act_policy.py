import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from models.act.policy import ACTPolicy  # noqa: E402

HIDDEN_DIMS = 128
NUM_LAYERS = 2
NUM_ATTENTION_HEADS = 8
CHUNK_SIZE = 10
NUM_IMG_TOKENS = 64
PROPRIO_DIMS = 9
ACTION_DIMS = 8
IMG_H = 64
IMG_W = 64


def make_model(*, device: torch.device) -> ACTPolicy:
    return ACTPolicy(
        hidden_dims=HIDDEN_DIMS,
        imgH=IMG_H,
        imgW=IMG_W,
        obs_dims=PROPRIO_DIMS,
        num_img_tokens=NUM_IMG_TOKENS,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        action_chunk_length=CHUNK_SIZE,
        action_dims=ACTION_DIMS,
    ).to(device)


def make_batch(*, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch_size, PROPRIO_DIMS, device=device),
        torch.randn(batch_size, CHUNK_SIZE, ACTION_DIMS, device=device),
        torch.randn(batch_size, 3, IMG_H, IMG_W, device=device),
        torch.zeros(batch_size, CHUNK_SIZE, dtype=torch.bool, device=device),
    )


def assert_forward_shape(*, batch_size: int, device: torch.device) -> None:
    model = make_model(device=device)
    out = model(make_batch(batch_size=batch_size, device=device))
    expected = (batch_size, CHUNK_SIZE, ACTION_DIMS)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"


def assert_backward_grads(*, device: torch.device) -> None:
    model = make_model(device=device)
    out = model(make_batch(batch_size=2, device=device))
    model.zero_grad()
    out.sum().backward()
    assert model.decoder_queries.grad is not None
    assert model.action_head.weight.grad is not None
    assert model.vision_encoder.conv1.weight.grad is not None


def assert_deterministic(*, device: torch.device) -> None:
    model = make_model(device=device)
    model.eval()
    batch = make_batch(batch_size=1, device=device)
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    torch.testing.assert_close(first, second)


def assert_query_param_shape(*, device: torch.device) -> None:
    model = make_model(device=device)
    expected = (CHUNK_SIZE, HIDDEN_DIMS)
    assert model.decoder_queries.shape == expected, (
        f"expected {expected}, got {tuple(model.decoder_queries.shape)}"
    )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert_forward_shape(batch_size=2, device=device)
    assert_forward_shape(batch_size=1, device=device)
    assert_backward_grads(device=device)
    assert_deterministic(device=device)
    assert_query_param_shape(device=device)

    print(f"ACTPolicy smoke test passed on {device}.")


if __name__ == "__main__":
    main()
