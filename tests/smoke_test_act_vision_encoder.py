import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from models.act.vision_encoder import ACTVisionEncoder  # noqa: E402

N_SPATIAL_TOKENS = 64
HIDDEN_DIMS = 128
EMBEDDING_PARAM_COUNT = N_SPATIAL_TOKENS * HIDDEN_DIMS


def make_model(*, device: torch.device) -> ACTVisionEncoder:
    return ACTVisionEncoder(hidden_dims=HIDDEN_DIMS).to(device)


def make_input(*, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 3, 64, 64, device=device)


def assert_forward_shape(*, batch_size: int, device: torch.device) -> None:
    model = make_model(device=device)
    out = model(make_input(batch_size=batch_size, device=device))
    expected = (batch_size, N_SPATIAL_TOKENS, HIDDEN_DIMS)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"


def assert_backward_grads(*, device: torch.device) -> None:
    model = make_model(device=device)
    out = model(make_input(batch_size=2, device=device))
    model.zero_grad()
    out.sum().backward()
    assert model.conv1.weight.grad is not None
    assert model.conv3.weight.grad is not None
    assert model.embedding.weight.grad is not None


def assert_pos_embed_applied(*, device: torch.device) -> None:
    model = make_model(device=device)
    x = make_input(batch_size=1, device=device)
    out = model(x)
    pos = model.embedding(torch.arange(N_SPATIAL_TOKENS, device=device))
    tokens_only = out - pos

    model.embedding.weight.data.zero_()
    out_without_pos = model(x)
    torch.testing.assert_close(out_without_pos, tokens_only)


def assert_input_guard(*, device: torch.device) -> None:
    model = make_model(device=device)
    bad_input = torch.randn(1, 3, 32, 64, device=device)
    try:
        model(bad_input)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid input shape")


def assert_embedding_param_count(*, device: torch.device) -> None:
    model = make_model(device=device)
    assert model.embedding.weight.numel() == EMBEDDING_PARAM_COUNT


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert_forward_shape(batch_size=2, device=device)
    assert_forward_shape(batch_size=1, device=device)
    assert_backward_grads(device=device)
    assert_pos_embed_applied(device=device)
    assert_input_guard(device=device)
    assert_embedding_param_count(device=device)

    print(f"ACTVisionEncoder smoke test passed on {device}.")


if __name__ == "__main__":
    main()
