import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from models.act.encoder import ACTEncoder  # noqa: E402

HIDDEN_DIMS = 128
NUM_LAYERS = 4
NUM_ATTENTION_HEADS = 8
NUM_IMG_TOKENS = 64
NUM_PROPRIO_TOKENS = 1
NUM_INPUT_TOKENS = NUM_IMG_TOKENS + NUM_PROPRIO_TOKENS


def make_model(*, device: torch.device) -> ACTEncoder:
    return ACTEncoder(
        hidden_dims=HIDDEN_DIMS,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_img_tokens=NUM_IMG_TOKENS,
        num_proprio_tokens=NUM_PROPRIO_TOKENS,
    ).to(device)


def make_input(*, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, NUM_INPUT_TOKENS, HIDDEN_DIMS, device=device)


def assert_forward_shape(*, batch_size: int, device: torch.device) -> None:
    model = make_model(device=device)
    out = model(make_input(batch_size=batch_size, device=device))
    expected = (batch_size, NUM_INPUT_TOKENS, HIDDEN_DIMS)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"


def assert_backward_grads(*, device: torch.device) -> None:
    model = make_model(device=device)
    out = model(make_input(batch_size=2, device=device))
    model.zero_grad()
    out.sum().backward()
    first_layer = model.act_encoder.layers[0]
    assert first_layer.self_attn.in_proj_weight.grad is not None
    assert model.act_encoder.layers[-1].linear2.weight.grad is not None


def assert_deterministic(*, device: torch.device) -> None:
    model = make_model(device=device)
    model.eval()
    x = make_input(batch_size=1, device=device)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    torch.testing.assert_close(first, second)


def assert_input_guard(*, device: torch.device) -> None:
    model = make_model(device=device)

    bad_rank = torch.randn(NUM_INPUT_TOKENS, HIDDEN_DIMS, device=device)
    try:
        model(bad_rank)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for ndim != 3")

    bad_token_count = torch.randn(1, NUM_INPUT_TOKENS - 1, HIDDEN_DIMS, device=device)
    try:
        model(bad_token_count)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong token count")

    bad_hidden = torch.randn(1, NUM_INPUT_TOKENS, HIDDEN_DIMS - 1, device=device)
    try:
        model(bad_hidden)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong hidden dim")


def assert_layer_count(*, device: torch.device) -> None:
    model = make_model(device=device)
    assert len(model.act_encoder.layers) == NUM_LAYERS


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert_forward_shape(batch_size=2, device=device)
    assert_forward_shape(batch_size=1, device=device)
    assert_backward_grads(device=device)
    assert_deterministic(device=device)
    assert_input_guard(device=device)
    assert_layer_count(device=device)

    print(f"ACTEncoder smoke test passed on {device}.")


if __name__ == "__main__":
    main()
