import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from models.act.decoder import ACTDecoder  # noqa: E402
from models.act.encoder import ACTEncoder  # noqa: E402

HIDDEN_DIMS = 128
NUM_LAYERS = 4
NUM_ATTENTION_HEADS = 8
NUM_QUERY_SLOTS = 10
NUM_IMG_TOKENS = 64
NUM_PROPRIO_TOKENS = 1
NUM_MEMORY_TOKENS = NUM_IMG_TOKENS + NUM_PROPRIO_TOKENS


def make_decoder(*, device: torch.device) -> ACTDecoder:
    return ACTDecoder(
        hidden_dims=HIDDEN_DIMS,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_query_slots=NUM_QUERY_SLOTS,
        num_img_tokens=NUM_IMG_TOKENS,
        num_proprio_tokens=NUM_PROPRIO_TOKENS,
    ).to(device)


def make_encoder(*, device: torch.device) -> ACTEncoder:
    return ACTEncoder(
        hidden_dims=HIDDEN_DIMS,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_img_tokens=NUM_IMG_TOKENS,
        num_proprio_tokens=NUM_PROPRIO_TOKENS,
    ).to(device)


def make_tgt(*, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, NUM_QUERY_SLOTS, HIDDEN_DIMS, device=device)


def make_memory(*, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, NUM_MEMORY_TOKENS, HIDDEN_DIMS, device=device)


def assert_forward_shape(*, batch_size: int, device: torch.device) -> None:
    model = make_decoder(device=device)
    out = model(make_tgt(batch_size=batch_size, device=device), make_memory(batch_size=batch_size, device=device))
    expected = (batch_size, NUM_QUERY_SLOTS, HIDDEN_DIMS)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"


def assert_backward_grads(*, device: torch.device) -> None:
    model = make_decoder(device=device)
    out = model(make_tgt(batch_size=2, device=device), make_memory(batch_size=2, device=device))
    model.zero_grad()
    out.sum().backward()
    first_layer = model.act_decoder.layers[0]
    assert first_layer.self_attn.in_proj_weight.grad is not None
    assert first_layer.multihead_attn.in_proj_weight.grad is not None
    assert model.act_decoder.layers[-1].linear2.weight.grad is not None


def assert_deterministic(*, device: torch.device) -> None:
    model = make_decoder(device=device)
    model.eval()
    tgt = make_tgt(batch_size=1, device=device)
    memory = make_memory(batch_size=1, device=device)
    with torch.no_grad():
        first = model(tgt, memory)
        second = model(tgt, memory)
    torch.testing.assert_close(first, second)


def assert_tgt_input_guard(*, device: torch.device) -> None:
    model = make_decoder(device=device)
    memory = make_memory(batch_size=1, device=device)

    bad_rank = torch.randn(NUM_QUERY_SLOTS, HIDDEN_DIMS, device=device)
    try:
        model(bad_rank, memory)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for tgt ndim != 3")

    bad_slot_count = torch.randn(1, NUM_QUERY_SLOTS - 1, HIDDEN_DIMS, device=device)
    try:
        model(bad_slot_count, memory)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong query slot count")

    bad_hidden = torch.randn(1, NUM_QUERY_SLOTS, HIDDEN_DIMS - 1, device=device)
    try:
        model(bad_hidden, memory)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong tgt hidden dim")


def assert_memory_input_guard(*, device: torch.device) -> None:
    model = make_decoder(device=device)
    tgt = make_tgt(batch_size=1, device=device)

    bad_rank = torch.randn(NUM_MEMORY_TOKENS, HIDDEN_DIMS, device=device)
    try:
        model(tgt, bad_rank)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for memory ndim != 3")

    bad_token_count = torch.randn(1, NUM_MEMORY_TOKENS - 1, HIDDEN_DIMS, device=device)
    try:
        model(tgt, bad_token_count)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong memory token count")

    bad_hidden = torch.randn(1, NUM_MEMORY_TOKENS, HIDDEN_DIMS - 1, device=device)
    try:
        model(tgt, bad_hidden)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong memory hidden dim")


def assert_layer_count(*, device: torch.device) -> None:
    model = make_decoder(device=device)
    assert len(model.act_decoder.layers) == NUM_LAYERS


def assert_encoder_decoder_stack(*, device: torch.device) -> None:
    encoder = make_encoder(device=device)
    decoder = make_decoder(device=device)
    tokens = make_memory(batch_size=2, device=device)
    tgt = make_tgt(batch_size=2, device=device)
    memory = encoder(tokens)
    out = decoder(tgt, memory)
    expected = (2, NUM_QUERY_SLOTS, HIDDEN_DIMS)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert_forward_shape(batch_size=2, device=device)
    assert_forward_shape(batch_size=1, device=device)
    assert_backward_grads(device=device)
    assert_deterministic(device=device)
    assert_tgt_input_guard(device=device)
    assert_memory_input_guard(device=device)
    assert_layer_count(device=device)
    assert_encoder_decoder_stack(device=device)

    print(f"ACTDecoder smoke test passed on {device}.")


if __name__ == "__main__":
    main()
