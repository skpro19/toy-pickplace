import numpy as np
import torch


def _print_1d_values(*, values: list[float], label: str = "", precision: int = 4) -> None:
    header = f"{label}:" if label else None
    if header:
        print(header)

    for idx, value in enumerate(values):
        print(f"  [{idx}] = {value:+.{precision}f}")


def print_1d_tensor(*, tensor: torch.Tensor, label: str = "", precision: int = 4) -> None:
    values = tensor.detach().squeeze().cpu()
    if values.ndim != 1:
        raise ValueError(f"expected 1D tensor, got shape {tuple(tensor.shape)}")

    _print_1d_values(values=values.tolist(), label=label, precision=precision)


def print_1d_array(*, array: np.ndarray, label: str = "", precision: int = 4, length: int | None = None) -> None:
    values = np.asarray(array).squeeze()
    if values.ndim != 1:
        raise ValueError(f"expected 1D array, got shape {tuple(values.shape)}")
    if length is not None:
        values = values[:length]

    _print_1d_values(values=values.tolist(), label=label, precision=precision)
