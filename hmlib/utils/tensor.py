from typing import Sequence

import torch


def make_const_tensor(
    value: float | int,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...] = (),
) -> torch.Tensor:
    """Utility to create a constant tensor."""
    if isinstance(value, float):
        tensor = torch.full(shape, value, dtype=dtype, device=device)
    else:
        tensor = torch.full(shape, value, dtype=dtype, device=device)
    return tensor


def new_full(tensor: torch.Tensor, shape: Sequence[int], value: float) -> torch.Tensor:
    """Create a full tensor like another tensor but with a different shape."""
    return make_const_tensor(
        value,
        shape=shape,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def new_zeros(tensor: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Create a zeros tensor like another tensor but with a different shape."""
    return torch.zeros(
        shape=shape,
        device=tensor.device,
        dtype=tensor.dtype,
    )
