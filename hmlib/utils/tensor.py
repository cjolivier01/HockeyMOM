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
