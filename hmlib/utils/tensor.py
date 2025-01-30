from typing import Union

import torch


def to_tensor_scalar(
    value: Union[int, torch.Tensor],
    device: torch.device,
    dtype=torch.int64,
    non_blocking: bool = False,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if device is not None and value.device != device:
            value = value.to(device=device, non_blocking=non_blocking)

        return value
    value = torch.tensor(value, dtype=dtype)
    if value.device != device:
        value = value.to(device, non_blocking=non_blocking)
    return value
