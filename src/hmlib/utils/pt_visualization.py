"""
PyTorch drawing functions
"""

from typing import Optional, Tuple, Union

import torch

from .image import (
    image_height,
    image_width,
    is_channels_last,
    make_channels_first,
    make_channels_last,
)


def draw_filled_square(
    image: torch.Tensor,
    center_x: int,
    center_y: int,
    size,
    color: Union[Tuple[int, int, int], torch.Tensor],
    alpha: int = 255,
) -> torch.Tensor:
    """
    Draw a square on the image at specified location using PyTorch.

    Parameters:
        image (torch.Tensor): The image tensor of shape [3, H, W]
        top_left_x (int): The x coordinate of the top left corner of the square
        top_left_y (int): The y coordinate of the top left corner of the square
        size (int): The size of the side of the square
        color (tuple): The RGB color of the square
    """
    # Ensure the square doesn't go out of the image boundaries
    assert image.ndim == 4
    assert alpha >= 0 and alpha <= 255

    if alpha == 0:
        # Nothing to do
        return image

    falpha = float(alpha) / 255.0

    was_channels_last = is_channels_last(image)
    if was_channels_last:
        image = make_channels_first(image)

    if not isinstance(color, torch.Tensor):
        # Convert color tuple to a tensor and reshape to [1, C, 1, 1] for broadcasting
        color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)

    top_left_x = int(center_x - size // 2)
    top_left_y = int(center_y - size // 2)
    H, W = image_height(image), image_width(image)
    if top_left_x + size > W or top_left_y + size > H:
        raise ValueError("Square goes out of image boundaries.")

    if alpha == 255:
        image[:, :, top_left_y : top_left_y + size, top_left_x : top_left_x + size] = color_tensor
    else:
        # Set the pixel values to the specified color  TODO: do all channels a once
        image[:, :, top_left_y : top_left_y + size, top_left_x : top_left_x + size] = (
            image[:, :, top_left_y : top_left_y + size, top_left_x : top_left_x + size]
            * (1 - alpha)
            + color_tensor * falpha
        )

    if was_channels_last:
        image = make_channels_last(image)
    return image
