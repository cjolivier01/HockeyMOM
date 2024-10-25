"""
PyTorch drawing functions
"""

from typing import Tuple, Union

import numpy as np
import torch

from hmlib.utils.box_functions import height, width

from ..utils.image import (
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

    # image = draw_text(image=image, x=100, y=100, text="HELLO WORLD")

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


def draw_horizontal_line(
    image: torch.Tensor,
    start_x: int,
    start_y: int,
    length: int,
    color: Union[Tuple[int, int, int], torch.Tensor],
    thickness: int = 1,
    alpha: int = 255,
) -> torch.Tensor:
    """
    Draw a horizontal line on the image at specified location using PyTorch.
    """
    # Ensure the square doesn't go out of the image boundaries
    assert image.ndim == 4
    assert alpha >= 0 and alpha <= 255

    if alpha == 0:
        # Nothing to do
        return image

    falpha = float(alpha) / 255.0

    if start_y >= thickness // 2:
        start_y -= thickness // 2

    was_channels_last = is_channels_last(image)
    if was_channels_last:
        image = make_channels_first(image)

    if not isinstance(color, torch.Tensor):
        # Convert color tuple to a tensor and reshape to [1, C, 1, 1] for broadcasting
        color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
    else:
        color_tensor = color

    H, W = image_height(image), image_width(image)

    if start_y < 0 or start_y > H:
        return image

    if start_x + length > W:
        length = W - start_x
        if length < 0:  # it's all off the screen?
            assert False  # just check
            return image
        # raise ValueError("Line goes out of image boundaries.")

    if alpha == 255:
        image[:, :, start_y : start_y + thickness, start_x : start_x + length] = color_tensor
    else:
        # Set the pixel values to the specified color  TODO: do all channels a once
        image[:, :, start_y : start_y + thickness, start_x : start_x + length] = (
            image[:, :, :, :, start_y : start_y + thickness, start_x : start_x + length]
            * (1 - alpha)
            + color_tensor * falpha
        )

    if was_channels_last:
        image = make_channels_last(image)
    return image


def draw_vertical_line(
    image: torch.Tensor,
    start_x: int,
    start_y: int,
    length: int,
    color: Union[Tuple[int, int, int], torch.Tensor],
    thickness: int = 1,
    alpha: int = 255,
) -> torch.Tensor:
    """
    Draw a horizontal line on the image at specified location using PyTorch.
    """
    # Ensure the square doesn't go out of the image boundaries
    # assert length > 1
    assert image.ndim == 4
    assert alpha >= 0 and alpha <= 255

    if alpha == 0:
        # Nothing to do
        return image

    falpha = float(alpha) / 255.0

    if start_x >= thickness // 2:
        start_x -= thickness // 2

    was_channels_last = is_channels_last(image)
    if was_channels_last:
        image = make_channels_first(image)

    if not isinstance(color, torch.Tensor):
        # Convert color tuple to a tensor and reshape to [1, C, 1, 1] for broadcasting
        color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
    else:
        color_tensor = color

    H, W = image_height(image), image_width(image)
    if start_x < 0 or start_x > W:
        return image
    if start_y < 0:
        start_y = 0
    if start_y + length > H:
        length = H - start_y
        if length < 0:  # it's all off the screen?
            assert False  # just check
            return image
        # raise ValueError("Line goes out of image boundaries.")

    if alpha == 255:
        image[:, :, start_y : start_y + length, start_x : start_x + thickness] = color_tensor
    else:
        # Set the pixel values to the specified color  TODO: do all channels a once
        image[:, :, start_y : start_y + length, start_x : start_x + thickness] = (
            image[:, :, start_y : start_y + length, start_x : start_x + thickness] * (1 - alpha)
            + color_tensor * falpha
        )

    if was_channels_last:
        image = make_channels_last(image)
    return image


def draw_box(
    image: torch.Tensor,
    tlbr: Union[
        torch.Tensor,
        np.ndarray,
        Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]],
    ],
    color: Union[Tuple[int, int, int], torch.Tensor],
    thickness: int = 1,
    alpha: int = 255,
) -> torch.Tensor:
    W = int(width(tlbr))
    H = int(height(tlbr))
    assert len(tlbr) == 4
    int_box = [int(i) for i in tlbr]

    # left side
    image = draw_vertical_line(
        image=image,
        start_x=int_box[0],
        start_y=int_box[1],
        length=H,
        color=color,
        thickness=thickness,
        alpha=alpha,
    )
    # right side
    image = draw_vertical_line(
        image=image,
        start_x=int_box[2],
        start_y=int_box[1],
        length=H,
        color=color,
        thickness=thickness,
        alpha=alpha,
    )
    # top
    image = draw_horizontal_line(
        image=image,
        start_x=int_box[0],
        start_y=int_box[1],
        length=W,
        color=color,
        thickness=thickness,
        alpha=alpha,
    )
    # bottom
    image = draw_horizontal_line(
        image=image,
        start_x=int_box[0],
        start_y=int_box[3],
        length=W,
        color=color,
        thickness=thickness,
        alpha=alpha,
    )
    return image
