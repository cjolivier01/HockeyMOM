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
    assert alpha >= 0 and alpha <= 255

    if alpha == 0:
        # Nothing to do
        return image

    sq = image.ndim == 3
    if sq:
        image = image.unsqueeze(0)

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
            return image

    if alpha == 255:
        image[:, :, start_y : start_y + thickness, start_x : start_x + length] = color_tensor
    else:
        # Set the pixel values to the specified color  TODO: do all channels a once
        image[:, :, start_y : start_y + thickness, start_x : start_x + length] = (
            image[:, :, start_y : start_y + thickness, start_x : start_x + length] * (1 - alpha)
            + color_tensor * falpha
        )

    if was_channels_last:
        image = make_channels_last(image)

    if sq:
        image = image.squeeze(0)

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
    assert alpha >= 0 and alpha <= 255

    if alpha == 0:
        # Nothing to do
        return image

    sq = image.ndim == 3
    if sq:
        image = image.unsqueeze(0)

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
            return image

    if alpha == 255:
        image[:, :, start_y : start_y + length, start_x : start_x + thickness] = color_tensor
    else:
        # Set the pixel values to the specified color  TODO: do all channels a once
        image[:, :, start_y : start_y + length, start_x : start_x + thickness] = (
            image[:, :, start_y : start_y + length, start_x : start_x + thickness] * (1 - falpha)
            + color_tensor * falpha
        )

    if was_channels_last:
        image = make_channels_last(image)

    if sq:
        image = image.squeeze(0)

    return image


import torch


def draw_line(
    image: torch.Tensor,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
):
    """
    Draws a line on the image tensor from (x1, y1) to (x2, y2) with the given color and thickness.

    Parameters:
    - image: torch.Tensor of shape (C, H, W) on the GPU.
    - x1, y1, x2, y2: Coordinates of the line endpoints.
    - color: torch.Tensor of shape (C,) representing the RGB color.
    - thickness: Line thickness in pixels.
    """

    if y1 == y2:
        # Horizontal
        start_x = min(x1, x2)
        stop_x = max(x1, x2)
        start_y = y1
        length = stop_x - start_x
        return draw_horizontal_line(
            image,
            start_x,
            start_y,
            length,
            color=color,
            thickness=thickness,
        )
    elif x1 == x2:
        # Vertical
        start_y = min(y1, y2)
        stop_y = max(y1, y2)
        start_x = x1
        length = stop_y - start_y
        return draw_vertical_line(
            image,
            start_x,
            start_y,
            length,
            color=color,
            thickness=thickness,
        )

    assert image.ndim == 3
    image = make_channels_first(image)

    # Get image dimensions
    C, H, W = image.shape
    device = image.device

    # Compute the bounding box around the line with thickness
    x_min = int(max(0, min(x1, x2) - thickness))
    x_max = int(min(W - 1, max(x1, x2) + thickness)) + 1
    y_min = int(max(0, min(y1, y2) - thickness))
    y_max = int(min(H - 1, max(y1, y2) + thickness)) + 1

    # Create coordinate grid within the bounding box
    ys, xs = torch.meshgrid(
        torch.arange(y_min, y_max, device=device, dtype=torch.float32),
        torch.arange(x_min, x_max, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Compute line coefficients A, B, and C for the line equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    CL = x2 * y1 - x1 * y2

    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float, device=image.device)

    # Compute the distance from each pixel to the line
    denominator = torch.sqrt(A**2 + B**2) + 1e-8  # Avoid division by zero
    distance = torch.abs(A * xs + B * ys + CL) / denominator

    # Create a mask where the distance is less than or equal to half the thickness
    mask = distance <= (thickness / 2.0)

    # Apply the mask to the image region
    mask = mask.unsqueeze(0).expand(C, -1, -1)  # Expand mask to (C, H', W')

    if not isinstance(color, torch.Tensor):
        # Convert color tuple to a tensor and reshape to [1, C, 1, 1] for broadcasting
        color = torch.tensor(color, dtype=image.dtype, device=image.device)

    color = color.view(C, 1, 1)  # Reshape color to (C, 1, 1)

    # Slice the image to the bounding box
    image_region = image[:, y_min:y_max, x_min:x_max]

    # Clone the image to avoid in-place modifications
    image = image.clone()

    # Modify the region in the image
    image[:, y_min:y_max, x_min:x_max] = torch.where(mask, color, image_region)

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
    filled: bool = False,
) -> torch.Tensor:
    W = int(width(tlbr))
    H = int(height(tlbr))
    assert len(tlbr) == 4
    int_box = [int(i) for i in tlbr]

    if filled:
        intbox = [int(i) for i in tlbr]
        h = intbox[3] - intbox[1]
        w = intbox[2] - intbox[0]
        start_x = intbox[0]
        start_y = intbox[1]
        use_dtype = image.dtype
        image = make_channels_first(image)

        if alpha == 255:
            if not isinstance(color, torch.Tensor):
                # Convert color tuple to a tensor and reshape to [1, C, 1, 1] for broadcasting
                color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(
                    1, -1, 1, 1
                )
            else:
                color_tensor = color

            image[:, :, start_y : start_y + h, start_x : start_x + w] = color
        else:
            if not torch.is_floating_point(image):
                image = image.to(torch.float, non_blocking=True)

            if not isinstance(color, torch.Tensor):
                # Convert color tuple to a tensor and reshape to [1, C, 1, 1] for broadcasting
                color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(
                    1, -1, 1, 1
                )
            else:
                color_tensor = color

            falpha = alpha / 255
            # Set the pixel values to the specified color  TODO: do all channels a once
            image[:, :, start_y : start_y + h, start_x : start_x + w] = (
                image[:, :, start_y : start_y + h, start_x : start_x + w] * (1 - falpha)
                + color_tensor * falpha
            )
        return image

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


def draw_circle(
    image: torch.Tensor,
    center_x: float,
    center_y: float,
    radius: float,
    color: Union[torch.Tensor, Tuple[int, int, int]],
    thickness: int = 1,
    fill: bool = False,
) -> torch.Tensor:
    """
    Draws a circle on the image tensor with the given center, radius, color, and thickness,
    optionally filling it.

    Parameters:
    - image: torch.Tensor of shape (C, H, W) on the GPU or CPU.
    - center_x: float, x-coordinate of the circle center.
    - center_y: float, y-coordinate of the circle center.
    - radius: float, radius of the circle.
    - color: torch.Tensor of shape (C,) representing the RGB color.
    - thickness: int, circle line thickness in pixels. Ignored if fill=True.
    - fill: bool, if True, fills the circle.

    Returns:
    - torch.Tensor: The image tensor with the circle drawn on it.
    """
    # Get image dimensions
    C, H, W = image.shape
    device = image.device

    # TODO: make batchable?
    assert image.ndim == 3
    image = make_channels_first(image)

    # Compute effective thickness
    t: float = 0.0 if fill else float(thickness)
    # Ensure t is non-negative
    t = max(t, 0.0)

    # Determine the bounding box of the circle with thickness
    x_min: int = int(max(0, center_x - radius - t))
    x_max: int = int(min(W - 1, center_x + radius + t))
    y_min: int = int(max(0, center_y - radius - t))
    y_max: int = int(min(H - 1, center_y + radius + t))

    # Adjust x_max and y_max to be inclusive in torch.arange
    x_max += 1
    y_max += 1

    # Create coordinate grid only within the bounding box
    ys: torch.Tensor
    xs: torch.Tensor
    ys, xs = torch.meshgrid(
        torch.arange(y_min, y_max, device=device, dtype=torch.float32),
        torch.arange(x_min, x_max, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Compute squared distance from circle center
    distance_sq: torch.Tensor = (xs - center_x) ** 2 + (ys - center_y) ** 2
    radius_sq: float = radius**2

    if fill:
        # Filled circle
        mask: torch.Tensor = distance_sq <= radius_sq
    else:
        # Circle outline with thickness
        inner_radius_sq: float = max(0.0, (radius - thickness)) ** 2
        outer_radius_sq: float = radius_sq
        mask = (distance_sq >= inner_radius_sq) & (distance_sq <= outer_radius_sq)

    # Apply the mask to the image region
    # Expand mask to (C, H', W')
    mask = mask.unsqueeze(0).expand(C, -1, -1)

    if not isinstance(color, torch.Tensor):
        color = torch.tensor(color, dtype=image.dtype, device=image.device)

    color = color.view(C, 1, 1)

    # Slice the image to the bounding box
    image_region: torch.Tensor = image[:, y_min:y_max, x_min:x_max]

    # Clone the image to avoid in-place modifications
    image = image.clone()

    # Modify the region in the image
    image[:, y_min:y_max, x_min:x_max] = torch.where(mask, color, image_region)

    return image
