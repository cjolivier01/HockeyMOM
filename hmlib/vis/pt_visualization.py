"""PyTorch geometric shape drawing helpers that run efficiently on GPU tensors.

Includes functions for drawing boxes, lines and circles, used heavily by
visualizers and tracking overlays.

@see @ref hmlib.vis.pt_text "pt_text" for text overlays.
"""

import math
from typing import Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.bbox.box_functions import height, width

from ..utils.image import is_channels_last, make_channels_first, make_channels_last


def _prepare_image_for_drawing(image: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
    if image.ndim not in (3, 4):
        raise ValueError("image must be 3D or 4D")

    was_channels_last = is_channels_last(image)
    batched = image.ndim == 4
    if not batched:
        image = image.unsqueeze(0)
    if was_channels_last:
        image = make_channels_first(image)
    return image, was_channels_last, batched


def _finalize_image_after_drawing(
    image: torch.Tensor,
    was_channels_last: bool,
    batched: bool,
) -> torch.Tensor:
    if was_channels_last:
        image = make_channels_last(image)
    if not batched:
        image = image.squeeze(0)
    return image


def _color_to_broadcast_tensor(
    color: Union[Tuple[int, int, int], torch.Tensor], image: torch.Tensor
) -> torch.Tensor:
    channels = image.shape[1]
    target_dtype = image.dtype

    if isinstance(color, torch.Tensor):
        color_tensor = color.to(device=image.device, dtype=target_dtype)
    else:
        color_tensor = torch.tensor(color, device=image.device, dtype=target_dtype)

    if color_tensor.ndim == 1:
        color_tensor = color_tensor.view(1, channels, 1, 1)
    elif color_tensor.ndim == 3:
        color_tensor = color_tensor.unsqueeze(0)
    elif color_tensor.ndim == 4:
        pass
    else:
        raise ValueError("color must be a tuple or tensor with channel dimension")

    if color_tensor.shape[1] != channels:
        raise ValueError("color channel count does not match image")

    return color_tensor


def _apply_solid_fill(
    image: torch.Tensor,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    color_tensor: torch.Tensor,
    alpha: int,
) -> None:
    height = image.shape[2]
    width = image.shape[3]

    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))

    if x0 >= x1 or y0 >= y1:
        return

    region = image[:, :, y0:y1, x0:x1]
    color_expanded = color_tensor.expand_as(region)

    if alpha >= 255:
        region.copy_(color_expanded)
        return

    if alpha <= 0:
        return

    falpha = float(alpha) / 255.0
    if torch.is_floating_point(region):
        region.lerp_(color_expanded, falpha)
    else:
        blended = torch.lerp(
            region.to(torch.float32),
            color_expanded.to(torch.float32),
            falpha,
        )
        region.copy_(blended.to(region.dtype))


def _thickness_offsets(thickness: int, device: torch.device) -> torch.Tensor:
    if thickness <= 1:
        return torch.zeros((1, 2), dtype=torch.int64, device=device)

    offsets = torch.arange(thickness, device=device, dtype=torch.int64) - (thickness // 2)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    return torch.stack((offset_y.reshape(-1), offset_x.reshape(-1)), dim=1)


def draw_filled_square(
    image: torch.Tensor,
    center_x: int,
    center_y: int,
    size,
    color: Union[Tuple[int, int, int], torch.Tensor],
    alpha: int = 255,
) -> torch.Tensor:
    """Draw a filled square centered at ``(center_x, center_y)``."""

    if alpha <= 0:
        return image

    size_int = int(size)
    if size_int <= 0:
        return image

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    color_tensor = _color_to_broadcast_tensor(color, work_image)

    cx = int(center_x)
    cy = int(center_y)
    top_left_x = cx - size_int // 2
    top_left_y = cy - size_int // 2

    width = work_image.shape[3]
    height = work_image.shape[2]
    if (
        top_left_x < 0
        or top_left_y < 0
        or top_left_x + size_int > width
        or top_left_y + size_int > height
    ):
        raise ValueError("Square goes out of image boundaries.")

    _apply_solid_fill(
        work_image,
        top_left_x,
        top_left_x + size_int,
        top_left_y,
        top_left_y + size_int,
        color_tensor,
        alpha,
    )

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


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
    if alpha <= 0 or length <= 0 or thickness <= 0:
        return image

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    color_tensor = _color_to_broadcast_tensor(color, work_image)

    length_int = int(length)
    thickness_int = max(1, int(thickness))
    start_x_int = int(start_x)
    start_y_int = int(start_y)

    top = start_y_int - thickness_int // 2
    bottom = top + thickness_int
    left = start_x_int
    right = start_x_int + length_int

    height = work_image.shape[2]
    width = work_image.shape[3]

    if right <= 0 or left >= width or bottom <= 0 or top >= height:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    _apply_solid_fill(
        work_image,
        left,
        right,
        top,
        bottom,
        color_tensor,
        alpha,
    )

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


def draw_vertical_line(
    image: torch.Tensor,
    start_x: int,
    start_y: int,
    length: int,
    color: Union[Tuple[int, int, int], torch.Tensor],
    thickness: int = 1,
    alpha: int = 255,
) -> torch.Tensor:
    """Draw a vertical line on the image at the specified location."""
    if alpha <= 0 or length <= 0 or thickness <= 0:
        return image

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    color_tensor = _color_to_broadcast_tensor(color, work_image)

    length_int = int(length)
    thickness_int = max(1, int(thickness))
    start_x_int = int(start_x)
    start_y_int = int(start_y)

    left = start_x_int - thickness_int // 2
    right = left + thickness_int
    top = start_y_int
    bottom = start_y_int + length_int

    height = work_image.shape[2]
    width = work_image.shape[3]

    if right <= 0 or left >= width or bottom <= 0 or top >= height:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    _apply_solid_fill(
        work_image,
        left,
        right,
        top,
        bottom,
        color_tensor,
        alpha,
    )

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


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

    if thickness <= 0:
        return image

    if y1 == y2:
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
    if x1 == x2:
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

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    if work_image.shape[0] != 1:
        raise ValueError("draw_line expects an image tensor or a batch of size 1")

    color_tensor = _color_to_broadcast_tensor(color, work_image)

    target = work_image[0]
    _, height, width = target.shape

    thickness_int = max(1, int(thickness))
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        center_x = int(x1) - thickness_int // 2
        center_y = int(y1) - thickness_int // 2
        _apply_solid_fill(
            work_image,
            center_x,
            center_x + thickness_int,
            center_y,
            center_y + thickness_int,
            color_tensor,
            255,
        )
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    device = target.device
    steps = int(max(abs(dx), abs(dy))) + 1
    t = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)

    xs = torch.round(t * dx + float(x1)).to(torch.int64)
    ys = torch.round(t * dy + float(y1)).to(torch.int64)
    points = torch.stack((ys, xs), dim=1)

    offsets = _thickness_offsets(thickness_int, device)
    if offsets.shape[0] > 1:
        points = points.unsqueeze(1) + offsets.unsqueeze(0)
        points = points.view(-1, 2)

    valid = (
        (points[:, 1] >= 0) & (points[:, 1] < width) & (points[:, 0] >= 0) & (points[:, 0] < height)
    )
    points = points[valid]

    if points.numel() == 0:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    points = torch.unique(points, dim=0)
    if points.numel() == 0:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    color_vector = color_tensor.view(color_tensor.shape[1])

    ys_idx = points[:, 0]
    xs_idx = points[:, 1]
    num_points = xs_idx.numel()

    values = color_vector.unsqueeze(1).expand(-1, num_points)
    target[:, ys_idx, xs_idx] = values

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


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
    if alpha <= 0:
        return image

    if len(tlbr) != 4:
        raise ValueError("Bounding box must have four elements")

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    color_tensor = _color_to_broadcast_tensor(color, work_image)

    left, top, right, bottom = (int(i) for i in tlbr)
    box_width = int(width(tlbr))
    box_height = int(height(tlbr))

    if box_width <= 0 or box_height <= 0:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    if filled:
        _apply_solid_fill(
            work_image,
            left,
            right,
            top,
            bottom,
            color_tensor,
            alpha,
        )
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    thickness_int = max(1, int(thickness))

    _apply_solid_fill(
        work_image,
        left,
        left + thickness_int,
        top,
        top + box_height,
        color_tensor,
        alpha,
    )
    _apply_solid_fill(
        work_image,
        right,
        right + thickness_int,
        top,
        top + box_height,
        color_tensor,
        alpha,
    )
    _apply_solid_fill(
        work_image,
        left,
        left + box_width,
        top,
        top + thickness_int,
        color_tensor,
        alpha,
    )
    _apply_solid_fill(
        work_image,
        left,
        left + box_width,
        bottom,
        bottom + thickness_int,
        color_tensor,
        alpha,
    )

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


def draw_circle(
    image: torch.Tensor,
    center_x: float,
    center_y: float,
    radius: float,
    color: Union[torch.Tensor, Tuple[int, int, int]],
    thickness: int = 1,
    fill: bool = False,
) -> torch.Tensor:
    """Draw a circle or disk using vectorized GPU-friendly operations."""

    if radius <= 0:
        return image

    thickness_int = max(1, int(thickness))

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    if work_image.shape[0] != 1:
        raise ValueError("draw_circle expects an image tensor or a batch of size 1")

    radius_float = float(radius)
    effective_thickness = 0.0 if fill else float(thickness_int)

    color_tensor = _color_to_broadcast_tensor(color, work_image)

    target = work_image[0]
    _, height, width = target.shape
    device = target.device

    x_min = math.floor(center_x - radius_float - effective_thickness)
    x_max = math.ceil(center_x + radius_float + effective_thickness) + 1
    y_min = math.floor(center_y - radius_float - effective_thickness)
    y_max = math.ceil(center_y + radius_float + effective_thickness) + 1

    x_min_clamped = max(0, x_min)
    y_min_clamped = max(0, y_min)
    x_max_clamped = min(width, x_max)
    y_max_clamped = min(height, y_max)

    if x_min_clamped >= x_max_clamped or y_min_clamped >= y_max_clamped:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    xs = torch.arange(x_min_clamped, x_max_clamped, device=device, dtype=torch.float32)
    ys = torch.arange(y_min_clamped, y_max_clamped, device=device, dtype=torch.float32)
    ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing="ij")

    distance_sq = (xs_grid - float(center_x)) ** 2 + (ys_grid - float(center_y)) ** 2
    radius_sq = radius_float**2

    if fill:
        mask = distance_sq <= radius_sq
    else:
        inner_radius = max(0.0, radius_float - float(thickness_int))
        inner_radius_sq = inner_radius**2
        mask = (distance_sq <= radius_sq) & (distance_sq >= inner_radius_sq)

    if not mask.any():
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    region = work_image[:, :, y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]
    mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(region)
    color_expanded = color_tensor.expand_as(region)

    region.copy_(torch.where(mask_expanded, color_expanded, region))

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


def draw_ellipse_axes(
    image: torch.Tensor,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    color: Union[torch.Tensor, Tuple[int, int, int]],
    thickness: int = 1,
    fill: bool = False,
) -> torch.Tensor:
    """Draw an ellipse or disk using vectorized GPU-friendly operations."""

    if radius_x <= 0 or radius_y <= 0:
        return image

    thickness_int = max(1, int(thickness))

    work_image, was_channels_last, batched = _prepare_image_for_drawing(image)
    if work_image.shape[0] != 1:
        raise ValueError("draw_ellipse_axes expects an image tensor or a batch of size 1")

    radius_x_float = float(radius_x)
    radius_y_float = float(radius_y)
    effective_thickness = 0.0 if fill else float(thickness_int)

    color_tensor = _color_to_broadcast_tensor(color, work_image)

    target = work_image[0]
    _, height, width = target.shape
    device = target.device

    x_min = math.floor(center_x - radius_x_float - effective_thickness)
    x_max = math.ceil(center_x + radius_x_float + effective_thickness) + 1
    y_min = math.floor(center_y - radius_y_float - effective_thickness)
    y_max = math.ceil(center_y + radius_y_float + effective_thickness) + 1

    x_min_clamped = max(0, x_min)
    y_min_clamped = max(0, y_min)
    x_max_clamped = min(width, x_max)
    y_max_clamped = min(height, y_max)

    if x_min_clamped >= x_max_clamped or y_min_clamped >= y_max_clamped:
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    xs = torch.arange(x_min_clamped, x_max_clamped, device=device, dtype=torch.float32)
    ys = torch.arange(y_min_clamped, y_max_clamped, device=device, dtype=torch.float32)
    ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing="ij")

    norm = ((xs_grid - float(center_x)) / radius_x_float) ** 2 + (
        (ys_grid - float(center_y)) / radius_y_float
    ) ** 2

    if fill:
        mask = norm <= 1.0
    else:
        inner_rx = max(0.0, radius_x_float - float(thickness_int))
        inner_ry = max(0.0, radius_y_float - float(thickness_int))
        if inner_rx <= 0.0 or inner_ry <= 0.0:
            mask = norm <= 1.0
        else:
            inner_norm = ((xs_grid - float(center_x)) / inner_rx) ** 2 + (
                (ys_grid - float(center_y)) / inner_ry
            ) ** 2
            mask = (norm <= 1.0) & (inner_norm >= 1.0)

    if not mask.any():
        return _finalize_image_after_drawing(work_image, was_channels_last, batched)

    region = work_image[:, :, y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]
    mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(region)
    color_expanded = color_tensor.expand_as(region)

    region.copy_(torch.where(mask_expanded, color_expanded, region))

    return _finalize_image_after_drawing(work_image, was_channels_last, batched)


def draw_ellipse(self, frame, bbox, color, track_id=None, team=None):
    y2 = int(bbox[3])
    x_center = (int(bbox[0]) + int(bbox[2])) // 2
    width = int(bbox[2]) - int(bbox[0])
    color = (255, 0, 0)
    text_color = (255, 255, 255)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width) // 2, int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        cv2.rectangle(
            frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED
        )

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10
        font_scale = 0.4
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness=2,
        )

    return frame
