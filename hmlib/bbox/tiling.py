# Tiling stuff
from typing import Tuple

import torch
import torch.nn.functional as F


def _to_float(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor
    return tensor.to(dtype=torch.float, non_blocking=True)


def pack_bounding_boxes_as_tiles(
    source_image: torch.Tensor, bounding_boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assume bounding_boxes is of shape (N, 4), where:
    # N: number of bounding boxes
    # 4: coordinates of the bounding box in (x1, y1, x2, y2) format
    # Assume source_image is of shape (3, H, W), where:
    # 3: number of channels (RGB)
    # H, W: height and width of the source image

    N, _ = bounding_boxes.shape

    # Sort bounding boxes by height (descending) for better packing
    # heights = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    # _, sorted_indices = torch.sort(heights, descending=True)
    # bounding_boxes = bounding_boxes[sorted_indices]

    # Normalize ROI scale for OCR:
    # - Downscale very large crops to limit compute
    # - Upscale very small crops so text recognizers have enough pixels
    max_height_allowed = 256
    min_height_allowed = 128
    new_bounding_boxes = []
    resized_regions = []
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cropped_region = _to_float(source_image[:, y1:y2, x1:x2])
        if h > max_height_allowed:
            scale_factor = max_height_allowed / max(1, h)
            new_w = int(w * scale_factor)
            new_h = max_height_allowed
            cropped_region = F.interpolate(
                cropped_region.unsqueeze(0),
                size=(new_h, new_w),
                mode="area",
            ).squeeze(0)
            x2 = x1 + new_w
            y2 = y1 + new_h
        elif h < min_height_allowed:
            scale_factor = min_height_allowed / max(1, h)
            new_w = int(max(1, w * scale_factor))
            new_h = int(min_height_allowed)
            cropped_region = F.interpolate(
                cropped_region.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            x2 = x1 + new_w
            y2 = y1 + new_h
        new_bounding_boxes.append([x1, y1, x2, y2])
        resized_regions.append(cropped_region)
    bounding_boxes = torch.tensor(new_bounding_boxes, dtype=torch.int)

    # Initialize packed image dimensions to accommodate the largest bounding box
    widths = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    heights = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    max_width = torch.max(widths).item()
    max_height = torch.max(heights).item()
    total_area = torch.sum(widths * heights)
    target_height = max(max_height, int(torch.sqrt(total_area / (16 / 9)).item()))
    target_width = max(max_width, int(target_height * (16 / 9)))

    # Ensure the target width does not exceed the sum of all bounding box widths
    max_total_width = torch.sum(widths).item()
    target_width = min(target_width, max_total_width)

    packed_image = (
        torch.zeros((3, target_height, target_width), dtype=torch.float) + 128
    )  # Assume 3 channels for RGB
    index_map = -torch.ones(
        (target_height, target_width), dtype=torch.long
    )  # Map to store indices of original boxes

    # Start packing from the top-left corner
    current_x, current_y = 0, 0
    max_row_height = 0

    for idx in range(N):
        x1, y1, x2, y2 = bounding_boxes[idx]
        w = x2 - x1
        h = y2 - y1
        cropped_region = resized_regions[idx]

        # Check if the bounding box fits in the current row, otherwise move to next row
        if current_x + w > target_width:
            current_x = 0
            current_y += max_row_height
            max_row_height = 0

        # If the current height exceeds the canvas, resize the canvas vertically
        if current_y + h > target_height:
            new_height = current_y + h
            packed_image = torch.nn.functional.pad(
                packed_image, (0, 0, 0, new_height - target_height)
            )
            index_map = torch.nn.functional.pad(
                index_map, (0, 0, 0, new_height - target_height), value=-1
            )
            target_height = new_height

        # Place the cropped region in the packed image
        packed_image[:, current_y : current_y + h, current_x : current_x + w] = cropped_region
        # Update the index map to indicate which bounding box this region came from
        # index_map[current_y : current_y + h, current_x : current_x + w] = sorted_indices[idx]
        index_map[current_y : current_y + h, current_x : current_x + w] = idx

        # Update current position and max row height
        current_x += w
        max_row_height = max(max_row_height, h)

    return packed_image, index_map


def _pack_bounding_boxes_as_tiles(
    source_image: torch.Tensor, bounding_boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assume bounding_boxes is of shape (N, 4), where:
    # N: number of bounding boxes
    # 4: coordinates of the bounding box in (x1, y1, x2, y2) format
    # Assume source_image is of shape (3, H, W), where:
    # 3: number of channels (RGB)
    # H, W: height and width of the source image

    N, _ = bounding_boxes.shape

    # Sort bounding boxes by height (descending) for better packing
    heights = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    _, sorted_indices = torch.sort(heights, descending=True)
    bounding_boxes = bounding_boxes[sorted_indices]

    # Initialize packed image dimensions to accommodate the largest bounding box
    widths = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    max_width = torch.max(widths).item()
    max_height = torch.max(heights).item()
    total_area = torch.sum(widths * heights)
    target_height = max(max_height, int(torch.sqrt(total_area / (16 / 9)).item()))
    target_width = max(max_width, int(target_height * (16 / 9)))

    # Ensure the target width does not exceed the sum of all bounding box widths
    max_total_width = torch.sum(widths).item()
    target_width = min(target_width, max_total_width)

    packed_image = torch.zeros(
        (3, target_height, target_width), dtype=source_image.dtype
    )  # Assume 3 channels for RGB
    index_map = -torch.ones(
        (target_height, target_width), dtype=torch.long
    )  # Map to store indices of original boxes

    # Start packing from the top-left corner
    current_x, current_y = 0, 0
    max_row_height = 0

    for idx in range(N):
        x1, y1, x2, y2 = bounding_boxes[idx]
        w = x2 - x1
        h = y2 - y1
        cropped_region = source_image[:, y1:y2, x1:x2]

        # Check if the bounding box fits in the current row, otherwise move to next row
        if current_x + w > target_width:
            current_x = 0
            current_y += max_row_height
            max_row_height = 0

        # If the current height exceeds the canvas, resize the canvas vertically
        if current_y + h > target_height:
            new_height = current_y + h
            packed_image = torch.nn.functional.pad(
                packed_image, (0, 0, 0, new_height - target_height)
            )
            index_map = torch.nn.functional.pad(
                index_map, (0, 0, 0, new_height - target_height), value=-1
            )
            target_height = new_height

        # Place the cropped region in the packed image
        packed_image[:, current_y : current_y + h, current_x : current_x + w] = cropped_region
        # Update the index map to indicate which bounding box this region came from
        index_map[current_y : current_y + h, current_x : current_x + w] = sorted_indices[idx]

        # Update current position and max row height
        current_x += w
        max_row_height = max(max_row_height, h)

    return packed_image, index_map


def get_original_bbox_index_from_tiled_image(index_map: torch.Tensor, y: int, x: int) -> int:
    # Given a point (y, x), return the original bounding box index
    return index_map[y, x]


def get_non_overlapping_bbox_indices(boxes: torch.Tensor) -> torch.Tensor:
    # boxes should be a tensor of shape [N, 4] representing [x1, y1, x2, y2] for each bounding box
    # Remove boxes with zero width or height
    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    valid_boxes = boxes[valid_mask]
    N: int = valid_boxes.size(0)
    batch_indices: torch.Tensor = torch.arange(N, device=valid_boxes.device)

    # Compute pairwise intersection between boxes
    x1: torch.Tensor = torch.max(valid_boxes[:, None, 0], valid_boxes[:, 0])
    y1: torch.Tensor = torch.max(valid_boxes[:, None, 1], valid_boxes[:, 1])
    x2: torch.Tensor = torch.min(valid_boxes[:, None, 2], valid_boxes[:, 2])
    y2: torch.Tensor = torch.min(valid_boxes[:, None, 3], valid_boxes[:, 3])

    # Compute intersection width and height
    inter_width: torch.Tensor = (x2 - x1).clamp(min=0)
    inter_height: torch.Tensor = (y2 - y1).clamp(min=0)
    intersection: torch.Tensor = inter_width * inter_height

    # Compute overlap matrix (N x N) and set diagonal to 0 (to ignore self-overlap)
    overlap: torch.Tensor = (intersection > 0).fill_diagonal_(False)

    # Get non-overlapping indices
    non_overlapping: torch.Tensor = overlap.sum(dim=1) == 0
    non_overlapping_indices: torch.Tensor = batch_indices[non_overlapping]

    return non_overlapping_indices


def clamp_boxes_to_image(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    # boxes should be a tensor of shape [N, 4] representing [x1, y1, x2, y2] for each bounding box
    # image_size should be a tensor of shape [2] representing [image_width, image_height]
    image_width, image_height = image_size

    # Clamp box coordinates to be within the image boundaries
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=image_width)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=image_height)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=image_width)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=image_height)

    return boxes
