# Tiling stuff
import torch


def pack_bounding_boxes_as_tiles(
    source_image: torch.Tensor, bounding_boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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

    # Initialize packed image dimensions with a roughly 16:9 aspect ratio
    widths = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    total_area = torch.sum(widths * heights)
    target_height = int(torch.sqrt(total_area / (16 / 9)).item())
    target_width = int(target_height * (16 / 9))

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
    """
    Get batch indices oif bounding boxes that don't overlap with any other bounding boxes
    """
    # boxes should be a tensor of shape [N, 4] representing [x1, y1, x2, y2] for each bounding box
    N: int = boxes.size(0)
    batch_indices: torch.Tensor = torch.arange(N, device=boxes.device)

    # Compute pairwise intersection between boxes
    x1: torch.Tensor = torch.max(boxes[:, None, 0], boxes[:, 0])
    y1: torch.Tensor = torch.max(boxes[:, None, 1], boxes[:, 1])
    x2: torch.Tensor = torch.min(boxes[:, None, 2], boxes[:, 2])
    y2: torch.Tensor = torch.min(boxes[:, None, 3], boxes[:, 3])

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
