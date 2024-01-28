import torch


def tlwh_centers(tlwhs: torch.Tensor):
    """
    Calculate the centers of bounding boxes given as [x1, y1, width, height].

    Parameters:
    bboxes (Tensor): A tensor of size (N, 4) where N is the number of boxes,
                     and each box is defined as [x1, y1, width, height].

    Returns:
    Tensor: A tensor of size (N, 2) containing [cx, cy] for each bounding box.
    """
    # Unpack the bounding box tensor into its components
    if len(tlwhs) == 0:
        return []

    x1, y1, width, height = tlwhs.unbind(-1)

    # Calculate the centers
    cx = x1 + (width / 2)
    cy = y1 + (height / 2)

    # Stack the centers into a single tensor
    centers = torch.stack((cx, cy), dim=-1)
    return centers


def width(box: torch.Tensor):
    return box[2] - box[0] + 1.0


def height(box: torch.Tensor):
    return box[3] - box[1] + 1.0


def center(box: torch.Tensor):
    return (box[:2] + box[2:]) / 2


def center_batch(boxes: torch.Tensor):
    if len(boxes) == 0:
        return []
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def clamp_value(low, val, high):
    return torch.clamp(val, min=low, max=high)


def clamp_box(box, clamp_box):
    clamped_box = torch.empty_like(box)
    clamped_box[0] = torch.clamp(box[0], min=clamp_box[0], max=clamp_box[2])
    clamped_box[1] = torch.clamp(box[1], min=clamp_box[1], max=clamp_box[3])
    clamped_box[2] = torch.clamp(box[2], min=clamp_box[0], max=clamp_box[2])
    clamped_box[3] = torch.clamp(box[3], min=clamp_box[1], max=clamp_box[3])
    return clamped_box


def make_box_at_center(center_point: torch.Tensor, w: torch.Tensor, h: torch.Tensor):
    box = torch.tensor(
        [
            center_point[0] - (w / 2.0) + 0.5,
            center_point[1] - (h / 2.0) + 0.5,
            center_point[0] + (w / 2.0) - 0.5,
            center_point[1] + (h / 2.0) - 0.5,
        ],
        dtype=torch.float32,
        device=center_point.device,
    )
    # assert np.isclose(width(box), w)
    # assert np.isclose(height(box), h)
    return box


def move_box_to_center(box: torch.Tensor, center_point: torch.Tensor):
    ww = width(box)
    hh = height(box)
    moved_box = make_box_at_center(center_point=center_point, w=ww, h=hh)
    assert ww == width(box)
    assert hh == height(box)
    return moved_box


def scale_box(box: torch.Tensor, scale_width: torch.Tensor, scale_height: torch.Tensor):
    center_point = center(box)
    w = width(box) * scale_width
    h = height(box) * scale_height
    return make_box_at_center(center_point=center_point, w=w, h=h)


def center_distance(box1, box2):
    assert box1 is not None and box2 is not None
    # Calculate the center points of each bounding box
    center_box1 = (box1[:2] + box1[2:]) / 2
    center_box2 = (box2[:2] + box2[2:]) / 2
    distance = torch.norm(center_box1 - center_box2, p=2)
    return distance


def center_x_distance(box1, box2) -> float:
    assert box1 is not None and box2 is not None
    return abs(center(box1)[0] - center(box2)[0])


def aspect_ratio(box):
    return width(box) / height(box)


def tlwh_to_tlbr_multiple(tlwh: torch.Tensor):
    # Calculate bottom-right coordinates
    top_left = tlwh[:, :2]  # Get all rows and first 2 columns (x1, y1)
    sizes = tlwh[:, 2:]  # Get all rows and last 2 columns (width, height)

    bottom_right = top_left + sizes  # Element-wise addition

    # Create the bounding box tensor of [x1, y1, x2, y2]
    bounding_boxes = torch.cat((top_left, bottom_right), dim=1)
    return bounding_boxes


def tlwh_to_tlbr_single(tlwh: torch.Tensor):
    # Calculate bottom-right coordinates
    top_left = tlwh[:2]  # Get all rows and first 2 columns (x1, y1)
    sizes = tlwh[2:]  # Get all rows and last 2 columns (width, height)

    bottom_right = top_left + sizes  # Element-wise addition

    # Create the bounding box tensor of [x1, y1, x2, y2]
    bounding_boxes = torch.cat((top_left, bottom_right), dim=0)
    return bounding_boxes


# def translate_box_to_edge(box: torch.Tensor, bounds: torch.Tensor):
#     """
#     Translate the box to the edge of the bounds if any point is outside.
#     Both box and bounds are in the form [x_min, y_min, x_max, y_max].
#     """
#     # Calculate the out-of-bounds distances
#     delta_left = bounds[0] - box[0]  # How much box is left of the bounds
#     delta_right = box[2] - bounds[2]  # How much box is right of the bounds
#     delta_top = bounds[1] - box[1]    # How much box is above the bounds
#     delta_bottom = box[3] - bounds[3] # How much box is below the bounds

#     # Calculate translation amounts
#     trans_x = max(delta_left, -delta_right, 0)
#     trans_y = max(delta_top, -delta_bottom, 0)

#     # Apply translation
#     box[0] += trans_x
#     box[2] += trans_x
#     box[1] += trans_y
#     box[3] += trans_y

#     return box


def shift_box_to_edge(box, bounding_box):
    """
    If a box is off the edge of the image, translate
    the box to be flush with the edge instead.
    """
    xw = width(bounding_box)
    xh = height(bounding_box)
    was_shifted_x = False
    was_shifted_y = False
    if box[0] < 0:
        box[2] += -box[0]
        box[0] += -box[0]
        was_shifted_x = True
    elif box[2] >= xw:
        offset = box[2] - (xw - 1)
        box[0] -= offset
        box[2] -= offset
        was_shifted_x = True

    if box[1] < 0:
        box[3] += -box[1]
        box[1] += -box[1]
        was_shifted_y = True
    elif box[3] >= xh:
        offset = box[3] - (xh - 1)
        box[1] -= offset
        box[3] -= offset
        was_shifted_y = True
    return box, was_shifted_x, was_shifted_y


def is_box_edge_on_or_outside_other_box_edge(box, bounding_box):
    return torch.tensor(
        [
            box[0] <= bounding_box[0],
            box[1] <= bounding_box[1],
            box[2] >= bounding_box[2],
            box[3] >= bounding_box[3],
        ],
        dtype=torch.bool,
    )


def check_for_box_overshoot(
    box: torch.Tensor,
    bounding_box: torch.Tensor,
    movement_directions: torch.Tensor,
    epsilon: float = 0.01,
):
    """
    Check is a proposed movement direction would push the given box off of the edge of the given boundary box.

    :param box:                     Box which is proposed to be moved [left, top, right, bottomo]
    :param bounding_box:            Box which it is to be inscribed [left, top, right, bottomo]
    :param movement_directions:     Proposed movement direction [dx_axis, dy_axis] (signs only)
    """
    any_on_edge = is_box_edge_on_or_outside_other_box_edge(
        box,
        bounding_box,
    )
    x_on_edge = torch.logical_or(
        torch.logical_and(any_on_edge[0], movement_directions[0] < epsilon),
        torch.logical_and(any_on_edge[2], movement_directions[0] > -epsilon),
    )
    y_on_edge = torch.logical_or(
        torch.logical_and(any_on_edge[1], movement_directions[1] < epsilon),
        torch.logical_and(any_on_edge[3], movement_directions[1] > -epsilon),
    )
    return torch.tensor(
        [x_on_edge, y_on_edge], dtype=x_on_edge.dtype, device=x_on_edge.device
    )


def remove_largest_bbox(batch_bboxes: torch.Tensor, min_boxes: int):
    """
    Remove the bounding box with the largest area from a batch of TLWH bboxes.

    :param batch_bboxes: A tensor of shape (N, 4) where N is the batch size,
                         and each bbox is in TLWH format.
    :param secondary_tensor: optional secondary tensor to also mast
    :return: A tensor of bboxes with one less item, excluding the largest bbox,
             along with the mast and the larges box itself
    """
    # Calculate areas (width * height)

    num_boxes = batch_bboxes.shape[0]
    if num_boxes < min_boxes:
        mask = torch.ones(num_boxes, dtype=torch.bool, device=batch_bboxes.device)
        return batch_bboxes, mask, None

    areas = batch_bboxes[:, 2] * batch_bboxes[:, 3]

    # Find the index of the bbox with the largest area
    largest_bbox_idx = torch.argmax(areas)

    # Remove the largest bbox
    mask = torch.ones(
        batch_bboxes.shape[0], dtype=torch.bool, device=batch_bboxes.device
    )
    mask[largest_bbox_idx] = False
    return batch_bboxes[mask], mask, batch_bboxes[largest_bbox_idx]


def get_enclosing_box(batch_boxes: torch.Tensor):
    """
    Get a bounding box that encompasses all bounding boxes in the batch.

    Parameters:
    - batch_boxes (Tensor): A tensor of shape [N, 4] representing bounding boxes,
                            where each box is [top, left, bottom, right].

    Returns:
    - Tensor: A tensor representing the enclosing bounding box.
    """
    # Separate the coordinates
    tops, lefts, bottoms, rights = (
        batch_boxes[:, 0],
        batch_boxes[:, 1],
        batch_boxes[:, 2],
        batch_boxes[:, 3],
    )

    # Find the minimum top-left and maximum bottom-right coordinates
    min_top, min_left = torch.min(tops), torch.min(lefts)
    max_bottom, max_right = torch.max(bottoms), torch.max(rights)

    # Create the enclosing box
    enclosing_box = torch.tensor([min_top, min_left, max_bottom, max_right])

    return enclosing_box


def scale_box_at_same_center(box: torch.Tensor, scale_factor):
    """
    Scale a bounding box by a given factor while maintaining the same center.

    Parameters:
    - bbox (Tensor): A tensor representing a single bounding box [top, left, bottom, right].
    - scale_factor (float): The scaling factor.

    Returns:
    - Tensor: The scaled bounding box.
    """
    top, left, bottom, right = box

    # Calculate the center of the bounding box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    # Calculate the width and height of the bounding box
    width = right - left
    height = bottom - top

    # Scale the width and height
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Calculate new top-left and bottom-right coordinates
    new_left = center_x - new_width / 2
    new_right = center_x + new_width / 2
    new_top = center_y - new_height / 2
    new_bottom = center_y + new_height / 2

    return torch.tensor([new_top, new_left, new_bottom, new_right])


# def scale_box_to_fit(
#     box: torch.Tensor, max_width: torch.Tensor, max_height: torch.Tensor
# ):
#     box_width = width(box)
#     box_height = height(box)
#     scale_w = None
#     scale_h = None
#     error_count = 0
#     if box_width > max_width:
#         error_count += 1
#         scale_w = box_width / max_width
#     if box_height > max_height:
#         error_count += 1
#         scale_h = box_height / max_height
#     if error_count:
#         if error_count == 2:
#             scale = torch.max(scale_h, scale_w)
#             return box / scale
#         elif scale_w is not None:
#             return box / scale_w
#         else:
#             return box / scale_h
#     return box
