import torch

from pt_autograph import pt_function


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


def center_distance(box1, box2):
    assert box1 is not None and box2 is not None
    # Calculate the center points of each bounding box
    center_box1 = (box1[:2] + box1[2:]) / 2
    center_box2 = (box2[:2] + box2[2:]) / 2
    distance = torch.norm(center_box1 - center_box2, p=2)
    return distance
    # c1 = center(box1)
    # c2 = center(box2)
    # if c1 == c2:
    #     return 0.0
    # w = c2[0] - c1[0]
    # h = c2[1] - c1[1]
    # return math.sqrt(w * w + h * h)


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
    if box[0] < 0:
        box[2] += -box[0]
        box[0] += -box[0]
    elif box[2] >= xw:
        offset = box[2] - (xw - 1)
        box[0] -= offset
        box[2] -= offset

    if box[1] < 0:
        box[3] += -box[1]
        box[1] += -box[1]
    elif box[3] >= xh:
        offset = box[3] - (xh - 1)
        box[1] -= offset
        box[3] -= offset
    return box
