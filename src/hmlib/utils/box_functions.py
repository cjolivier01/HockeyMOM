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


