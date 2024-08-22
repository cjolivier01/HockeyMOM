import torch
import torch.nn.functional as F
from hmlib.utils.image import (
    image_width,
    image_height,
    make_channels_first,
    make_channels_last,
)


import torch
import torch.nn.functional as F


def warp_affine_pytorch(image, transform_matrix, output_size):
    """
    Apply affine transformation on an image using PyTorch.

    Parameters:
    - image: Source image tensor of shape (C, H, W).
    - transform_matrix: 2x3 affine transformation matrix.
    - output_size: Desired output size as (H, W).

    Returns:
    - Transformed image tensor of shape (C, H, W).
    """
    C, H, W = image.shape
    out_H, out_W = output_size
    device = transform_matrix.device
    # Convert the 2x3 transformation matrix to 3x3
    transform_matrix = torch.tensor(transform_matrix, dtype=torch.float)
    transform_matrix = torch.cat(
        [
            transform_matrix,
            torch.tensor([[0, 0, 1]], dtype=torch.float, device=device),
        ],
        dim=0,
    )

    # Invert the transformation matrix
    transform_matrix = torch.inverse(transform_matrix)

    # Create normalized 2D grid coordinates
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, out_W, device=device),
        torch.linspace(-1, 1, out_H, device=device),
    )
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=2).reshape(-1, 3).t()

    # Apply the transformation
    transformed_grid = torch.matmul(transform_matrix[:2, :], grid)
    transformed_grid = transformed_grid.t().reshape(out_H, out_W, 2)

    # Perform grid sampling
    transformed_image = F.grid_sample(
        image.unsqueeze(0), transformed_grid.unsqueeze(0), align_corners=True
    )

    return transformed_image.squeeze(0)
