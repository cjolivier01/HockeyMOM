import torch
import torch.nn.functional as F
from hmlib.utils.image import image_width, image_height, make_channels_first, make_channels_last

def warp_affine_pytorch(image, affine_matrix, output_size):
    """
    Apply an affine transformation to an image tensor

    Parameters:
    - image: A (C, H, W) or (N, C, H, W) image tensor.
    - affine_matrix: A (2, 3) affine transformation matrix.
    - output_size: A tuple (H, W) specifying the output image size.

    Returns:
    - Transformed image tensor.
    """
    # Ensure the image tensor is 4D (batch, channels, height, width)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    iw = image_width(image)
    ih = image_height(image)

    # Create the affine grid
    if affine_matrix.dtype != image.dtype:
        affine_matrix = affine_matrix.to(image.dtype, non_blocking=True)
    theta = affine_matrix.unsqueeze(0)  # Add batch dimension
    grid = F.affine_grid(
        theta,
        [1, 3, output_size[0], output_size[1]],
        align_corners=False,
    )

    # Apply the transformation
    image = make_channels_first(image)
    transformed_image = F.grid_sample(image, grid, align_corners=False)
    transformed_image = make_channels_last(transformed_image)

    return transformed_image


# Example usage
# Assuming `image_tensor` is your image tensor of shape (C, H, W) or (N, C, H, W)
# and `affine_matrix` is a (2, 3) tensor representing your affine transformation
# output_size = (H, W)  # Desired output size
# transformed_image = warp_affine_pytorch(image_tensor, affine_matrix, output_size)
