import math
import torch

from hmlib.utils.image import make_channels_first, is_channels_first, make_channels_last

import torch
import torch.nn.functional as F


def unsharp_mask(
    image: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
    amount: float = 1.0,
    channels: int = 3,
) -> torch.Tensor:
    """
    Apply sharp masking to an RGB image tensor.

    Args:
        image (torch.Tensor): Input image tensor of shape (3, H, W) or (B, 3, H, W)
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation for Gaussian kernel
        amount (float): Strength of the sharpening effect

    Returns:
        torch.Tensor: Sharpened image tensor of same shape as input
    """
    # Add batch dimension if needed

    assert torch.is_floating_point(image)

    is_cf = is_channels_first(image)
    if not is_cf:
        image = make_channels_first(image)

    assert image.dim() == 4

    image /= 255.0

    # Create Gaussian kernel
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size))
    center = kernel_size // 2

    # Fill Gaussian kernel
    for x in range(kernel_size):
        for y in range(kernel_size):
            dist = (x - center) ** 2 + (y - center) ** 2
            kernel[:, 0, x, y] = math.exp(-dist / (2 * sigma**2))

    # Normalize kernel
    kernel = kernel / kernel.sum([2, 3], keepdim=True)

    # Apply Gaussian blur
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel.to(image.device), padding=padding, groups=channels)

    # Calculate high-frequency components (details)
    high_freq = image - blurred

    # Add scaled high-frequency components back to original
    sharpened = image + amount * high_freq

    # Ensure pixel values stay in valid range
    sharpened = torch.clamp(sharpened, 0, 1)

    image = (sharpened * 255.0).clamp(0, 255)
    if not is_cf:
        image = make_channels_last(image)
    return image
