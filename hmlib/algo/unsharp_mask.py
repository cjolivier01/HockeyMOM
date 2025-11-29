"""Unsharp masking filters for sharpening hockey video frames.

These utilities operate on float image tensors and are typically wired into
Aspen pipelines via :class:`hmlib.transforms.video_frame.HmUnsharpMask`.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from hmlib.utils.image import is_channels_first, make_channels_first, make_channels_last


def unsharp_mask(
    image: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
    amount: float = 0.5,
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

    if False:
        image = unsharp_mask_batch(image)
    else:

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
        image = torch.clamp(sharpened, 0, 1)

    image = (image * 255.0).clamp(0, 255)

    if not is_cf:
        image = make_channels_last(image)
    return image


def unsharp_mask_batch(
    images: Tensor,
    radius_luma: float = 3.0,
    amount_luma: float = 1.5,
    radius_chroma: float = 3.0,
    amount_chroma: float = 0.5,
) -> Tensor:
    """
    Apply unsharp masking to a batch of float32 RGB images using PyTorch.

    Args:
        images: Tensor of shape (batch_size, 3, height, width) with values in range [0, 1]
        radius_luma: gaussian blur radius for luminance
        amount_luma: strength of sharpening for luminance
        radius_chroma: gaussian blur radius for chrominance
        amount_chroma: strength of sharpening for chrominance

    Returns:
        Tensor: Sharpened images of same shape and dtype as input
    """
    if not torch.is_tensor(images):
        raise TypeError("Input must be a PyTorch tensor")

    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("Input must have shape (batch_size, 3, height, width)")

    device = images.device
    batch_size, _, height, width = images.shape

    # RGB to YUV conversion matrix
    rgb_to_yuv = torch.tensor(
        [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]],
        dtype=images.dtype,
        device=device,
    )

    # Reshape for matrix multiplication
    images_reshaped = images.permute(0, 2, 3, 1)  # (B, H, W, 3)
    yuv = torch.matmul(images_reshaped, rgb_to_yuv.T)
    yuv = yuv.permute(0, 3, 1, 2)  # (B, 3, H, W)

    # Gaussian kernel generation
    def create_gaussian_kernel(radius: float) -> Tensor:
        kernel_size = int(radius * 2) | 1  # Ensure odd size
        sigma = radius / 3
        grid_x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
        grid_y = grid_x.unsqueeze(1)
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    # Apply unsharp mask
    sharp_yuv = yuv.clone()

    # Y channel (luminance)
    kernel_y = create_gaussian_kernel(radius_luma)
    padding = kernel_y.shape[0] // 2
    kernel_y = kernel_y.expand(1, 1, -1, -1)

    blurred_y = torch.nn.functional.conv2d(yuv[:, 0:1], kernel_y, padding=padding, groups=1)
    sharp_yuv[:, 0:1] += (yuv[:, 0:1] - blurred_y) * amount_luma

    # UV channels (chrominance)
    kernel_uv = create_gaussian_kernel(radius_chroma)
    padding = kernel_uv.shape[0] // 2
    kernel_uv = kernel_uv.expand(1, 1, -1, -1)

    for i in range(1, 3):
        blurred_uv = torch.nn.functional.conv2d(yuv[:, i : i + 1], kernel_uv, padding=padding, groups=1)
        sharp_yuv[:, i : i + 1] += (yuv[:, i : i + 1] - blurred_uv) * amount_chroma

    # YUV to RGB conversion
    yuv_to_rgb = torch.linalg.inv(rgb_to_yuv)
    sharp_yuv = sharp_yuv.permute(0, 2, 3, 1)  # (B, H, W, 3)
    sharp_rgb = torch.matmul(sharp_yuv, yuv_to_rgb.T)
    sharp_rgb = sharp_rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)

    return torch.clamp(sharp_rgb, 0, 1)


def example_usage() -> None:
    # Example usage with random batch of images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, height, width = 32, 3, 720, 1280

    # Create random test images
    images = torch.rand((batch_size, channels, height, width), dtype=torch.float32, device=device)

    # Apply unsharp masking
    sharpened = unsharp_mask_batch(images)

    assert sharpened.shape == images.shape
    assert torch.all((sharpened >= 0) & (sharpened <= 1))


if __name__ == "__main__":
    example_usage()
