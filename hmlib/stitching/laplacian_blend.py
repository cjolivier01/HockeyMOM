"""Laplacian pyramid based image blending utilities.

Implements multi-level blending on tensors and is used by the Python
blender path of the stitching pipeline.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hmlib.ui import show_image
from hmlib.utils.image import make_visible_image


def create_gaussian_kernel(
    size=5, device=torch.device("cpu"), channels=3, sigma=1, dtype=torch.float
):
    """Create a separable Gaussian kernel tensor for convolution.

    @param size: Kernel side length (odd).
    @param device: Target device.
    @param channels: Number of output channels.
    @param sigma: Standard deviation of the Gaussian.
    @param dtype: Torch dtype.
    @return: Tensor of shape ``(channels, 1, size, size)``.
    """
    # Create Gaussian Kernel. In Numpy
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    # print(ax)
    xx, yy = np.meshgrid(ax, ax)
    # print(xx)
    # print(yy)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    # print(kernel)
    kernel /= np.sum(kernel)
    # print(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
    return kernel_tensor.to(device)


@torch.jit.script
def gaussian_conv2d(x, g_kernel):
    assert x.dtype != torch.uint8
    # Assumes input of x is of shape: (minibatch, depth, height, width)
    # Infer depth automatically based on the shape
    channels = g_kernel.shape[0]
    padding = g_kernel.shape[-1] // 2  # Kernel size needs to be odd number
    if len(x.shape) != 4:
        raise IndexError(
            "Expected input tensor to be of shape: (batch, depth, height, width) but got: "
            + str(x.shape)
        )
    y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
    return y


@torch.jit.script
def downsample(x: torch.Tensor) -> torch.Tensor:
    downsample = F.avg_pool2d(x, kernel_size=2)
    return downsample


@torch.jit.script
def upsample(image: torch.Tensor, size: List[int]) -> torch.Tensor:
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


@torch.jit.script
def create_laplacian_pyramid(
    x: torch.Tensor, kernel: torch.Tensor, levels: int
) -> List[torch.Tensor]:
    pyramids = []
    current_x = x
    for _ in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x)
        laplacian = current_x - upsample(down, size=gauss_filtered_x.shape[-2:])
        pyramids.append(laplacian)
        current_x = down
    pyramids.append(current_x)
    return pyramids


@torch.jit.script
def one_level_gaussian_pyramid(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # Gaussian blur on img
    gauss_filtered_x = gaussian_conv2d(img, kernel)
    # print(f"gauss_filtered_x: min={torch.min(gauss_filtered_x)}, max={torch.max(gauss_filtered_x)}")
    # Downsample blurred A
    down = downsample(gauss_filtered_x)
    # print(down.shape)
    return down


@torch.jit.script
def to_float(img: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if img is None:
        return None
    if img.dtype == torch.uint8:
        img = img.to(dtype, non_blocking=True)
    return img


@torch.jit.script
def simple_make_full(
    img_1: torch.Tensor,
    mask_1: Optional[torch.Tensor],
    x1: int,
    y1: int,
    img_2: torch.Tensor,
    mask_2: Optional[torch.Tensor],
    x2: int,
    y2: int,
    canvas_w: int,
    canvas_h: int,
    adjust_origin: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Pad two images (and optional masks) into a common canvas."""

    h1 = img_1.shape[2]
    w1 = img_1.shape[3]
    h2 = img_2.shape[2]
    w2 = img_2.shape[3]

    assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0

    if adjust_origin:
        if y1 <= y2:
            y2 -= y1
            y1 = 0
        elif y2 < y1:
            y1 -= y2
            y2 = 0
        if x1 <= x2:
            x2 -= x1
            x1 = 0
        elif x2 < x1:
            x1 -= x2
            x2 = 0

    # If these hit, you may have not passed "-s" to autotoptimiser
    assert int(x1) == 0 or int(x2) == 0  # for now this is the case
    assert int(y1) == 0 or int(y2) == 0  # for now this is the case

    full_1 = torch.nn.functional.pad(
        img_1,
        [
            x1,
            canvas_w - x1 - w1,
            y1,
            canvas_h - y1 - h1,
        ],
        mode="constant",
    )
    if mask_1 is not None:
        mask_1 = torch.nn.functional.pad(
            mask_1,
            [
                x1,
                canvas_w - x1 - w1,
                y1,
                canvas_h - y1 - h1,
            ],
            mode="constant",
            value=True,
        )

    full_2 = torch.nn.functional.pad(
        img_2,
        [
            x2,
            canvas_w - x2 - w2,
            y2,
            canvas_h - y2 - h2,
        ],
        mode="constant",
    )

    if mask_2 is not None:
        mask_2 = torch.nn.functional.pad(
            mask_2,
            [
                x2,
                canvas_w - x2 - w2,
                y2,
                canvas_h - y2 - h2,
            ],
            mode="constant",
            value=True,
        )

    return full_1, mask_1, full_2, mask_2


# def upsample_ignore_alpha(image, size):
#     """
#     Args:
#         image: torch.Tensor of shape (B, C, H, W) or (C, H, W)
#         size: tuple of (H_new, W_new) for target size
#     """
#     if len(image.shape) == 3:
#         image = image.unsqueeze(0)

#     rgb = image[:, :3]
#     alpha = image[:, 3:4]
#     mask = (alpha > 0).float()

#     rgb_masked = rgb * mask
#     rgb_upsampled = F.interpolate(rgb_masked, size=size, mode="bilinear", align_corners=False)
#     alpha_upsampled = F.interpolate(alpha, size=size, mode="nearest")
#     mask_upsampled = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)

#     mask_upsampled = torch.clamp(mask_upsampled, min=1e-6)
#     rgb_normalized = rgb_upsampled / mask_upsampled

#     result = torch.cat([rgb_normalized, alpha_upsampled], dim=1)
#     return result.squeeze(0) if len(image.shape) == 3 else result


@torch.jit.script
def simple_blend_in_place(
    left_small_gaussian_blurred: torch.Tensor,
    right_small_gaussian_blurred: torch.Tensor,
    mask_small_gaussian_blurred: torch.Tensor,
    level: int,
) -> torch.Tensor:
    mask_left = mask_small_gaussian_blurred
    mask_right = 1.0 - mask_small_gaussian_blurred
    left_small_gaussian_blurred *= mask_left
    right_small_gaussian_blurred *= mask_right
    left_small_gaussian_blurred += right_small_gaussian_blurred
    return left_small_gaussian_blurred


def _simple_blend_in_place(
    left_small_gaussian_blurred: torch.Tensor,  # RGBA
    right_small_gaussian_blurred: torch.Tensor,  # RGBA
    mask_small_gaussian_blurred: torch.Tensor,
    level: int,
) -> torch.Tensor:
    left_alpha = left_small_gaussian_blurred[:, 3:4]
    right_alpha = right_small_gaussian_blurred[:, 3:4]

    left_blank = (left_alpha == 0).squeeze(0).squeeze(0)
    right_blank = (right_alpha == 0).squeeze(0).squeeze(0)

    left_nonblank = (left_alpha != 0).squeeze(0).squeeze(0)
    right_nonblank = (right_alpha != 0).squeeze(0).squeeze(0)

    mask_left = mask_small_gaussian_blurred
    mask_right = 1.0 - mask_small_gaussian_blurred

    assert left_small_gaussian_blurred.shape == right_small_gaussian_blurred.shape

    left_orig = left_small_gaussian_blurred.clone()
    right_orig = right_small_gaussian_blurred.clone()

    # if level == 0:
    #     show_image(
    #         "left_" + str(level) + str(left_small_gaussian_blurred.shape),
    #         left_small_gaussian_blurred,
    #         wait=False,
    #     )

    left_small_gaussian_blurred[:, :3, left_nonblank] *= mask_left[left_nonblank]
    right_small_gaussian_blurred[:, :3, right_nonblank] *= mask_right[right_nonblank]
    left_small_gaussian_blurred[:, :3, right_nonblank] += right_small_gaussian_blurred[
        :, :3, right_nonblank
    ]

    # left_small_gaussian_blurred[:, :3, right_blank] = left_orig[:, :, right_blank]

    return left_small_gaussian_blurred


def get_alpha_mask(img: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask where RGBA alpha channel is zero."""
    assert img.ndim == 4
    mask = (img[:, 3:4] == 0).squeeze(0).squeeze(0)
    return mask


class LaplacianBlend(torch.jit.ScriptModule):
    # class LaplacianBlend(torch.nn.Module):
    def __init__(
        self,
        max_levels=4,
        channels=3,
        kernel_size=5,
        sigma=1,
        dtype: torch.dtype = torch.float,
        seam_mask: Optional[torch.Tensor] = None,
        xor_mask: Optional[torch.Tensor] = None,
    ):
        """Create a LaplacianBlend instance.

        @param max_levels: Number of pyramid levels.
        @param channels: Number of image channels.
        @param kernel_size: Gaussian kernel size.
        @param sigma: Gaussian sigma.
        @param dtype: Working dtype (usually float).
        @param seam_mask: Optional mask defining left/right regions.
        @param xor_mask: Optional XOR mask (for debugging).
        """
        super().__init__()
        self.max_levels: int = max_levels
        self.kernel_size: int = kernel_size
        self.channels: int = channels
        self.sigma: int = sigma
        self._dtype = dtype
        if seam_mask is not None:
            self.register_buffer("seam_mask", seam_mask)
        else:
            self.seam_mask = None
        if xor_mask is not None:
            self.register_buffer("xor_mask", xor_mask)
        else:
            self.xor_mask = None
        self.mask_small_gaussian_blurred: List[torch.Tensor] = []

        self.register_buffer(
            "gaussian_kernel",
            create_gaussian_kernel(
                size=self.kernel_size,
                channels=self.channels,
                sigma=self.sigma,
                device=self.seam_mask.device,
                dtype=self._dtype,
            ),
        )
        self.register_buffer(
            "mask_gaussian_kernel",
            create_gaussian_kernel(
                size=self.kernel_size,
                channels=1,
                sigma=self.sigma,
                device=self.seam_mask.device,
                dtype=self._dtype,
            ),
        )
        self.create_masks(input_shape=None, device=self.seam_mask.device)

    def create_masks(self, input_shape: torch.Size, device: torch.device):
        if self.seam_mask is None:
            mask = torch.zeros(input_shape[-2:], dtype=self._dtype, device=device)
            mask[:, : mask.shape[-1] // 2] = 1.0
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            assert input_shape is None
            mask = self.seam_mask.unsqueeze(0).unsqueeze(0).clone()
            unique_values = torch.unique(mask)
            assert len(unique_values) == 2
            left_value = self.seam_mask[self.seam_mask.shape[0] // 2][0]
            right_value = self.seam_mask[self.seam_mask.shape[0] // 2][self.seam_mask.shape[1] - 1]
            # Can we make assumption that they were discovered left-to-right?
            assert left_value == unique_values[0]
            assert right_value == unique_values[1]
            mask[mask == left_value] = 1
            mask[mask == right_value] = 0
            mask = mask.to(self._dtype)

        self._unique_values = torch.unique(self.seam_mask)
        self._left_value = self._unique_values[0]
        self._right_value = self._unique_values[1]

        mask_img = mask
        self.mask_small_gaussian_blurred = [mask.squeeze(0).squeeze(0)]
        for _ in range(self.max_levels + 1):
            mask_img = one_level_gaussian_pyramid(mask_img, self.mask_gaussian_kernel)
            self.mask_small_gaussian_blurred.append(mask_img.squeeze(0).squeeze(0))

        for i in range(len(self.mask_small_gaussian_blurred)):
            self.mask_small_gaussian_blurred[i] = self.mask_small_gaussian_blurred[i] / torch.max(
                self.mask_small_gaussian_blurred[i]
            )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for i, m in enumerate(self.mask_small_gaussian_blurred):
            self.mask_small_gaussian_blurred[i] = m.to(*args, **kwargs)

    # @torch.jit.script_method
    def forward(
        self,
        left: torch.Tensor,
        alpha_mask_left: torch.Tensor,
        x1: int,
        y1: int,
        right: torch.Tensor,
        alpha_mask_right: torch.Tensor,
        x2: int,
        y2: int,
        canvas_w: int,
        canvas_h: int,
    ) -> torch.Tensor:
        assert left.shape[-2:] == alpha_mask_left.shape
        assert right.shape[-2:] == alpha_mask_right.shape

        left, alpha_mask_left, right, alpha_mask_right = simple_make_full(
            img_1=left,
            mask_1=alpha_mask_left,
            x1=x1,
            y1=y1,
            img_2=right,
            mask_2=alpha_mask_right,
            x2=x2,
            y2=y2,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )

        # show_image("left", left, wait=False)

        # For unmapped pixels, put pixels from the other image there, or else
        # it will get interpolated towards black.

        # show_image("left", left.clone(), wait=False, scale=0.2)
        # show_image("alpha_mask_left", alpha_mask_left, wait=False, scale=0.2)
        # show_image("right", right.clone(), wait=False, scale=0.2)

        # show_image("amright", right[:, :, alpha_mask_left], wait=False, scale=0.25)

        # cv2.imwrite("left_ready.png", make_visible_image(left))
        # cv2.imwrite("right_ready.png", make_visible_image(right))
        # cv2.imwrite("seam_mask_ready.png", make_visible_image(self.seam_mask))

        left[:, :, alpha_mask_left] = right[:, :, alpha_mask_left]
        right[:, :, alpha_mask_right] = left[:, :, alpha_mask_right]

        # cv2.imwrite("left_ready.png", make_visible_image(left))
        # cv2.imwrite("right_ready.png", make_visible_image(right))
        # cv2.imwrite("seam_mask_ready.png", make_visible_image(self.seam_mask))

        # show_image("masked_left", left, wait=False, scale=0.2)

        left = to_float(left, dtype=self._dtype)
        right = to_float(right, dtype=self._dtype)

        left_laplacian = create_laplacian_pyramid(
            x=left, kernel=self.gaussian_kernel, levels=self.max_levels
        )
        right_laplacian = create_laplacian_pyramid(
            x=right, kernel=self.gaussian_kernel, levels=self.max_levels
        )

        left_small_gaussian_blurred = left_laplacian[-1]
        right_small_gaussian_blurred = right_laplacian[-1]

        F_2 = simple_blend_in_place(
            left_small_gaussian_blurred,
            right_small_gaussian_blurred,
            self.mask_small_gaussian_blurred[self.max_levels],
            level=self.max_levels,
        )

        # for this_level in reversed(range(self.max_levels)):
        this_level = self.max_levels - 1
        while this_level >= 0:
            mask_1d = self.mask_small_gaussian_blurred[this_level]

            F_1 = upsample(F_2, size=mask_1d.shape[-2:])
            upsampled_F1 = gaussian_conv2d(F_1, self.gaussian_kernel)

            L_left = left_laplacian[this_level]
            L_right = right_laplacian[this_level]

            F_2 = simple_blend_in_place(L_left, L_right, mask_1d, level=this_level)
            F_2 += upsampled_F1
            this_level -= 1

        return F_2
