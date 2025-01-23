from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hmlib.ui import show_image


def create_gaussian_kernel(
    size=5, device=torch.device("cpu"), channels=3, sigma=1, dtype=torch.float
):
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


def downsample(x: torch.Tensor) -> torch.Tensor:
    downsample = F.avg_pool2d(x, kernel_size=2)
    return downsample


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


def one_level_gaussian_pyramid(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # Gaussian blur on img
    gauss_filtered_x = gaussian_conv2d(img, kernel)
    print(f"gauss_filtered_x: min={torch.min(gauss_filtered_x)}, max={torch.max(gauss_filtered_x)}")
    # Downsample blurred A
    down = downsample(gauss_filtered_x)
    # print(down.shape)
    return down


def to_float(img: torch.Tensor, scale_variance: bool = False) -> torch.Tensor:
    if img is None:
        return None
    if img.dtype == torch.uint8:
        img = img.to(torch.float, non_blocking=True)
        if scale_variance:
            assert False
            img /= 255.0
    return img


@torch.jit.script
def simple_make_full(
    img_1: torch.Tensor,
    x1: int,
    y1: int,
    img_2: torch.Tensor,
    x2: int,
    y2: int,
    canvas_w: int,
    canvas_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    h1 = img_1.shape[2]
    w1 = img_1.shape[3]
    h2 = img_2.shape[2]
    w2 = img_2.shape[3]

    assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
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
    assert x1 == 0 or x2 == 0  # for now this is the case
    assert y1 == 0 or y2 == 0  # for now this is the case

    full_left = torch.nn.functional.pad(
        img_1,
        [
            x1,
            canvas_w - x1 - w1,
            y1,
            canvas_h - y1 - h1,
        ],
        mode="constant",
    )

    full_right = torch.nn.functional.pad(
        img_2,
        [
            x2,
            canvas_w - x2 - w2,
            y2,
            canvas_h - y2 - h2,
        ],
        mode="constant",
    )

    return full_left, full_right


class LaplacianBlend(torch.nn.Module):
    def __init__(
        self,
        max_levels=4,
        channels=3,
        kernel_size=5,
        sigma=1,
        seam_mask: Optional[torch.Tensor] = None,
        xor_mask: Optional[torch.Tensor] = None,
    ):
        super(LaplacianBlend, self).__init__()
        self.max_levels: int = max_levels
        self.kernel_size: int = kernel_size
        self.channels: int = channels
        self.sigma: int = sigma
        if seam_mask is not None:
            # self.register_buffer("seam_mask", to_float(seam_mask))
            self.register_buffer("seam_mask", seam_mask)
        else:
            self.seam_mask = None
        if xor_mask is not None:
            # self.register_buffer("xor_mask", to_float(xor_mask))
            self.register_buffer("xor_mask", xor_mask)
        else:
            self.xor_mask = None
        self.mask_small_gaussian_blurred: List[torch.Tensor] = []
        self._initialized = False

    def create_masks(self, input_shape: torch.Size, device: torch.device):
        if self.seam_mask is None:
            mask = torch.zeros(input_shape[-2:], dtype=torch.float, device=device)
            mask[:, : mask.shape[-1] // 2] = 1.0
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            assert input_shape is None
            mask = self.seam_mask.unsqueeze(0).unsqueeze(0)
            unique_values = torch.unique(mask)
            assert len(unique_values) == 2
            left_value = unique_values[0]
            right_value = unique_values[1]
            mask[mask == left_value] = 1
            mask[mask == right_value] = 0
            mask = mask.to(torch.float)

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

    def initialize(self, input_shape: torch.Size, device: torch.device):
        assert not self._initialized
        self.gaussian_kernel = create_gaussian_kernel(
            size=self.kernel_size,
            channels=self.channels,
            sigma=self.sigma,
            device=device,
        )
        self.mask_gaussian_kernel = create_gaussian_kernel(
            size=self.kernel_size,
            channels=1,
            sigma=self.sigma,
            device=device,
        )
        self.create_masks(input_shape=input_shape, device=device)
        self._initialized = True

    @staticmethod
    def make_full(
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        level: int,
        level_ainfo_1,
        level_ainfo_2,
        level_canvas_dims,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ainfo_1 = level_ainfo_1[level]
        ainfo_2 = level_ainfo_2[level]

        h1, w1, x1, y1 = ainfo_1
        h2, w2, x2, y2 = ainfo_2

        # If these hit, you may have not passed "-s" to autotoptimiser
        assert x1 == 0 or x2 == 0  # for now this is the case
        assert y1 == 0 or y2 == 0  # for now this is the case

        canvas_dims = level_canvas_dims[level]

        full_left = torch.nn.functional.pad(
            img_1,
            [
                x1,
                canvas_dims[1] - x1 - w1,
                y1,
                canvas_dims[0] - y1 - h1,
            ],
            mode="constant",
        )

        full_right = torch.nn.functional.pad(
            img_2,
            [
                x2,
                canvas_dims[1] - x2 - w2,
                y2,
                canvas_dims[0] - y2 - h2,
            ],
            mode="constant",
        )

        return full_left, full_right

    # @torch.jit.script_method
    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        level_ainfo_1,
        level_ainfo_2,
        level_canvas_dims,
    ) -> torch.Tensor:
        make_full_first: bool = True
        if make_full_first:

            h1, w1, x1, y1 = level_ainfo_1[0]
            h2, w2, x2, y2 = level_ainfo_2[0]

            canvas_w = level_canvas_dims[0][1]
            canvas_h = level_canvas_dims[0][0]

            left, right = simple_make_full(
                img_1=left,
                x1=x1,
                y1=y1,
                img_2=right,
                x2=x2,
                y2=y2,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )

            # left, right = self.make_full(
            #     left,
            #     right,
            #     level=0,
            #     level_ainfo_1=level_ainfo_1,
            #     level_ainfo_2=level_ainfo_2,
            #     level_canvas_dims=level_canvas_dims,
            # )

        left = to_float(left, scale_variance=False)
        right = to_float(right, scale_variance=False)

        # assert left.shape == right.shape  # They should be "full" already
        if not self._initialized:
            # Let's not do this anymore
            assert False

        left_laplacian = create_laplacian_pyramid(
            x=left, kernel=self.gaussian_kernel, levels=self.max_levels
        )
        right_laplacian = create_laplacian_pyramid(
            x=right, kernel=self.gaussian_kernel, levels=self.max_levels
        )

        left_small_gaussian_blurred = left_laplacian[-1]
        right_small_gaussian_blurred = right_laplacian[-1]

        mask_1d = self.mask_small_gaussian_blurred[self.max_levels]
        mask_left = mask_1d
        mask_right = 1 - mask_1d

        assert level_canvas_dims is not None

        if not make_full_first:
            (
                left_small_gaussian_blurred,
                right_small_gaussian_blurred,
            ) = self.make_full(
                left_small_gaussian_blurred,
                right_small_gaussian_blurred,
                level=self.max_levels,
                level_ainfo_1=level_ainfo_1,
                level_ainfo_2=level_ainfo_2,
                level_canvas_dims=level_canvas_dims,
            )

        F_2 = left_small_gaussian_blurred * mask_left + right_small_gaussian_blurred * mask_right
        # show_image("F_2", F_2)

        for this_level in reversed(range(self.max_levels)):
            mask_1d = self.mask_small_gaussian_blurred[this_level]
            mask_left = mask_1d
            mask_right = 1 - mask_1d

            F_1 = upsample(F_2, size=mask_1d.shape[-2:])
            upsampled_F1 = gaussian_conv2d(F_1, self.gaussian_kernel)

            L_left = left_laplacian[this_level]
            L_right = right_laplacian[this_level]

            if level_canvas_dims is not None:
                if not make_full_first:
                    L_left, L_right = self.make_full(
                        L_left,
                        L_right,
                        level=this_level,
                        level_ainfo_1=level_ainfo_1,
                        level_ainfo_2=level_ainfo_2,
                        level_canvas_dims=level_canvas_dims,
                    )
                    assert L_left.shape[-2:] == mask_left.shape
                    assert L_right.shape[-2:] == mask_right.shape

            L_left *= mask_left
            L_right *= mask_right
            L_left += L_right
            L_left += upsampled_F1
            F_2 = L_left

        # if self.xor_mask is not None:
        #     l_full, r_full = self.make_full(
        #         left,
        #         right,
        #         level=0,
        #         level_ainfo_1=level_ainfo_1,
        #         level_ainfo_2=level_ainfo_2,
        #         level_canvas_dims=level_canvas_dims,
        #     )
        #     show_image("l_full", l_full, wait=False, enable_resizing=0.1)
        #     show_image("r_full", r_full, wait=False, enable_resizing=0.1)
        #     F_2[:, :, self.xor_mask == self._left_value] = l_full[
        #         :, :, self.xor_mask == self._left_value
        #     ]
        #     # F_2[:, :, self.xor_mask == self._right_value] = r_full[
        #     #     :, :, self.xor_mask == self._right_value
        #     # ]
        #     show_image("F_2", F_2, wait=False, enable_resizing=0.1)

        return F_2
