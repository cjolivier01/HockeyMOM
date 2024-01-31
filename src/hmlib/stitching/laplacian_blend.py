import cv2
import torch
import torchvision as tv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hmlib.utils.image import (
    make_channels_first,
    image_width,
    image_height,
    resize_image,
)
from hmlib.video_out import make_visible_image


def create_gaussian_kernel(
    size=5, device=torch.device("cpu"), channels=3, sigma=1, dtype=torch.float
):
    # Create Gaussian Kernel. In Numpy
    # interval = (2 * sigma + 1) / (size)
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
    return kernel_tensor.to(device)


def gaussian_conv2d(x, g_kernel, dtype=torch.float):
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


def downsample(x):
    # Downsamples along  image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
    # return x[:, :, ::2, ::2]
    downsample = F.avg_pool2d(x, kernel_size=2)
    return downsample


def upsample(image, size):
    # upsampled = F.interpolate(blurred, scale_factor=2, mode='bilinear', align_corners=False)
    # print(f"upsample {image.shape[-2:]} -> {size}")
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


# def F_transform(img, kernel, size):
#     # upsample = torch.nn.Upsample(
#     #     scale_factor=2
#     # )  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
#     large = upsample(img, size)
#     upsampled = gaussian_conv2d(large, kernel)
#     return upsampled


def show(label: str, img: torch.Tensor, wait: bool = True, min_width: int = 300):
    if img.ndim == 2:
        # grayscale
        img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    if min_width and image_width(img) < min_width:
        print("Increasing image size for viewing...")
        ar = float(image_width(img)) / image_height(img)
        w = min_width
        h = min_width / ar
        img = resize_image(
            img,
            new_width=w,
            new_height=h,
            mode=tv.transforms.InterpolationMode.BILINEAR,
        )
    if img.ndim == 4:
        for i in img:
            cv2.imshow(label, make_visible_image(i))
            cv2.waitKey(1 if not wait else 0)
    else:
        cv2.imshow(label, make_visible_image(img))
        cv2.waitKey(1 if not wait else 0)


def create_laplacian_pyramid(x, kernel, levels):
    pyramids = []
    small_gaussian_blurred = []
    current_x = x
    for level in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x)
        laplacian = current_x - upsample(down, size=gauss_filtered_x.shape[-2:])
        pyramids.append(laplacian)
        current_x = down
    pyramids.append(current_x)
    return pyramids


def one_level_gaussian_pyramid(img, kernel):
    # Gaussian blur on img
    gauss_filtered_x = gaussian_conv2d(img, kernel)
    # Downsample blurred A
    down = downsample(gauss_filtered_x)
    # print(down.shape)
    return down


def pad_to_multiple_of(tensor, mult: int, left: bool):
    # Calculate the desired height and width after padding
    height, width = tensor.size(2), tensor.size(3)
    new_height = ((height - 1) // int(mult) + 1) * int(mult)
    new_width = ((width - 1) // int(mult) + 1) * int(mult)

    # Calculate the amount of padding needed
    pad_height = new_height - height
    pad_width = new_width - width

    # Apply padding to the tensor
    if left:
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, pad_width, pad_height, 0), mode="constant"
        )
    elif left is not None:
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, pad_width, 0, pad_height), mode="constant"
        )
    else:
        ww = pad_width // 2
        # hh = pad_height // 2
        padded_tensor = torch.nn.functional.pad(
            tensor, (pad_width - ww, ww, 0, pad_height), mode="replicate"
        )
    return padded_tensor


def to_float(img: torch.Tensor, scale_variance: bool = True):
    if img is None:
        return None
    if img.dtype == torch.uint8:
        img = img.to(torch.float)
        # if scale_variance:
        #     img /= 255.0
    return img


class LaplacianBlend(torch.nn.Module):
    def __init__(
        self,
        max_levels=4,
        channels=3,
        kernel_size=5,
        sigma=1,
        seam_mask: torch.Tensor = None,
        xor_mask: torch.Tensor = None,
    ):
        super(LaplacianBlend, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.channels = channels
        self.sigma = sigma
        if seam_mask is not None:
            self.register_buffer("seam_mask", to_float(seam_mask))
        else:
            self.seam_mask = None
        if xor_mask is not None:
            self.register_buffer("xor_mask", to_float(xor_mask))
        else:
            self.seam_mask = None
        self.register_buffer("ONE", torch.tensor(1.0, dtype=torch.float))
        self.mask_small_gaussian_blurred = []
        self._initialized = False

    def create_masks(self, input_shape: torch.Size, device: torch.device):
        if self.seam_mask is None:
            mask = torch.zeros(input_shape[-2:], dtype=torch.float, device=device)
            # mask[:, : mask.shape[-1] // 2] = 255.0
            mask[:, : mask.shape[-1] // 2] = 1.0
            mask = mask.unsqueeze(0).unsqueeze(0)
            # show("mask", mask[0].repeat(3, 1, 1))
        else:
            mask = self.seam_mask.unsqueeze(0).unsqueeze(0)
            unique_values = torch.unique(mask)
            assert len(unique_values) == 2
            left_value = unique_values[0]
            right_value = unique_values[1]
            mask[mask == left_value] = 0
            mask[mask == right_value] = 1.0
            mask = pad_to_multiple_of(mask, 64, left=None)  # this pad will be bad

        img = mask

        self.mask_small_gaussian_blurred = [mask.squeeze(0).squeeze(0)]
        for _ in range(self.max_levels + 1):
            img = one_level_gaussian_pyramid(img, self.mask_gaussian_kernel)
            self.mask_small_gaussian_blurred.append(img.squeeze(0).squeeze(0))

        img = self.mask_small_gaussian_blurred[-1]

        for i in range(len(self.mask_small_gaussian_blurred)):
            self.mask_small_gaussian_blurred[i] = self.mask_small_gaussian_blurred[
                i
            ] / torch.max(self.mask_small_gaussian_blurred[i])

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

    def forward(self, left, right):
        if not self._initialized:
            self.initialize(input_shape=left.shape, device=left.device)
        if False:
            return self.gpt_forward(left, right)
        else:
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
            mask_right = self.ONE - mask_1d

            F_2 = (
                left_small_gaussian_blurred * mask_left
                + right_small_gaussian_blurred * mask_right
            )

            for this_level in reversed(range(self.max_levels)):
                mask_1d = self.mask_small_gaussian_blurred[this_level]
                mask_left = mask_1d
                mask_right = self.ONE - mask_1d

                F_1 = upsample(F_2, size=mask_1d.shape[-2:])
                upsampled_F1 = gaussian_conv2d(F_1, self.gaussian_kernel)

                L_left = left_laplacian[this_level]
                L_right = right_laplacian[this_level]

                L_c = (mask_left * L_left) + (mask_right * L_right)
                F_2 = L_c + upsampled_F1

            return F_2

    #
    # GPT
    #
    def gpt_forward(self, image1, image2):
        if not self._initialized:
            self.initialize(input_shape=image1.shape, device=image1.device)

        # Build Laplacian pyramids for both images
        pyramid1 = self.build_laplacian_pyramid(image1)
        pyramid2 = self.build_laplacian_pyramid(image2)

        mask_1d = self.mask_small_gaussian_blurred[self.max_levels].repeat(3, 1, 1)

        # Blend the pyramids using the seam mask
        blended_pyramid = []
        for level in range(self.max_levels):
            seam_mask = self.mask_small_gaussian_blurred[level]
            assert seam_mask.shape[-2:] == pyramid1[level].shape[-2:]
            blended_level = pyramid1[level] * seam_mask + pyramid2[level] * (
                1 - seam_mask
            )
            blended_pyramid.append(blended_level)

        # Reconstruct the blended image from the blended pyramid
        blended_image = self.reconstruct_laplacian_pyramid(blended_pyramid)

        return blended_image

    def build_laplacian_pyramid(self, image):
        pyramid = []
        for _ in range(self.max_levels):
            blurred = F.avg_pool2d(image, kernel_size=2)
            upsampled = F.interpolate(
                blurred, scale_factor=2, mode="bilinear", align_corners=False
            )
            residual = image - upsampled
            pyramid.append(residual)
            image = blurred
        pyramid.append(image)  # Low-pass residual image
        return pyramid

    def reconstruct_laplacian_pyramid(self, pyramid):
        image = pyramid[-1]
        for i in range(self.max_levels - 2, -1, -1):
            upsampled = F.interpolate(
                image, scale_factor=2, mode="bilinear", align_corners=False
            )
            image = upsampled + pyramid[i]
        return image
