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

from hmlib.stitching.laplacian_blend import LaplacianBlend

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
    kernel_tensor.to(device)
    return kernel_tensor


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
    return x[:, :, ::2, ::2]


def F_transform(img, kernel):
    upsample = torch.nn.Upsample(
        scale_factor=2
    )  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    large = upsample(img)
    upsampled = gaussian_conv2d(large, kernel)
    return upsampled


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
    upsample = torch.nn.Upsample(
        scale_factor=2
    )  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    pyramids = []
    small_gaussian_blurred = []
    current_x = x
    for level in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x)
        # Original Algorithm does indeed: L_i  = G_i  - expand(G_i+1), with L_i as current laplacian layer, and G_i as current gaussian filtered image, and G_i+1 the next.
        # Some implementations skip expand(G_i+1) and use gaussian_conv(G_i). We decided to use expand, as is the original algorithm
        laplacian = current_x - upsample(down)
        pyramids.append(laplacian)
        small_gaussian_blurred.append(down)
        current_x = down
    pyramids.append(current_x)
    return pyramids, small_gaussian_blurred


def one_level_gaussian_pyramid(img, kernel):
    # Gaussian blur on img
    gauss_filtered_x = gaussian_conv2d(img, kernel)
    # Downsample blurred A
    down = downsample(gauss_filtered_x)
    return down


class LaplacianPyramidLoss(torch.nn.Module):
    def __init__(
        self,
        max_levels=3,
        channels=3,
        kernel_size=5,
        sigma=1,
        device=torch.device("cpu"),
        dtype=torch.float,
    ):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel = create_gaussian_kernel(
            size=kernel_size, channels=channels, sigma=sigma, dtype=dtype
        )

    def forward(self, x, target):
        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels)
        target_pyramid = create_laplacian_pyramid(target, self.kernel, self.max_levels)
        return sum(
            torch.nn.functional.l1_loss(x, y)
            for x, y in zip(input_pyramid, target_pyramid)
        )


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
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_width, pad_height, 0))
    else:
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

    return padded_tensor


def test():
    # Test Gaussian Convolution
    kernel = create_gaussian_kernel(size=3)
    x = torch.ones(1, 3, 3, 3)
    y = gaussian_conv2d(x, kernel)
    print("Gaussian  kernel of size 3, sigma=1")
    print(create_gaussian_kernel(size=3))
    print("\n")
    print("A 3x3 image of ones convoluted with above filter:")
    print(y)
    print(y.shape)

    # Below are the difference of: expand(G_i+1) and using G_i but filtered, that have been observed in certain Laplacian Loss implementations.
    #
    x = torch.randn(1, 3, 12, 12)
    y = gaussian_conv2d(x, kernel)
    upsample = torch.nn.Upsample(
        scale_factor=2
    )  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    down = downsample(y)
    up = upsample(down)
    diff = y - up
    print("The difference between the two tensors: up(down)-sampling vs only smooth")
    print(torch.sum(diff))
    print(" x - up: ")
    print(torch.sum(x - up))
    print("x - y:")
    print(torch.sum(x - y))

    x = torch.randn(1, 3, 64, 64)
    y = torch.randn(1, 3, 64, 64)
    encoder = torch.nn.Conv2d(
        3, 3, 1
    )  # A test convolution that retains the dimensionality.
    optimizer = torch.optim.Adam(encoder.parameters())
    loss = LaplacianPyramidLoss()
    print("Laplacian Loss: ")
    lap_loss = loss(encoder(x), y)
    print(lap_loss)
    print("gradient before backward(): (weight, bias)")
    print(encoder.weight.grad)
    print(encoder.bias.grad)
    lap_loss.backward()
    print("gradient after backward(): (weight, bias)")
    print(encoder.weight.grad)
    print(encoder.bias.grad)
    print(encoder.weight.grad)
    for i in range(1000):
        optimizer.zero_grad()
        lap_loss = loss(encoder(x), y)
        lap_loss.backward()
        optimizer.step()
        print(f"Loss: {lap_loss}", end="\r")


def read_image_as_float(path: str, device):
    return (
        make_channels_first(torch.from_numpy(cv2.imread(path)).unsqueeze(0))
        .to(device)
        .to(torch.float32)
        / 255.0
    )


if __name__ == "__main__":
    # test()

    device = "cuda"
    # device = "cpu"

    levels = 4

    # Load Images
    # left = read_image_as_float(
    #     "/home/colivier/src/laplacian_blend/left.png",
    #     device=device,
    # )
    # right = read_image_as_float(
    #     "/home/colivier/src/laplacian_blend/right.png",
    #     device=device,
    # )

    left = read_image_as_float(
        "/mnt/ripper-data/Videos/sharks-bb1-2/left.png",
        device=device,
    )
    right = read_image_as_float(
        "/mnt/ripper-data/Videos/sharks-bb1-2/right.png",
        device=device,
    )

    left = pad_to_multiple_of(left, mult=64, left=True)
    right = pad_to_multiple_of(right, mult=64, left=False)

    if True:
        blender = LaplacianBlend()
        blender = blender.to(device)
        F_2 = blender.forward(left=left, right=right)
    else:

        gaussian_kernel = create_gaussian_kernel(size=5).to(device)
        mask_gaussian_kernel = create_gaussian_kernel(size=5, channels=1).to(device)

        # show("left", left[0])
        # show("right", right[0])

        left_laplacian, left_small_gaussian_blurred = create_laplacian_pyramid(
            x=left, kernel=gaussian_kernel, levels=levels
        )
        right_laplacian, right_small_gaussian_blurred = create_laplacian_pyramid(
            x=right, kernel=gaussian_kernel, levels=levels
        )

        # Skip last downsampled that was put at the end
        # left_laplacian = left_laplacian[:-1]
        # right_laplacian = right_laplacian[:-1]

        # left_reconstructed_imgs = []
        # right_reconstructed_imgs = []

        # start_F = left_small_gaussian_blurred[-1]

        # for i in reversed(range(0, levels)):
        #     reconstructed_F = F_transform(start_F, gaussian_kernel) + left_laplacian[i]
        #     left_reconstructed_imgs.append(reconstructed_F)
        #     start_F = reconstructed_F

        # start_F = right_small_gaussian_blurred[-1]
        # for i in reversed(range(0, levels)):
        #     reconstructed_F = F_transform(start_F, gaussian_kernel) + right_laplacian[i]
        #     right_reconstructed_imgs.append(reconstructed_F)
        #     start_F = reconstructed_F

        # TODO: we don't need anything but rows and cols in the mask
        # mask = torch.zeros_like(right)
        mask = torch.zeros(right.shape[-2:], dtype=right.dtype, device=right.device)
        mask[:, : mask.shape[-1] // 2] = 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        # show("mask", mask[0].repeat(3, 1, 1))

        # I guess here we just blur the mask seam?

        img = mask

        mask_small_gaussian_blurred = [mask.squeeze(0).squeeze(0)]
        for _ in range(levels + 1):
            img = one_level_gaussian_pyramid(img, mask_gaussian_kernel)
            mask_small_gaussian_blurred.append(img.squeeze(0).squeeze(0))

        img = mask_small_gaussian_blurred[-1]
        # show("mask", img)

        # TODO: as stacked batch (makes max element of 1.0)
        for i in range(len(mask_small_gaussian_blurred)):
            mask_small_gaussian_blurred[i] = mask_small_gaussian_blurred[i] / torch.max(
                mask_small_gaussian_blurred[i]
            )

        # show("mask_G_small_gaussian_blurred", mask_small_gaussian_blurred[-1][0])
        # plt.imshow(mask_small_gaussian_blurred[-1][0])

        # for i in mask_small_gaussian_blurred:
        #     print(i.shape)
        # print("")
        # for i in right_small_gaussian_blurred:
        #     print(i.shape)
        # print("")
        # for i in right_laplacian:
        #     print(i.shape)
        # print("")

        ONE = torch.tensor(1.0, dtype=torch.float, device=img.device)

        up_sample = torch.nn.Upsample(
            scale_factor=2
        )  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]

        #
        # Perform the Laplacian blending
        #
        if False:
            # mask_left_1d = mask_small_gaussian_blurred[-2]
            # mask_right_1d = ONE - mask_small_gaussian_blurred[-2]

            # mask_left = mask_left_1d.repeat(3, 1, 1)
            # mask_right = mask_right_1d.repeat(3, 1, 1)

            # G_c = (
            #     left_small_gaussian_blurred[-1] * mask_left
            #     + right_small_gaussian_blurred[-1] * mask_right
            # )

            # F_1 = up_sample(G_c)
            # upsampled_F1 = gaussian_conv2d(F_1, gaussian_kernel)

            # mask_left_1d = mask_small_gaussian_blurred[-3]
            # mask_right_1d = ONE - mask_small_gaussian_blurred[-3]

            # mask_left = mask_left_1d.repeat(3, 1, 1)
            # mask_right = mask_right_1d.repeat(3, 1, 1)

            # # show("mask_left", mask_left, wait=True)
            # # show("mask_right", mask_right, wait=True)

            # L_a = left_laplacian[-1]
            # L_o = left_laplacian[-1]

            # # F_2 = L_c + upsampled_F1
            # print(upsampled_F1.shape)

            # # show("upsampled_F1", upsampled_F1, wait=True)

            # L_c = (mask_left * L_a) + (mask_right * L_o)

            # F_2 = L_c + upsampled_F1

            # show("F_2", F_2, wait=True)

            # Start over for some reason fucking illustrative purposes or something
            # Need to make a loop

            reconstructed_imgs = []

            this_level = levels - 1  # 4

            # TODO: do math before repeated
            mask_1d = mask_small_gaussian_blurred[this_level + 1].repeat(3, 1, 1)
            mask_left = mask_1d
            mask_right = ONE - mask_1d

            G_c = (
                left_small_gaussian_blurred[this_level] * mask_left
                + right_small_gaussian_blurred[this_level] * mask_right
            )

            F_1 = up_sample(G_c)
            upsampled_F1 = gaussian_conv2d(F_1, gaussian_kernel)

            mask_1d = mask_small_gaussian_blurred[this_level].repeat(3, 1, 1)
            mask_left = mask_1d
            mask_right = ONE - mask_1d

            L_a = left_laplacian[this_level]
            L_o = right_laplacian[this_level]

            L_c = (mask_left * L_a) + (mask_right * L_o)
            F_2 = L_c + upsampled_F1

            # show("F_2", F_2, wait=True)

            reconstructed_imgs.append(F_2)
            this_level -= 1

            # Second set
            F_1 = up_sample(F_2)
            upsampled_F1 = gaussian_conv2d(F_1, gaussian_kernel)

            mask_1d = mask_small_gaussian_blurred[this_level].repeat(3, 1, 1)
            mask_left = mask_1d
            mask_right = ONE - mask_1d

            L_a = left_laplacian[this_level]
            L_o = right_laplacian[this_level]

            L_c = (mask_left * L_a) + (mask_right * L_o)
            F_2 = L_c + upsampled_F1

            print(upsampled_F1.shape)
            # show("F_2", F_2, wait=True)

            reconstructed_imgs.append(F_2)
            this_level -= 1

            # Third set
            F_1 = up_sample(F_2)
            # show("F_1", F_1, wait=True)
            upsampled_F1 = gaussian_conv2d(F_1, gaussian_kernel)

            mask_1d = mask_small_gaussian_blurred[this_level].repeat(3, 1, 1)
            mask_left = mask_1d
            mask_right = ONE - mask_1d

            # show("mask_left", mask_left, wait=True)
            # show("mask_right", mask_right, wait=True)

            L_a = left_laplacian[this_level]
            L_o = right_laplacian[this_level]

            # show("L_a", L_a, wait=True)
            # show("L_o", L_o, wait=True)

            L_c = (mask_left * L_a) + (mask_right * L_o)
            F_2 = L_c + upsampled_F1
            print(upsampled_F1.shape)

            show("F_2", F_2, wait=True)

            reconstructed_imgs.append(F_2)
            this_level -= 1

            # Fourth set
            F_1 = up_sample(F_2)
            upsampled_F1 = gaussian_conv2d(F_1, gaussian_kernel)

            mask_1d = mask_small_gaussian_blurred[this_level].repeat(3, 1, 1)
            mask_left = mask_1d
            mask_right = ONE - mask_1d

            L_a = left_laplacian[this_level]
            L_o = right_laplacian[this_level]

            L_c = (mask_left * L_a) + (mask_right * L_o)
            F_2 = L_c + upsampled_F1
            print(upsampled_F1.shape)
            show("F_2", F_2, wait=True)
        else:
            mask_1d = mask_small_gaussian_blurred[levels].repeat(3, 1, 1)
            mask_left = mask_1d
            mask_right = ONE - mask_1d

            F_2 = (
                left_small_gaussian_blurred[levels - 1] * mask_left
                + right_small_gaussian_blurred[levels - 1] * mask_right
            )

            for this_level in reversed(range(levels)):
                F_1 = up_sample(F_2)
                upsampled_F1 = gaussian_conv2d(F_1, gaussian_kernel)

                mask_1d = mask_small_gaussian_blurred[this_level].repeat(3, 1, 1)
                mask_left = mask_1d
                mask_right = ONE - mask_1d

                L_a = left_laplacian[this_level]
                L_o = right_laplacian[this_level]

                L_c = (mask_left * L_a) + (mask_right * L_o)
                F_2 = L_c + upsampled_F1

    show("F_2", F_2, wait=True)

    print("Done.")
