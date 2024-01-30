import cv2
import torch
import numpy as np
import torch.nn.functional as F

from hmlib.utils.image import make_channels_first
from hmlib.video_out import make_visible_image


def gaussian_kernel(
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


def show(label: str, img: torch.Tensor, wait: bool = True):
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
        self.kernel = gaussian_kernel(
            size=kernel_size, channels=channels, sigma=sigma, dtype=dtype
        )

    def forward(self, x, target):
        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels)
        target_pyramid = create_laplacian_pyramid(target, self.kernel, self.max_levels)
        return sum(
            torch.nn.functional.l1_loss(x, y)
            for x, y in zip(input_pyramid, target_pyramid)
        )


def test():
    # Test Gaussian Convolution
    kernel = gaussian_kernel(size=3)
    x = torch.ones(1, 3, 3, 3)
    y = gaussian_conv2d(x, kernel)
    print("Gaussian  kernel of size 3, sigma=1")
    print(gaussian_kernel(size=3))
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


def read_image_as_float(path: str):
    return (
        make_channels_first(torch.from_numpy(cv2.imread(path)).unsqueeze(0)).to(
            torch.float32
        )
        / 255.0
    )


if __name__ == "__main__":
    # test()

    gaussian_kernel = gaussian_kernel(size=5)

    levels = 4

    # Load Images
    apple = read_image_as_float("/home/colivier/src/laplacian_blend/apple.png")
    orange = read_image_as_float("/home/colivier/src/laplacian_blend/orange.png")
    # show("apple", apple[0])
    # show("orange", orange[0])

    apple_laplacian, apple_small_gaussian_blurred = create_laplacian_pyramid(
        x=apple, kernel=gaussian_kernel, levels=levels
    )
    orange_laplacian, orange_small_gaussian_blurred = create_laplacian_pyramid(
        x=orange, kernel=gaussian_kernel, levels=levels
    )

    # Skip last downsampled that was put at the end
    apple_laplacian = apple_laplacian[:-1]
    orange_laplacian = orange_laplacian[:-1]

    apple_reconstructed_imgs = []
    orange_reconstructed_imgs = []

    start_F = apple_small_gaussian_blurred[-1]

    for i in reversed(range(0, levels)):
        reconstructed_F = F_transform(start_F, gaussian_kernel) + apple_laplacian[i]
        apple_reconstructed_imgs.append(reconstructed_F)
        start_F = reconstructed_F

    start_F = orange_small_gaussian_blurred[-1]
    for i in reversed(range(0, levels)):
        reconstructed_F = F_transform(start_F, gaussian_kernel) + orange_laplacian[i]
        orange_reconstructed_imgs.append(reconstructed_F)
        start_F = reconstructed_F

    # mask = cv2.imread("/home/colivier/src/laplacian_blend/mask.png")
    # mask = make_channels_first(torch.from_numpy(mask))
    # show("mask", mask)
    # is this w/h backwards?
    # new_mask = torch.zeros((orange.shape[-1], orange.shape[-2]), dtype=torch.uint8)
    # new_mask = torch.zeros((orange.shape[-2], orange.shape[-1]), dtype=torch.uint8)
    # half and half
    # ncols = orange.shape[-1]
    # ncols = orange.shape[-2]
    # new_mask[:, : ncols // 2] = 255
    # new_mask[: ncols // 2, :] = 255
    # mask = new_mask.unsqueeze(0).unsqueeze(0)
    # mask = make_channels_first(mask).repeat(1, 3, 1, 1)
    mask = torch.zeros_like(orange)
    mask[:, :, :, : mask.shape[-1] // 2] = 255.0
    # show("mask", mask[0])

    # I guess here we just blur the mask seam?

    img = mask

    mask_small_gaussian_blurred = []
    for _ in range(levels + 1):
        img = one_level_gaussian_pyramid(img, gaussian_kernel)
        mask_small_gaussian_blurred.append(img)

    img = mask_small_gaussian_blurred[-1]
    show("mask", img[0])

    # TODO: as stacked batch
    for i in range(len(mask_small_gaussian_blurred)):
        mask_small_gaussian_blurred[i] = mask_small_gaussian_blurred[i] / torch.max(
            mask_small_gaussian_blurred[i]
        )
    show("mask_G_small_gaussian_blurred", mask_small_gaussian_blurred[-1][0])

    print("Done.")
