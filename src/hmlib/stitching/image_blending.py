#!/usr/bin/env python
# coding: utf-8

# ### Import dependencies.

# In[54]:


import numpy as np
import matplotlib.pyplot as plt

# from PIL import Image
# from pslib import *
import torchvision as tv
from torchvision.io import read_image
import torch
import torch.nn.functional as F
import cv2
from hmlib.video_out import resize_image

# ### Load provided Gaussian kernel.

# In[55]:


gaussian_kernel = np.load("/home/colivier/src/laplacian_blend/gaussian-kernel.npy")
# gaussian_kernel = np.array(
#     [
#         [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625],
#         [0.015625, 0.0625, 0.09375, 0.0625, 0.015625],
#         [0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375],
#         [0.015625, 0.0625, 0.09375, 0.0625, 0.015625],
#         [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625],
#     ]
# )


# Function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel.
    """
    # Create a coordinate grid
    x = torch.arange(size).float() - size // 2
    y = x.view(size, 1)

    # Calculate the 2D Gaussian kernel
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel (so the sum is 1.0)
    g /= g.sum()

    return g


# Example usage:
kernel_size = 5  # Size of the kernel
sigma = 1.0  # Standard deviation of the Gaussian

# Generate the kernel. The unsqueeze operations are to add the required number of dimensions
# for conv2d, which are batch and channel dimensions.

# TODO: expand in channel dim
gaussian_kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
# gaussian_kernel_3_channels = gaussian_kernel.repeat(1, 3, 1, 1)
gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

# print(gaussian_kernel)

# ### Define helper functions and cross-correlation/convolution function.

# In[56]:


def showable(tensor: torch.Tensor):
    t = (tensor * 255).clamp(min=0, max=255.0).to(torch.uint8)
    # t = tensor.clamp(min=0, max=255.0).to(torch.uint8)
    # t = tensor
    print(torch.min(t))
    print(torch.max(t))
    return t.numpy().transpose(1, 2, 0)


def convolution(img, kernel):
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = img.to(torch.float32)
    # print(torch.min(img))
    # print(torch.max(img))
    # img = img.unsqueeze(0)
    # cv2.imshow("", showable(img[0]))
    # cv2.waitKey(0)
    convolved = F.conv2d(img, kernel, padding=kernel_size // 2, groups=3)
    print(torch.min(img))
    print(torch.max(img))
    # cv2.imshow("", showable(convolved[0]))
    # cv2.waitKey(0)
    return convolved

    # img = img.transpose(1, 2, 0)
    # assert img.shape[-1] == 3
    # MAX_ROWS = img.shape[1]
    # MAX_COLS = img.shape[2]
    # kernel_size = kernel.shape[0]
    # pad_amount = int(kernel_size / 2)
    # gaussian_convolved_img = np.zeros(img.shape)
    # for i in range(3):
    #     # zero_padded = np.pad(img[:,:,i], u_pad=pad_amount, v_pad=pad_amount)
    #     zero_padded = np.pad(
    #         img[:, :, i],
    #         ((0, pad_amount), (0, pad_amount)),
    #         mode="constant",
    #         constant_values=0,
    #     )
    #     # zero_padded = pad(img[:,:,i], u_pad=pad_amount, v_pad=pad_amount)
    #     for r in range(pad_amount, MAX_ROWS + pad_amount):
    #         for c in range(pad_amount, MAX_COLS + pad_amount):
    #             print("r-pad_amount", r-pad_amount)
    #             print("r-pad_amount+kernel_size", r-pad_amount+kernel_size)
    #             conv = np.multiply(
    #                 zero_padded[
    #                     r - pad_amount : r - pad_amount + kernel_size,
    #                     c - pad_amount : c - pad_amount + kernel_size,
    #                 ],
    #                 kernel,
    #             )
    #             conv = np.sum(conv)
    #             gaussian_convolved_img[r - pad_amount, c - pad_amount, i] = float(conv)
    # return gaussian_convolved_img


# In[57]:


# def make_one_D_kernel(kernel):
#     MAX_ROWS = img.shape[0]
#     MAX_COLS = img.shape[1]
#     one_d_gaussian_kernel = kernel

#     kernel_matrix = np.zeros((MAX_ROWS, MAX_ROWS))
#     # print(kernel_matrix.shape)
#     for m in range(MAX_ROWS):
#         #     print(m)
#         #     print(m+(len(one_d_gaussian_kernel)))
#         #     print(one_d_gaussian_kernel)
#         #     print()
#         over = int(len(one_d_gaussian_kernel) / 2)
#         mid = over
#         lower = max(0, m - over)
#         upper = min(m + over, MAX_ROWS)
#         kernel_lower = mid - over if m - over >= 0 else abs(m - over)
#         kernel_upper = (
#             mid + over if m + over < MAX_ROWS else (mid + over) - (m + over - MAX_ROWS)
#         )
#         kernel_matrix[m, lower:upper] = one_d_gaussian_kernel[kernel_lower:kernel_upper]
#     return kernel_matrix


# In[58]:


def down_sample(img, factor=2):
    MAX_ROWS = img.shape[-2]
    MAX_COLS = img.shape[-1]
    assert img.dtype == torch.float32
    img = resize_image(
        img=img,
        new_width=MAX_COLS // 2,
        new_height=MAX_ROWS // 2,
        mode=tv.transforms.InterpolationMode.BILINEAR,
    )
    # small_img = np.zeros((int(MAX_ROWS / 2), int(MAX_COLS / 2), 3))

    # small_img[:, :, 0] = resize(
    #     image=img[:, :, 0], size=[int(MAX_ROWS / 2), int(MAX_COLS / 2)]
    # )
    # small_img[:, :, 1] = resize(
    #     image=img[:, :, 1], size=[int(MAX_ROWS / 2), int(MAX_COLS / 2)]
    # )
    # small_img[:, :, 2] = resize(
    #     image=img[:, :, 2], size=[int(MAX_ROWS / 2), int(MAX_COLS / 2)]
    # )
    return img


# In[59]:


def up_sample(img, factor=2):
    MAX_ROWS = img.shape[-2]
    MAX_COLS = img.shape[-1]
    img = resize_image(
        img=img,
        new_width=int(MAX_COLS * 2),
        new_height=int(MAX_ROWS * 2),
        mode=tv.transforms.InterpolationMode.BILINEAR,
    )
    # small_img = np.zeros((int(MAX_ROWS * 2), int(MAX_COLS * 2), 3))

    # small_img[:, :, 0] = resize(
    #     image=img[:, :, 0], size=[int(MAX_ROWS * 2), int(MAX_COLS * 2)]
    # )
    # small_img[:, :, 1] = resize(
    #     image=img[:, :, 1], size=[int(MAX_ROWS * 2), int(MAX_COLS * 2)]
    # )
    # small_img[:, :, 2] = resize(
    #     image=img[:, :, 2], size=[int(MAX_ROWS * 2), int(MAX_COLS * 2)]
    # )
    # return small_img
    return img


# In[60]:


def one_level_laplacian(img, G):
    # generate Gaussian pyramid for Apple
    A = img.clone()

    # Gaussian blur on Apple
    blurred_A = convolution(A, G)

    # Downsample blurred A
    small_A = down_sample(blurred_A)

    # Upsample small, blurred A
    # insert zeros between pixels, then apply a gaussian low pass filter
    large_A = up_sample(small_A)
    upsampled_A = convolution(large_A, G)

    # generate Laplacian level for A
    laplace_A = A - upsampled_A

    # reconstruct A
    #     reconstruct_A = laplace_A + upsampled_A

    return small_A, upsampled_A, laplace_A


# In[61]:


def F_transform(small_A, G):
    large_A = up_sample(small_A)
    upsampled_A = convolution(large_A, G)
    return upsampled_A


# In[62]:


# def gamma_decode(img):
#     new_img = np.zeros((img.shape))
#     for r in range(img.shape[0]):
#         for c in range(img.shape[1]):
#             new_img[r, c, 0] = np.power(img[r, c, 0], 1 / 1.2)
#             new_img[r, c, 1] = np.power(img[r, c, 1], 1 / 1.2)
#             new_img[r, c, 2] = np.power(img[r, c, 2], 1 / 1.2)
#     return new_img


def gamma_decode(img):
    return img.to(torch.float32) / 255.0


# ## Run Laplacian Pyramid on Apple

# In[63]:


img = torch.from_numpy(
    cv2.imread("/home/colivier/src/laplacian_blend/apple.png")
).unsqueeze(0)
# plt.imshow(img)
# plt.show()
# cv2.imshow("", img.numpy())
# cv2.waitKey(0)

# img = gamma_decode(img) / 255.0
# img /= 255.0

# cv2.imshow("", img.numpy())

# In[64]:


# plt.imshow(img)
# plt.show()


# In[65]:


img.shape


# In[66]:


MAX_ROWS = img.shape[-2]
MAX_COLS = img.shape[-1]
print("MAX_ROWS = ", MAX_ROWS)
print("MAX_COLS = ", MAX_COLS)


# In[67]:


# G = gaussian_kernel


# In[68]:


G = gaussian_kernel
apple_G_small_gaussian_blurred = []
#apple_F_upsampled = []
apple_L_laplace = []

# Load Images
apple = read_image("/home/colivier/src/laplacian_blend/apple.png")
orange = read_image("/home/colivier/src/laplacian_blend/orange.png")
# apple = apple.transpose(1, 2, 0)
# orange = orange.transpose(1, 2, 0)
apple = gamma_decode(apple).unsqueeze(0)
orange = gamma_decode(orange).unsqueeze(0)

img = apple

for i in range(4):
    small_A, upsampled_A, laplace_A = one_level_laplacian(img, G)
    apple_G_small_gaussian_blurred.append(small_A)
    #apple_F_upsampled.append(upsampled_A)
    apple_L_laplace.append(laplace_A)
    img = small_A


# In[ ]:


# reconstruct image
apple_reconstructed_imgs = []
start_F = apple_G_small_gaussian_blurred[-1]

for i in reversed(range(0, 4)):
    #     print(start_F.shape)
    #     print(L_laplace[i].shape)
    reconstructed_F = F_transform(start_F, G) + apple_L_laplace[i]
    #     print(reconstructed_F.shape)
    apple_reconstructed_imgs.append(reconstructed_F)
    start_F = reconstructed_F


# ## Run Laplacian Pyramid on Orange

# In[ ]:


G = gaussian_kernel
orange_G_small_gaussian_blurred = []
#orange_F_upsampled = []
orange_L_laplace = []

# # Load Images
# apple = read_image('ps2/apple.png')
# orange = read_image('ps2/orange.png')
# apple = gamma_decode(apple)
# orange = gamma_decode(orange)

img = orange.clone()

for i in range(4):
    small_A, upsampled_A, laplace_A = one_level_laplacian(img, G)
    orange_G_small_gaussian_blurred.append(small_A)
    #orange_F_upsampled.append(upsampled_A)
    orange_L_laplace.append(laplace_A)
    img = small_A


# In[ ]:


# reconstruct image
orange_reconstructed_imgs = []
start_F = orange_G_small_gaussian_blurred[-1]

for i in reversed(range(0, 4)):
    #     print()
    #     print(start_F.shape)
    #     print(orange_L_laplace[i].shape)
    reconstructed_F = F_transform(start_F, G) + orange_L_laplace[i]
    #     print(reconstructed_F.shape)
    orange_reconstructed_imgs.append(reconstructed_F)
    start_F = reconstructed_F


# ## Run Gaussian Pyramid on Mask.png

# In[ ]:


# mask = read_image("/home/colivier/src/laplacian_blend/mask.png")
mask = cv2.imread("/home/colivier/src/laplacian_blend/mask.png")
mask = torch.from_numpy(mask).permute(2, 1, 0)


# In[ ]:


new_mask = torch.zeros((orange.shape[-1], orange.shape[-2]), dtype=torch.uint8)


# In[ ]:


ncols = orange.shape[-1]


# In[ ]:


new_mask[:, : int(ncols / 2)] = 255


# In[ ]:


mask = new_mask


# In[ ]:


# plt.imshow(mask)
# plt.show()


# In[ ]:


mask.shape


# In[ ]:


def one_layer_convolution(img, kernel):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    kernel_size = kernel.shape[0]
    #     pad_amount = int(kernel_size/2)
    gaussian_convolved_img = np.zeros(img.shape, dtype="uint8")
    #     zero_padded = np.pad(img[:,:], pad_amount, pad_with, padder=0)
    for r in range(0, MAX_ROWS):
        for c in range(0, MAX_COLS):
            #             print("r-pad_amount", r-pad_amount)
            #             print("r-pad_amount+kernel_size", r-pad_amount+kernel_size)
            kernel_r_upper = (
                kernel_size
                if r + kernel_size < MAX_ROWS
                else MAX_ROWS - (r + kernel_size) + 1
            )
            kernel_c_upper = (
                kernel_size
                if c + kernel_size < MAX_COLS
                else MAX_COLS - (c + kernel_size) + 1
            )

            conv = np.multiply(
                img[r : r + kernel_size, c : c + kernel_size],
                kernel[0:kernel_r_upper, 0:kernel_c_upper],
            )
            conv = np.sum(conv)
            gaussian_convolved_img[r, c] = float(conv)
    return gaussian_convolved_img


# In[ ]:


def one_layer_convolution(img, kernel):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    kernel_size = kernel.shape[0]
    #     pad_amount = int(kernel_size/2)
    gaussian_convolved_img = np.zeros(img.shape)
    #     zero_padded = np.pad(img[:,:], pad_amount, pad_with, padder=0)
    for r in range(0, MAX_ROWS):
        for c in range(0, MAX_COLS):
            #             print("r-pad_amount", r-pad_amount)
            #             print("r-pad_amount+kernel_size", r-pad_amount+kernel_size)
            kernel_r_upper = (
                kernel_size
                if r + kernel_size < MAX_ROWS
                else MAX_ROWS - (r + kernel_size) + 1
            )
            kernel_c_upper = (
                kernel_size
                if c + kernel_size < MAX_COLS
                else MAX_COLS - (c + kernel_size) + 1
            )

            conv = np.multiply(
                img[r : r + kernel_size, c : c + kernel_size],
                kernel[0:kernel_r_upper, 0:kernel_c_upper],
            )
            conv = np.sum(conv)
            gaussian_convolved_img[r, c] = float(conv)
    return gaussian_convolved_img


# In[ ]:


def one_layer_convolution(img, kernel):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    kernel_size = kernel.shape[0]
    #     pad_amount = int(kernel_size/2)
    gaussian_convolved_img = torch.zeros(img.shape)
    #     zero_padded = np.pad(img[:,:], pad_amount, pad_with, padder=0)
    for r in range(0, MAX_ROWS):
        for c in range(0, MAX_COLS):
            #             print("r-pad_amount", r-pad_amount)
            #             print("r-pad_amount+kernel_size", r-pad_amount+kernel_size)
            kernel_r_upper = (
                kernel_size
                if r + kernel_size <= MAX_ROWS
                else kernel_size - ((r + kernel_size) - MAX_ROWS)
            )
            kernel_c_upper = (
                kernel_size
                if c + kernel_size <= MAX_COLS
                else kernel_size - ((c + kernel_size) - MAX_COLS)
            )
            new_kernel = kernel[0:kernel_r_upper, 0:kernel_c_upper] / torch.sum(
                kernel[0:kernel_r_upper, 0:kernel_c_upper]
            )
            the_slice = img[
                r : min(MAX_ROWS, r + kernel_size),
                c : min(MAX_COLS, c + kernel_size),
            ]
            kern = new_kernel[
                    : min(MAX_ROWS, r + kernel_size), : min(MAX_COLS, c + kernel_size)
                ]
            conv = the_slice * kern
            # conv = torch.multiply(
            #     img[
            #         r : min(MAX_ROWS, r + kernel_size),
            #         c : min(MAX_COLS, c + kernel_size),
            #     ],
            #     new_kernel,
            # )
            conv = torch.sum(conv)
            gaussian_convolved_img[r, c] = float(conv)
    return gaussian_convolved_img


# In[ ]:


def one_layer_down_sample(img, factor=2):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    small_img = np.zeros((int(MAX_ROWS / 2), int(MAX_COLS / 2)))

    small_img[:, :] = resize(
        image=img[:, :], size=[int(MAX_ROWS / 2), int(MAX_COLS / 2)]
    )

    return small_img


# In[ ]:


def one_level_gaussian_pyramid(img, G):
    # generate Gaussian pyramid for img
    A = img.clone()

    # Gaussian blur on img
    blurred_A = one_layer_convolution(A, G)

    # Downsample blurred A
    small_A = one_layer_down_sample(blurred_A)

    return small_A


# In[ ]:


G = gaussian_kernel
mask_G_small_gaussian_blurred = [mask]
# F_upsampled = []
# L_laplace = []

# Load Images


img = mask.clone()

for i in range(5):
    small_A = one_level_gaussian_pyramid(img, G)
    mask_G_small_gaussian_blurred.append(small_A)
    #     F_upsampled.append(upsampled_A)
    #     L_laplace.append(laplace_A)
    img = small_A


# In[ ]:


for i in range(len(mask_G_small_gaussian_blurred)):
    mask_G_small_gaussian_blurred[i] = mask_G_small_gaussian_blurred[i] / np.max(
        mask_G_small_gaussian_blurred[i]
    )


# In[ ]:


np.max(mask_G_small_gaussian_blurred[-1])


# In[ ]:


plt.imshow(mask_G_small_gaussian_blurred[-1])


# ## Time for Laplacian blending!

# In[ ]:


# orange_G_small_gaussian_blurred = []
# orange_F_upsampled = []
# orange_L_laplace = []


# In[ ]:


for i in mask_G_small_gaussian_blurred:
    print(i.shape)


# In[ ]:


for i in orange_G_small_gaussian_blurred:
    print(i.shape)


# In[ ]:


for i in orange_L_laplace:
    print(i.shape)


# In[ ]:


mask_apple_1d = mask_G_small_gaussian_blurred[-2]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[-2]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)

G_c = (
    apple_G_small_gaussian_blurred[-1] * mask_apple
    + orange_G_small_gaussian_blurred[-1] * mask_orange
)


# In[ ]:


F_1 = up_sample(G_c)
upsampled_F1 = convolution(F_1, G)


# In[ ]:


mask_apple_1d = mask_G_small_gaussian_blurred[-3]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[-3]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)


# In[ ]:


L_a = apple_L_laplace[-1]
L_o = orange_L_laplace[-1]


# In[ ]:


plt.imshow(mask_apple)


# In[ ]:


plt.imshow(mask_orange)


# In[ ]:


L_c = (mask_apple * L_a) + (mask_orange * L_o)


# In[ ]:


plt.imshow(L_c)


# In[ ]:


F_2 = L_c + upsampled_F1


# In[ ]:


upsampled_F1.shape


# In[ ]:


F1_plot = np.array(upsampled_F1)
plt.imshow(F1_plot)


# In[ ]:


new_L_c = L_c + (1 - L_c)


# In[ ]:


plt.imshow(L_c)


# In[ ]:


F_2 = L_c + upsampled_F1


# In[ ]:


F2_plot = np.array(F_2)
plt.imshow(F2_plot)


# In[ ]:


reconstructed_imgs = []
mask_apple_1d = mask_G_small_gaussian_blurred[4]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[4]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)

G_c = (
    apple_G_small_gaussian_blurred[3] * mask_apple
    + orange_G_small_gaussian_blurred[3] * mask_orange
)

F_1 = up_sample(G_c)
upsampled_F1 = convolution(F_1, G)

mask_apple_1d = mask_G_small_gaussian_blurred[3]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[3]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)

L_a = apple_L_laplace[3]
L_o = orange_L_laplace[3]

L_c = (mask_apple * L_a) + (mask_orange * L_o)
F_2 = L_c + upsampled_F1
reconstructed_imgs.append(F_2)


# In[ ]:


# F_1 = F_2
F_1 = up_sample(F_2)
upsampled_F1 = convolution(F_1, G)

mask_apple_1d = mask_G_small_gaussian_blurred[2]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[2]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)
print(mask_apple.shape)

L_a = apple_L_laplace[2]
L_o = orange_L_laplace[2]

L_c = (mask_apple * L_a) + (mask_orange * L_o)
print(upsampled_F1.shape)
F_2 = L_c + upsampled_F1
reconstructed_imgs.append(F_2)


# In[ ]:


plt.imshow(F_2)


# In[ ]:


# F_1 = F_2
F_1 = up_sample(F_2)
upsampled_F1 = convolution(F_1, G)

mask_apple_1d = mask_G_small_gaussian_blurred[1]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[1]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)
print(mask_apple.shape)

L_a = apple_L_laplace[1]
L_o = orange_L_laplace[1]

L_c = (mask_apple * L_a) + (mask_orange * L_o)
print(upsampled_F1.shape)
F_2 = L_c + upsampled_F1
reconstructed_imgs.append(F_2)


# In[ ]:


plt.imshow(F_2)


# In[ ]:


# F_1 = F_2
F_1 = up_sample(F_2)
upsampled_F1 = convolution(F_1, G)

mask_apple_1d = mask_G_small_gaussian_blurred[0]
mask_orange_1d = 1 - mask_G_small_gaussian_blurred[0]

mask_apple = np.stack(np.array([mask_apple_1d, mask_apple_1d, mask_apple_1d]), axis=2)
mask_orange = np.stack(
    np.array([mask_orange_1d, mask_orange_1d, mask_orange_1d]), axis=2
)
print(mask_apple.shape)

L_a = apple_L_laplace[0]
L_o = orange_L_laplace[0]

L_c = (mask_apple * L_a) + (mask_orange * L_o)
print(upsampled_F1.shape)
F_2 = L_c + upsampled_F1
reconstructed_imgs.append(F_2)


# In[ ]:


plt.imshow(F_2)


# In[ ]:


write_image(path="apple-and-orange.png", image=F_2)


# In[ ]:
