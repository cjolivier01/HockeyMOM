from typing import List

import cv2
import torch.nn.functional as F


def calculate_letterbox(shape: List[int], height: int, width: int):
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    return new_shape, ratio, dw, dh


def letterbox(img, height, width, color=(127.5, 127.5, 127.5)):
    new_shape, ratio, dw, dh = calculate_letterbox(shape=img.shape[:2], height=height, width=width)

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized_img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    letterbox_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return letterbox_img, resized_img, ratio, dw, dh


def py_calculate_letterbox(shape: List[int], height: int, width: int):
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    return new_shape, ratio, dw, dh


def copy_make_border_pytorch_batch(images, top, bottom, left, right, border_type, value):
    """
    Mimics cv2.copyMakeBorder function in PyTorch for a batch of images.

    Args:
    - images (Tensor): The input batch of image tensors of shape (B, C, H, W).
    - top, bottom, left, right (int): The number of pixels to add in each direction.
    - border_type (str): Type of border ('constant' or 'replicate').
    - value (int or tuple): Value for constant border; ignored for replicate border.

    Returns:
    - Tensor: Padded batch of image tensors.
    """
    if border_type == "constant":
        # Adding constant border
        padding = (left, right, top, bottom)
        return F.pad(images, padding, "constant", value)

    elif border_type == "replicate":
        # Adding replicated border
        padding = (left, right, top, bottom)
        return F.pad(images, padding, "replicate")

    else:
        raise ValueError("Unsupported border type. Use 'constant' or 'replicate'.")


def py_letterbox(img, height, width, color=0.5):
    new_shape, ratio, dw, dh = py_calculate_letterbox(
        shape=img.shape[2:], height=height, width=width
    )

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized_image = F.interpolate(img, size=(new_shape[1], new_shape[0]), mode="area")
    letterbox_img = copy_make_border_pytorch_batch(
        resized_image, top, bottom, left, right, "constant", value=color
    )
    return letterbox_img, resized_image, ratio, dw, dh
