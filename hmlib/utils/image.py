# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import random
from typing import Dict, List, Set, Union

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as TF
import torchvision as tv
from torchvision.transforms import functional as F

from hmlib.log import logger
from hmlib.utils.gpu import StreamTensorBase


def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def pt_transform_preds(coords, center, scale, output_size, trans):
    target_coords = torch.zeros_like(coords)
    zero = torch.tensor(0, dtype=torch.float, device=target_coords.device)
    if trans is None:
        trans = get_affine_transform(
            center.cpu().numpy(),
            scale.cpu().numpy(),
            zero.cpu().numpy(),
            output_size.cpu().numpy(),
            inv=1,
        )
        trans = torch.from_numpy(trans).to(torch.float).to(coords.device)
    # for p in range(coords.shape[0]):
    #     target_coords[p, 0:2] = pt_affine_transform(coords[p, 0:2], trans)
    target_c = all_dets_pt_affine_transform(coords[:, 0:2], trans)
    target_coords[:, 0:2] = target_c
    return target_coords, trans


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def pt_cv2_get_affine_transform(src, dst):
    """
    src, dst: torch.Tensor
        Source and destination points, shape (3, 2)
    """
    ones = torch.ones(3, dtype=torch.float, device=src.device)
    src_h = torch.cat([src, ones.unsqueeze(1)], dim=1)  # Convert to homogeneous coordinates
    dst_h = dst

    # Solve the system of linear equations
    transform_matrix = torch.mm(dst_h.t(), torch.inverse(src_h.t()))

    return transform_matrix


def pt_get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = torch.tensor([scale, scale], dtype=torch.float, device=scale.device)
    if isinstance(shift, np.ndarray):
        shift = torch.from_numpy(shift).to(device=center.device)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = rot * np.pi / 180
    src_dir = pt_get_dir(torch.tensor([0.0, src_w * -0.5], device=src_w.device), rot_rad)
    dst_dir = torch.tensor([0, dst_w * -0.5], dtype=torch.float, device=dst_w.device)

    src = torch.zeros((3, 2), dtype=torch.float, device=center.device)
    dst = torch.zeros((3, 2), dtype=torch.float, device=center.device)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = torch.tensor([dst_w * 0.5, dst_h * 0.5], device=dst_w.device)
    dst[1, :] = (
        torch.tensor([dst_w * 0.5, dst_h * 0.5], dtype=torch.float, device=dst_w.device) + dst_dir
    )

    src[2:, :] = pt_get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = pt_get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = pt_cv2_get_affine_transform(dst.to(torch.float), src.to(torch.float))
    else:
        trans = pt_cv2_get_affine_transform(src.to(torch.float), dst.to(torch.float))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def pt_affine_transform(pt, t):
    new_pt = torch.tensor([pt[0], pt[1], 1.0], dtype=torch.float, device=pt.device).T
    new_pt = torch.matmul(t, new_pt)
    return new_pt[:2]


def all_dets_pt_affine_transform(pt, t):
    bs = pt.shape[0]
    ones = torch.ones((bs, 1), dtype=pt.dtype, device=pt.device)
    new_pt = torch.cat((pt, ones), dim=1).unsqueeze(2)
    t = t.unsqueeze(0).repeat(bs, 1, 1)
    result = torch.bmm(t, new_pt)
    return result.squeeze(2)[:, :2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def pt_get_3rd_point(a, b):
    direct = a - b
    return b + torch.tensor([-direct[1], direct[0]], dtype=torch.float, device=direct.device)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def pt_get_dir(src_point, rot_rad):
    sn, cs = torch.sin(rot_rad), torch.cos(rot_rad)

    src_result = torch.tensor(
        [src_point[0] * cs - src_point[1] * sn, src_point[0] * sn + src_point[1] * cs],
        device=src_point.device,
    )

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])), flags=cv2.INTER_LINEAR
    )

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_regmap = regmap[:, y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    masked_reg = reg[:, radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1]
        )
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top : y + bottom, x - left : x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= 1 - alpha
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


class ImageColorScaler:
    def __init__(self, image_channel_adjustment: List[float]):
        self._image_channel_adjustment = image_channel_adjustment
        self._scale_color_tensor = None

    def maybe_scale_image_colors(self, image: torch.Tensor):
        if not self._image_channel_adjustment:
            return image
        if self._scale_color_tensor is None:
            if isinstance(image, torch.Tensor):
                self._scale_color_tensor = torch.tensor(
                    self._image_channel_adjustment,
                    dtype=torch.float,
                    device=image.device,
                )
                self._scale_color_tensor = self._scale_color_tensor.view(1, 1, 3)
            else:
                self._scale_color_tensor = np.array(
                    self._image_channel_adjustment, dtype=np.float32
                )
                self._scale_color_tensor = np.expand_dims(
                    np.expand_dims(self._scale_color_tensor, 0), 0
                )
        if isinstance(image, torch.Tensor):
            image = torch.clamp(
                image.to(torch.float) * self._scale_color_tensor, min=0, max=255.0
            ).to(torch.uint8)
        else:
            image = np.clip(
                image.astype(np.float32) * self._scale_color_tensor,
                a_min=0,
                a_max=255.0,
            ).astype(np.uint8)
        return image


def _permute(t, *args):
    if isinstance(t, np.ndarray):
        return t.transpose(*args)
    return t.permute(*args)


def make_channels_first(img: torch.Tensor):
    if len(img.shape) == 4:
        if img.shape[-1] in [1, 3, 4]:
            return _permute(img, 0, 3, 1, 2)
    else:
        assert len(img.shape) == 3
        if img.shape[-1] in [1, 3, 4]:
            return _permute(img, 2, 0, 1)
    return img


def make_channels_last(
    img: Union[torch.Tensor, StreamTensorBase],
) -> Union[torch.Tensor, StreamTensorBase]:
    if len(img.shape) == 4:
        if img.shape[1] in [1, 3, 4]:
            return _permute(img, 0, 2, 3, 1)
    else:
        assert len(img.shape) == 3
        if img.shape[0] in [1, 3, 4]:
            return _permute(img, 1, 2, 0)
    return img


def is_channels_first(img: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> bool:
    if len(img.shape) == 4:
        return img.shape[1] in [1, 3, 4]
    else:
        assert len(img.shape) == 3
        return img.shape[0] in [1, 3, 4]


def is_channels_last(img: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> bool:
    return img.shape[-1] in [1, 3, 4]


def image_width(img: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> int:
    if img.ndim == 2:
        return img.shape[1]
    if isinstance(img, (torch.Tensor, StreamTensorBase)):
        if img.ndim == 4:
            if img.shape[-1] in [1, 3, 4]:
                return img.shape[-2]
            else:
                assert img.shape[1] in [1, 3, 4]
                return img.shape[-1]
        elif img.ndim == 2:
            return img.shape[1]
        else:
            assert img.ndim == 3
            if img.shape[-1] in [1, 3, 4]:
                return img.shape[-2]
            else:
                assert img.shape[0] in [1, 3, 4]
                return img.shape[-1]
    assert img.shape[-1] in [1, 3, 4]
    if len(img.shape) == 4:
        return img.shape[2]
    return img.shape[1]


def jittable_image_width(img: torch.Tensor) -> int:
    if img.ndim == 2:
        return img.shape[1]
    assert isinstance(img, torch.Tensor)
    if img.ndim == 4:
        if img.shape[-1] in [1, 3, 4]:
            return img.shape[-2]
        else:
            assert img.shape[1] in [1, 3, 4]
            return img.shape[-1]
    elif img.ndim == 2:
        return img.shape[1]
    else:
        assert img.ndim == 3
        if img.shape[-1] in [1, 3, 4]:
            return img.shape[-2]
        else:
            assert img.shape[0] in [1, 3, 4]
            return img.shape[-1]


def image_height(img: torch.Tensor | StreamTensorBase | np.ndarray) -> int:
    if img.ndim == 2:
        return img.shape[0]
    if isinstance(img, (torch.Tensor, StreamTensorBase)):
        if img.ndim == 4:
            if img.shape[-1] in [1, 3, 4]:
                return img.shape[-3]
            else:
                assert img.shape[1] in [1, 3, 4]
                return img.shape[-2]
        elif img.ndim == 2:
            return img.shape[0]
        else:
            assert img.ndim == 3
            if img.shape[-1] in [1, 3, 4]:
                return img.shape[-3]
            else:
                assert img.shape[0] in [1, 3, 4]
                return img.shape[-2]
    assert img.shape[-1] in [1, 3, 4]
    if len(img.shape) == 4:
        return img.shape[1]
    return img.shape[0]


def jittable_image_height(img: torch.Tensor) -> int:
    if img.ndim == 2:
        return img.shape[0]
    assert isinstance(img, torch.Tensor)
    if img.ndim == 4:
        if img.shape[-1] in [1, 3, 4]:
            return img.shape[-3]
        else:
            assert img.shape[1] in [1, 3, 4]
            return img.shape[-2]
    elif img.ndim == 2:
        return img.shape[0]
    else:
        assert img.ndim == 3
        if img.shape[-1] in [1, 3, 4]:
            return img.shape[-3]
        else:
            assert img.shape[0] in [1, 3, 4]
            return img.shape[-2]


def crop_image(img, left, top, right, bottom):
    if isinstance(img, PIL.Image.Image):
        return img.crop((left, top, right, bottom))
    assert img.ndim == 3 and img.shape[-1] in [3, 4]
    return img[top:bottom, left:right, :]


def get_best_resize_mode(
    w1: int, h1: int, w2: int, h2: int, interpolate: bool = False, verbose: bool = True
) -> Union[int, str]:
    if h2 & 1 != 0 and w2 & 1 != 0:
        print(f"Why are you resizing to odd dimensions? {w1}x{h1} -> {w2}x{h2}")
    if w1 > w2:
        # Just a sanity check assumign we aren't
        # purposely trying to distort
        assert h1 > h2 or abs(h2 - h1) < 1.1
        if h1 == h2 and abs(w2 - w1) < 1.1:
            if verbose:
                # Maybe you have a one-off match error somewhere
                # causing an expensive resize?
                logger.warning(f"PERF WARNING: Almost trival resize from {w1}x{h1} -> {w2}x{h2}")
        # Downsampling
        # return F.InterpolationMode.BOX
        return "area"
    elif w2 > w1:
        # Just a sanity check assumign we aren't
        # purposely trying to distort
        assert h2 > h1 or abs(h2 - h1) < 1.1
        if h1 == h2 and abs(w2 - w1) == 1.1:
            if verbose:
                # Maybe you have a one-off match error somewhere
                # causing an expensive resize?
                logger.warning(f"PERF WARNING: Almost trivial resize from {w1}x{h1} -> {w2}x{h2}")
        # Upsampling
        return "bilinear"
    elif w1 == w2:
        # Just a sanity check assumign we aren't
        # purposely trying to distort
        # In order to have even height/width, we may have a 1 pixel difference
        assert abs(h1 - h2) <= 1
        return None
    # Make sure we're resizing to even dimensions
    assert False and "Should not get here"
    return "bilinear"


_CONVERT_MODE: Dict[int, str] = {
    F.InterpolationMode.NEAREST: "nearest",
    F.InterpolationMode.BILINEAR: "bilinear",
    F.InterpolationMode.BICUBIC: "bicubic",
}
_CORNER_ALIGNABLE_MODES: Set[str] = {"linear", "bilinear", "bicubic", "trilinear"}


def _allow_align_corners(mode: str, align_corners: Union[bool, None]) -> Union[bool, None]:
    if mode not in _CORNER_ALIGNABLE_MODES:
        return None
    return align_corners


def resize_mode_to_str_mode(mode: Union[str, int]) -> str:
    if isinstance(mode, str):
        return mode
    return _CONVERT_MODE[mode]


def resize_image(
    img,
    new_width: int,
    new_height: int,
    mode: str = None,
    antialias: bool = True,
    float_dtype: torch.dtype = torch.float,
):
    w = int(new_width)
    h = int(new_height)
    if isinstance(img, torch.Tensor):
        was_channels_last = is_channels_last(img)
        if was_channels_last:
            img = make_channels_first(img)
        if mode is None:
            # We know it's channels-first by now, so last two size items are H, W
            mode = get_best_resize_mode(w1=img.shape[-1], h1=img.shape[-2], w2=w, h2=h)
        if mode is not None:
            if True:
                # use interpolate, change to float if necessary
                if not torch.is_floating_point(img):
                    img = img.to(dtype=float_dtype, non_blocking=True)
                mode = resize_mode_to_str_mode(mode)
                # TF.interpolate wants a batch dimension
                was_batched = img.ndim == 4
                if not was_batched:
                    img = img.unsqueeze(0)
                img = TF.interpolate(
                    img, size=(h, w), mode=mode, align_corners=_allow_align_corners(mode, False)
                )
                if not was_batched:
                    img = img.squeeze(0)
                # Assert that it reshaped as we expected
                assert img.shape[-2] == h and img.shape[-1] == w
            else:
                img = F.resize(
                    img=img,
                    size=(h, w),
                    interpolation=mode,
                    antialias=antialias,
                )
        if was_channels_last:
            img = make_channels_last(img)
        return img
    elif isinstance(img, PIL.Image.Image):
        return img.resize((w, h))
    return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)


# Function to pad tensor to the target size
def pad_tensor_to_size(tensor, target_width, target_height, pad_value):
    if len(tensor.shape) == 2:
        pad_height = target_height - tensor.size(0)
        pad_width = target_width - tensor.size(1)
    else:
        assert len(tensor.shape) == 3
        pad_height = target_height - tensor.size(1)
        pad_width = target_width - tensor.size(2)
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)
    if pad_height == 0 and pad_width == 0:
        return tensor
    padding = [0, pad_width, 0, pad_height]
    padded_tensor = TF.pad(tensor, padding, "constant", pad_value)
    return padded_tensor


def pad_tensor_to_size_batched(
    tensor: torch.Tensor, target_width: int, target_height: int, pad_value: Union[int, float]
) -> torch.Tensor:
    pad_height = target_height - jittable_image_height(tensor)
    pad_width = target_width - jittable_image_width(tensor)
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)
    if pad_height == 0 and pad_width == 0:
        return tensor
    padding = [0, pad_width, 0, pad_height]
    if pad_value is None:
        padded_tensor = TF.pad(tensor, padding, "replicate")
    else:
        padded_tensor = TF.pad(tensor, padding, "constant", float(pad_value))
    return padded_tensor


def make_showable_type(
    img: Union[torch.Tensor, np.ndarray],
    scale_elements: Union[float, None] = None,
    force_numpy: bool = False,
):
    if isinstance(img, torch.Tensor):
        if img.ndim == 2:
            # 2D grayscale
            img = img.unsqueeze(0).repeat(3, 1, 1)
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        assert len(img.shape) == 3
        img = make_channels_last(img)
        if img.dtype in [torch.float16, torch.bfloat16, torch.float, torch.float64]:
            # if img.dtype == torch.float16 or img.dtype == torch.bfloat16:
            #     img = img.to(torch.float32)
            if scale_elements and scale_elements != 1:
                img = img * scale_elements
            img = torch.clamp(img, min=0, max=255.0).to(torch.uint8)
        elif img.dtype == torch.bool:
            img = img.to(torch.uint8) * 255
        if force_numpy or img.device.type != "cuda":
            img = np.ascontiguousarray(img.cpu().numpy())
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            # 2D grayscale
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]
        assert len(img.shape) == 3
        if img.shape[-1] not in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            if scale_elements and scale_elements != 1:
                img = img * scale_elements
            img = np.clip(img, a_min=0, a_max=255.0).astype(np.uint8)
    return img


def make_visible_image(
    img,
    enable_resizing: Union[bool, float] = None,
    scale_elements: Union[float, None] = None,
    force_numpy=False,
):
    if enable_resizing is None:
        return make_showable_type(img, scale_elements, force_numpy=force_numpy)
    width = image_width(img)
    if enable_resizing != 0:
        vis_w = width * enable_resizing
        mult = 1.0
    else:
        vis_w = get_complete_monitor_width()
        mult = 0.7
    if isinstance(img, torch.Tensor) and img.dtype == torch.bool:
        img = img.to(torch.uint8) * 255
    if vis_w and width and width > vis_w:
        height = image_height(img)
        ar = width / height
        new_w = vis_w * mult
        new_h = new_w / ar
        img = resize_image(img, new_width=int(new_w), new_height=int(new_h))
    return make_showable_type(img, force_numpy=force_numpy)


def get_complete_monitor_width():
    width = 0
    from screeninfo import get_monitors

    for monitor in get_monitors():
        width += monitor.width
    return width


def to_float_image(
    tensor: torch.Tensor,
    apply_scale: bool = False,
    non_blocking: bool = False,
    dtype: torch.dtype = torch.float,
):
    assert not apply_scale
    if tensor.dtype == torch.uint8:
        if apply_scale:
            assert False
            return tensor.to(dtype, non_blocking=non_blocking) / 255.0
        else:
            return tensor.to(dtype, non_blocking=non_blocking)
    else:
        assert torch.is_floating_point(tensor)
    return tensor


def to_uint8_image(tensor: torch.Tensor, apply_scale: bool = False):
    assert not apply_scale
    if isinstance(tensor, np.ndarray):
        assert tensor.dtype == np.uint8
        return tensor
    if tensor.dtype != torch.uint8:
        if apply_scale:
            assert False
            assert torch.is_floating_point(tensor)
            return (
                # note, no scale applied here (I removed before adding assert)
                tensor.clamp(min=0, max=255.0).to(torch.uint8)
            )
        else:
            # There has got to be a more elegant way to do this with reflection
            def _clamp(t, *args, **kwargs):
                return t.clamp(*args, **kwargs).to(torch.uint8)

            if isinstance(tensor, StreamTensorBase):
                tensor = tensor.wait()
            return _clamp(tensor, min=0, max=255.0)
    return tensor


def rotate_image(img, angle: float, rotation_point: List[int]):
    rotation_point = [int(i) for i in rotation_point]
    if isinstance(img, torch.Tensor):
        current_dtype = img.dtype
        if img.dim() == 4:
            # H, W, C -> C, W, H
            img = img.permute(0, 3, 2, 1)
            angle = -angle
            if current_dtype == torch.half:
                img = img.to(torch.float32, non_blocking=True)
            img = F.rotate(
                img=img,
                angle=angle,
                center=(rotation_point[1], rotation_point[0]),
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=None,
            )
            # W, H, C -> C, H, W
            img = img.permute(0, 3, 2, 1)
        else:
            # H, W, C -> C, W, H
            img = img.permute(2, 1, 0)
            angle = -angle
            if current_dtype == torch.half:
                img = img.to(torch.float32, non_blocking=True)
            img = F.rotate(
                img=img,
                angle=angle,
                center=(rotation_point[1], rotation_point[0]),
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=None,
            )
            # W, H, C -> C, H, W
            img = img.permute(2, 1, 0)
    elif isinstance(img, PIL.Image.Image):
        img = img.rotate(
            angle, resample=PIL.Image.BICUBIC, center=(rotation_point[0], rotation_point[1])
        )
    else:
        rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (image_width(img), image_height(img)))
    return img
