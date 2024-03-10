import numbers
from typing import List, Optional, Tuple, Union, no_type_check
import warnings
import torch
from torchvision.transforms import functional as F

import cv2
import numpy as np
import mmcv
from hmlib.builder import PIPELINES
from hmlib.utils.image import image_width, image_height, resize_image


try:
    from PIL import Image
except ImportError:
    Image = None


cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}

# Pillow >=v9.1.0 use a slightly different naming scheme for filters.
# Set pillow_interp_codes according to the naming scheme used.
if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }


def hm_imrescale(
    img: np.ndarray,
    scale: Union[float, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray, Tuple[Union[np.ndarray, torch.Tensor], float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    if img.ndim == 3:
        h, w = img.shape[:2]
    else:
        assert img.ndim == 4 and img.shape[-1] in (3, 4)
        h, w = img.shape[1:3]
    new_size, scale_factor = mmcv.rescale_size((w, h), scale, return_scale=True)
    rescaled_img = hm_imresize(
        img, new_size, interpolation=interpolation, backend=backend
    )
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def hm_imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = mmcv.imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    elif isinstance(img, torch.Tensor):
        w = size[0]
        h = size[1]
        if img.dim() == 4:
            # Probably doesn't work
            permuted = img.shape[-1] == 3 or img.shape[-1] == 4
            if permuted:
                # H, W, C -> C, W, H
                img = img.permute(0, 3, 2, 1)
            assert img.shape[1] == 3 or img.shape[1] == 4
            resized_img = F.resize(
                img=img,
                size=(w, h) if permuted else (h, w),
                interpolation=pillow_interp_codes[interpolation],
                antialias=True,
            )
            if permuted:
                # C, W, H -> H, W, C
                resized_img = resized_img.permute(0, 3, 2, 1)
        else:
            permuted = img.shape[-1] == 3 or img.shape[-1] == 4
            if permuted:
                # H, W, C -> C, W, H
                img = img.permute(2, 1, 0)
            resized_img = F.resize(
                img=img,
                size=(w, h) if permuted else (h, w),
                interpolation=pillow_interp_codes[interpolation],
                antialias=True,
            )
            if permuted:
                # C, W, H -> H, W, C
                resized_img = resized_img.permute(2, 1, 0)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
        )
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def hm_imresize_to_multiple(
    img: np.ndarray,
    divisor: Union[int, Tuple[int, int]],
    size: Union[int, Tuple[int, int], None] = None,
    scale_factor: Union[float, Tuple[float, float], None] = None,
    keep_ratio: bool = False,
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image according to a given size or scale factor and then rounds
    up the the resized or rescaled image size to the nearest value that can be
    divided by the divisor.

    Args:
        img (ndarray): The input image.
        divisor (int | tuple): Resized image size will be a multiple of
            divisor. If divisor is a tuple, divisor should be
            (w_divisor, h_divisor).
        size (None | int | tuple[int]): Target size (w, h). Default: None.
        scale_factor (None | float | tuple[float]): Multiplier for spatial
            size. Should match input size if it is a tuple and the 2D style is
            (w_scale_factor, h_scale_factor). Default: None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: False.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = image_height(img), image_width(img)
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is None and scale_factor is None:
        raise ValueError("one of size or scale_factor should be defined")
    elif size is not None:
        size = to_2tuple(size)
        if keep_ratio:
            size = mmcv.rescale_size((w, h), size, return_scale=False)
    else:
        size = _scale_size((w, h), scale_factor)

    divisor = to_2tuple(divisor)
    size = tuple(int(np.ceil(s / d)) * d for s, d in zip(size, divisor))
    resized_img, w_scale, h_scale = hm_imresize(
        img,
        size,
        return_scale=True,
        interpolation=interpolation,
        out=out,
        backend=backend,
    )
    if return_scale:
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def hm_impad(
    img: np.ndarray,
    *,
    shape: Optional[Tuple[int, int]] = None,
    padding: Union[int, tuple, None] = None,
    pad_val: Union[float, List] = 0,
    padding_mode: str = "constant",
) -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - image_width(img), 0)
        height = max(shape[0] - image_height(img), 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(
            "pad_val must be a int or a tuple. " f"But received {type(pad_val)}"
        )

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(
            "Padding must be a int or a 2, or 4 element tuple."
            f"But received {padding}"
        )

    # check padding mode
    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    if isinstance(img, torch.Tensor):
        if img.ndim == 3:
            img = img.permute(2, 0, 1)
        else:
            assert img.ndim == 4
            if img.shape[-1] in (3, 4):
                img = img.permute(0, 3, 1, 2)
        img = torch.nn.functional.pad(
            img,
            (padding[0], padding[2], padding[1], padding[3]),
            padding_mode,
            value=pad_val,
        )
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        else:
            img = img.permute(0, 2, 3, 1)
    else:
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val,
        )

    return img


def hm_impad_to_multiple(
    img: np.ndarray, divisor: int, pad_val: Union[float, List] = 0
) -> np.ndarray:
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(image_height(img) / divisor)) * divisor
    pad_w = int(np.ceil(image_width(img) / divisor)) * divisor
    return hm_impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


@PIPELINES.register_module()
class HmImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys, scale_factor=None):
        self.keys = keys
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        permute the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and permuted to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if isinstance(img, torch.Tensor) and not torch.is_floating_point(img):
                if len(img.shape) < 3:
                    img = img.unsqueeze(0)
                assert img.dtype == torch.uint8
                results[key] = img.to(torch.float, non_blocking=True)
                if self.scale_factor is not None:
                    results[key] *= self.scale_factor
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"


@PIPELINES.register_module()
class HmPad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(
        self,
        size=None,
        size_divisor=None,
        pad_to_square=False,
        pad_val=dict(img=0, masks=0, seg=255),
    ):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                "pad_val of float type is deprecated now, "
                f"please use pad_val=dict(img={pad_val}, "
                f"masks={pad_val}, seg=255) instead.",
                DeprecationWarning,
            )
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, (
                "The size and size_divisor must be None " "when pad2square is True"
            )
        else:
            assert (
                size is not None or size_divisor is not None
            ), "only one of size and size_divisor should be valid"
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get("img", 0)
        for key in results.get("img_fields", ["img"]):
            if self.pad_to_square:
                max_size = max([image_width(results[key]), image_height(results[key])])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = hm_impad(results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = hm_impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val
                )
            results[key] = padded_img
        results["pad_shape"] = [image_height(padded_img), image_width(padded_img)]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results["pad_shape"][:2]
        pad_val = self.pad_val.get("masks", 0)
        for key in results.get("mask_fields", []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get("seg", 255)
        for key in results.get("seg_fields", []):
            results[key] = hm_impad(
                results[key], shape=results["pad_shape"][:2], pad_val=pad_val
            )

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_to_square={self.pad_to_square}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class HmResize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        interpolation="bilinear",
        override=False,
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get("img_fields", ["img"]):
            if self.keep_ratio:
                img, scale_factor = hm_imrescale(
                    results[key],
                    results["scale"],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                result_img = results[key]
                new_h, new_w = [image_height(img), image_width(img)]
                h, w = [image_height(result_img), image_width(result_img)]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = hm_imresize(
                    results[key],
                    results["scale"],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
            results[key] = img

            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
            results["img_shape"] = [image_height(img), image_width(img), 3]
            # in case that there is no padding
            results["pad_shape"] = results["img_shape"]
            results["scale_factor"] = scale_factor
            results["keep_ratio"] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get("bbox_fields", []):
            bboxes = results[key] * results["scale_factor"]
            if self.bbox_clip_border:
                img_shape = results["img_shape"]
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get("mask_fields", []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results["scale"])
            else:
                results[key] = results[key].resize(results["img_shape"][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get("seg_fields", []):
            if self.keep_ratio:
                gt_seg = hm_imrescale(
                    results[key],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            else:
                gt_seg = hm_imresize(
                    results[key],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            if "scale_factor" in results:
                img_shape = results["img"].shape[:2]
                scale_factor = results["scale_factor"]
                assert isinstance(scale_factor, float)
                results["scale"] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1]
                )
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert (
                    "scale_factor" not in results
                ), "scale and scale_factor cannot be both set."
            else:
                results.pop("scale")
                if "scale_factor" in results:
                    results.pop("scale_factor")
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str
