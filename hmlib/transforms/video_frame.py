"""Transforms that operate on per-frame video tensors (cropping, sharpening).

These classes are registered as mmengine transforms and typically appear in
Aspen pipelines to prepare frames for downstream models.

@see @ref hmlib.hm_opts.hm_opts "hm_opts" for CLI flags that configure video-frame behavior.
"""

from typing import Any, Dict, List, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.algo.unsharp_mask import unsharp_mask
from hmlib.log import logger
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import crop_image, image_height, image_width, resize_image, to_float_image


def _slow_to_tensor(tensor: Union[torch.Tensor, StreamTensorBase]) -> torch.Tensor:
    """Convert a possibly streamed tensor to a concrete :class:`torch.Tensor`.

    @param tensor: Plain tensor or :class:`hmlib.utils.gpu.StreamTensorBase`.
    @return: Synchronized tensor on the same device.
    @see @ref hmlib.utils.gpu.StreamTensorBase "StreamTensorBase" for details.
    """
    if isinstance(tensor, StreamTensorBase):
        tensor._verbose = True
        # return tensor.get()
        return tensor.wait()
    return tensor


@TRANSFORMS.register_module()
class HmCropToVideoFrame:
    """Crop camera images to a fixed video frame aspect and size.

    Expects ``results`` to contain a list of images under ``\"img\"`` and
    bounding boxes under ``\"camera_box\"``. The cropped, resized tensors
    are stacked back into ``results[\"img\"]``.

    @see @ref HmUnsharpMask "HmUnsharpMask" for optional sharpening later in the pipeline.
    """

    def __init__(self, crop_image: bool = True, unsharp_mask: bool = True):
        self._crop_image = crop_image
        self._unsharp_mask = unsharp_mask

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        video_frame_cfg = results["video_frame_cfg"]
        images = results.pop("img")
        if not self._crop_image:
            if isinstance(images, list):
                # since no creopping, they should be the same size
                results["img"] = torch.stack(images)
            else:
                # Just put it back
                results["img"] = images
            return results
        current_boxes = results["camera_box"]
        cropped_images: List[torch.Tensor] = []
        for img, bbox in zip(images, current_boxes):
            src_image_height = image_height(img)
            src_image_width = image_width(img)
            img = _slow_to_tensor(img)
            img = to_float_image(img, non_blocking=True, dtype=torch.float)
            intbox = [int(i) for i in bbox]
            x1 = intbox[0]
            y1 = intbox[1]
            y2 = intbox[3]
            x2 = int(x1 + int(float(y2 - y1) * video_frame_cfg["output_aspect_ratio"]))
            if True:
                if not (y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0):
                    pass
            assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
            if y1 > src_image_height or y2 > src_image_height:
                logger.info(f"y1 ({y1}) or y2 ({y2}) is too large, should be < {src_image_height}")
                y1 = min(y1, src_image_height)
                y2 = min(y2, src_image_height)
            if x1 > src_image_width or x2 > src_image_width:
                logger.info(f"x1 {x1} or x2 {x2} is too large, should be < {src_image_width}")
                x1 = min(x1, src_image_width)
                x2 = min(x2, src_image_width)

            img = crop_image(img, x1, y1, x2, y2)
            video_frame_width = int(video_frame_cfg["output_frame_width"])
            video_frame_height = int(video_frame_cfg["output_frame_height"])
            if image_height(img) != video_frame_height or image_width(img) != video_frame_width:
                img = resize_image(
                    img=img,
                    new_width=video_frame_width,
                    new_height=video_frame_height,
                )
            # By here, our image size should be exactly the video frame output size
            assert image_height(img) == video_frame_height
            assert image_width(img) == video_frame_width
            cropped_images.append(img)
        results["img"] = torch.stack(cropped_images)
        return results


@TRANSFORMS.register_module()
class HmUnsharpMask:
    """Apply an unsharp mask filter to images in a pipeline result dict.

    @param enabled: If ``True``, apply the filter to each call.
    @param image_label: Key inside ``results`` that holds the image tensor.
    @see @ref hmlib.algo.unsharp_mask.unsharp_mask "unsharp_mask" for implementation details.
    """

    def __init__(self, enabled: bool = False, image_label: str = "img") -> None:
        self._enabled = enabled
        self._image_label = image_label
        # self._count = 0
        # self._flip = False

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self._enabled:
            # if self._count % 250 == 0:
            #     self._flip = not self._flip
            #     print("off" if not self._flip else "on")
            # if self._flip:
            results[self._image_label] = unsharp_mask(results[self._image_label])
        # self._count += 1
        return results
