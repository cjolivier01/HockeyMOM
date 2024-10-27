from contextlib import contextmanager
from typing import Any, Dict, Tuple, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.config import get_clip_box, get_config, get_nested_value
from hmlib.log import logger
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.scoreboard.selector import configure_scoreboard
from hmlib.utils.box_functions import center, height, width
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import (
    crop_image,
    image_height,
    image_width,
    make_channels_last,
    resize_image,
    rotate_image,
    to_float_image,
)
from hmlib.utils.iterators import CachedIterator
from hmlib.video.video_stream import VideoStreamReader


def _slow_to_tensor(tensor: Union[torch.Tensor, StreamTensor]) -> torch.Tensor:
    """
    Give up on the stream and get the sync'd tensor
    """
    if isinstance(tensor, StreamTensor):
        tensor._verbose = True
        # return tensor.get()
        return tensor.wait()
    return tensor


@TRANSFORMS.register_module()
class HmCropToVideoFrame:
    def __init__(self, crop_image: bool = True):
        self._crop_image = crop_image

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        video_frame_cfg = results["video_frame_cfg"]
        images = results.pop("img")
        if not self._crop_image:
            if isinstance(images, list):
                # since no creopping, they should be the same size
                results["img"] = torch.stack(images)
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
            assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
            if y1 >= src_image_height or y2 >= src_image_height:
                logger.info(f"y1 ({y1}) or y2 ({y2}) is too large, should be < {src_image_height}")
                y1 = min(y1, src_image_height)
                y2 = min(y2, src_image_height)
            if x1 >= src_image_width or x2 >= src_image_width:
                logger.info(f"x1 {x1} or x2 {x2} is too large, should be < {src_image_width}")
                x1 = min(x1, src_image_width)
                x2 = min(x2, src_image_width)

            img = crop_image(img, x1, y1, x2, y2)
            if (
                image_height(img) != video_frame_cfg["output_frame_height"]
                or image_width(img) != video_frame_cfg["output_frame_width"]
            ):
                img = resize_image(
                    img=img,
                    new_width=video_frame_cfg["output_frame_width"],
                    new_height=video_frame_cfg["output_frame_height"],
                )
            assert image_height(img) == video_frame_cfg["output_frame_height"]
            assert image_width(img) == video_frame_cfg["output_frame_width"]
            cropped_images.append(img)
        results["img"] = torch.stack(cropped_images)

        return results
