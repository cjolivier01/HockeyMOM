from contextlib import contextmanager
from typing import Any, Dict, Tuple, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.bbox.box_functions import center, height, width
from hmlib.config import get_clip_box, get_config, get_nested_value
from hmlib.log import logger
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.scoreboard.selector import configure_scoreboard
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


@TRANSFORMS.register_module()
class HmImageOverlays:
    def __init__(self):
        pass

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:

        return results
