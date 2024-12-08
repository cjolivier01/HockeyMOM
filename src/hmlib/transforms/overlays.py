from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import torch
from mmengine.registry import TRANSFORMS

from hmlib.bbox.box_functions import center, height, width
from hmlib.config import get_clip_box, get_config, get_nested_value
from hmlib.log import logger
from hmlib.scoreboard.selector import configure_scoreboard
from hmlib.tracking_utils import visualization as vis
from hmlib.ui import show_image
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import (
    crop_image,
    image_height,
    image_width,
    is_channels_first,
    make_channels_first,
    make_channels_last,
    resize_image,
    rotate_image,
    to_float_image,
)
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.time import format_duration_to_hhmmss
from hmlib.video.video_stream import VideoStreamReader


@TRANSFORMS.register_module()
class HmImageOverlays:

    def __init__(
        self,
        frame_number: bool = False,
        frame_time: bool = False,
        watermark_image: str = None,
        colors: Dict[str, Tuple[int, int, int]] = None,
    ):
        self._draw_frame_number = frame_number
        self._draw_frame_time = frame_time
        self._watermark_image = watermark_image
        self._colors = colors if colors is not None else {}
        self._image_height_percent: float = 0.001

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Does not seem to work for some reason
        if True:
            draw_msg: str = ""
            frame_id: Optional[int] = None
            if self._draw_frame_number:
                frame_ids = results.get("frame_ids")
                if frame_ids is not None:
                    # Only first frame id is drawn
                    frame_id = int(frame_ids[0])
                    draw_msg += f"F: {frame_id}\n"
            if self._draw_frame_time and frame_id is not None:
                assert frame_id > 0  # Should start at 1
                fps = results.get("fps")
                if fps:
                    frame_time = frame_id / fps
                    draw_msg += f"{format_duration_to_hhmmss(frame_time, decimals=2)}\n"
            if draw_msg:
                img = results["img"]
                icf = is_channels_first(img)
                if not icf:
                    img = make_channels_first(img)
                h = image_height(img)
                font_scale = max(h * self._image_height_percent, 2)
                img = vis.plot_text(
                    img=make_channels_first(img),
                    text=draw_msg,
                    org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=font_scale,
                    color=self._colors.get("frame_number", (0, 0, 255)),
                    thickness=10,
                )
                if not icf:
                    img = make_channels_last(img)
                results["img"] = img

        return results
