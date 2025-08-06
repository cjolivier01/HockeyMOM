from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import torch
from mmengine.registry import TRANSFORMS

from hmlib.bbox.box_functions import center, height, width
from hmlib.config import get_clip_box, get_config, get_nested_value, prepend_root_dir
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


def paste_watermark_at_position(dest_image, watermark_rgb_channels, watermark_mask, x: int, y: int):
    assert dest_image.ndim == 4
    assert dest_image.device == watermark_rgb_channels.device
    assert dest_image.device == watermark_mask.device
    watermark_height = image_height(watermark_rgb_channels)
    watermark_width = image_width(watermark_rgb_channels)
    dest_image[:, y : y + watermark_height, x : x + watermark_width] = (
        dest_image[:, y : y + watermark_height, x : x + watermark_width] * (1 - watermark_mask)
        + watermark_rgb_channels * watermark_mask
    )
    return dest_image


@TRANSFORMS.register_module()
class HmImageOverlays:

    def __init__(
        self,
        frame_number: bool = False,
        frame_time: bool = False,
        watermark_config: Dict[str, Any] = None,
        colors: Dict[str, Tuple[int, int, int]] = None,
        device: Optional[torch.device] = None,
    ):
        self._draw_frame_number = frame_number
        self._draw_frame_time = frame_time
        self._watermark_config = watermark_config
        self._colors = colors if colors is not None else {}
        self._image_height_percent: float = 0.001
        self._watermark = None
        self._device = device
        if self._watermark_config:
            self._load_watermark()

    def _load_watermark(self):
        # TODO: Make watermark separate
        if self._watermark_config and "image" in self._watermark_config:
            self._watermark = self._watermark_config["image"]
            if self._watermark is None:
                return
            if isinstance(self._watermark, str):
                self._watermark = cv2.imread(
                    prepend_root_dir(self._watermark),
                    cv2.IMREAD_UNCHANGED,
                )
            if self._watermark is None:
                raise InvalidArgumentError(f"Could not load watermark image: {self._watermark_image}")
            self._watermark_height = image_height(self._watermark)
            self._watermark_width = image_width(self._watermark)
            self._watermark_rgb_channels = self._watermark[:, :, :3]
            watermark_alpha_channel = self._watermark[:, :, 3]
            self._watermark_mask = cv2.merge(
                [
                    watermark_alpha_channel,
                    watermark_alpha_channel,
                    watermark_alpha_channel,
                ]
            )

            if self._device is not None:
                self._watermark_rgb_channels = torch.from_numpy(self._watermark_rgb_channels).to(self._device)
                self._watermark_mask = torch.from_numpy(self._watermark_mask).to(self._device).to(torch.half)
                # Scale mask to [0, 1]
                self._watermark_mask = self._watermark_mask / torch.max(self._watermark_mask)
        else:
            self._watermark = None

    def _draw_watermark(self, img: torch.Tensor) -> torch.Tensor:
        if self._watermark is not None:
            y = int(image_height(img) - self._watermark_height)
            x = int(image_width(img) - self._watermark_width - self._watermark_width / 10)
            img = paste_watermark_at_position(
                img,
                watermark_rgb_channels=self._watermark_rgb_channels,
                watermark_mask=self._watermark_mask,
                x=x,
                y=y,
            )
        return img

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self._watermark is not None:
            results["img"] = self._draw_watermark(results["img"])

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
