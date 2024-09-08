import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from hmlib.config import get_nested_value
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.log import logger
from hmlib.utils.box_functions import center
from hmlib.utils.letterbox import py_letterbox

ZONE_LEFT: str = "LEFT"
ZONE_MIDDLE: str = "MIDDLE"
ZONE_RIGHT: str = "RIGHT"


class EndZones(torch.nn.Module):

    def __init__(
        self,
        lines: Dict[str, List[Tuple[int, int]]],
        output_width: int,
        output_height: int,
        output_dtype: torch.dtype = torch.uint8,
        box_key: str = "current_fast_box",
        image_key: str = "end_zone_img",
        *args,
        **kwargs,
    ):
        super(EndZones, self).__init__(*args, **kwargs)
        self._lines = lines
        self._box_key = box_key
        self._image_key = image_key
        self._output_width: int = output_width
        self._output_height: int = output_height
        self._output_dtype = output_dtype
        self._args = args
        self._current_zone = ZONE_MIDDLE

    def draw(self, img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        was_torch = isinstance(img, torch.Tensor)
        if was_torch:
            was_dtype = img.dtype
            was_device = img.device
        for label, line in self._lines.items():
            if "start" in label:
                color = (0, 255, 0)  # green
            else:
                color = (255, 255, 0)  # yellow
            pt1 = line[0]
            pt2 = line[1]
            img = vis.plot_line(img, [pt1[0], pt1[1]], [pt2[0], pt2[1]], color=color, thickness=8)
        if was_torch:
            img = torch.from_numpy(img).to(device=was_device).to(dtype=was_dtype)
        return img

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        bbox = data.get(self._box_key)
        if bbox is None:
            return data
        cc = center(bbox)

        # See if we're out of the current left or right zone
        if self._current_zone == ZONE_LEFT:
            # Make sure we aren't to the right of the stop line
            pos = point_line_position(self._lines["left_stop"], cc)
            if pos > 0:
                self._current_zone = ZONE_MIDDLE
                logger.info("EZ: MIDDLE")
        elif self._current_zone == ZONE_RIGHT:
            # Make sure we aren't to the right of the stop line
            pos = point_line_position(self._lines["right_stop"], cc)
            if pos < 0:
                self._current_zone = ZONE_MIDDLE
                logger.info("EZ: MIDDLE")
        # See if we're in the left or right zone
        if self._current_zone == ZONE_MIDDLE:
            if point_line_position(self._lines["left_start"], cc) < 0:
                self._current_zone = ZONE_LEFT
                logger.info("EZ: LEFT")
            elif point_line_position(self._lines["right_start"], cc) > 0:
                self._current_zone = ZONE_RIGHT
                logger.info("EZ: RIGHT")

        replacement_image = None
        if self._current_zone == ZONE_LEFT and "far_left" in data["data"]:
            replacement_image = data["data"]["far_left"][0]
        elif self._current_zone == ZONE_RIGHT and "far_right" in data["data"]:
            replacement_image = data["data"]["far_right"][0]
        if replacement_image is not None:
            replacement_image, _, _, _, _ = py_letterbox(
                img=replacement_image.get(),
                height=self._output_height,
                width=self._output_width,
                color=0,
            )
            if replacement_image.dtype != self._output_dtype:
                if torch.is_floating_point(replacement_image) and self._output_dtype == torch.uint8:
                    replacement_image = replacement_image.clamp(0, 255)
                    replacement_image = replacement_image.to(
                        dtype=self._output_dtype, non_blocking=True
                    )
            data[self._image_key] = replacement_image

        return data


def point_line_position(
    line_segment: Union[torch.Tensor, List[List[int]]], point: Union[torch.Tensor, List[int]]
) -> int:
    """
    Determines the position of a point relative to a line segment.

    Args:
    line_segment (Tensor): A tensor of shape [2, 2] representing the endpoints of the line segment.
    point (Tensor): A tensor of shape [2] representing the point.

    Returns:
    int: -1 if the point's x is left of the point on the line at the same y,
         +1 if it is to the right, and 0 if it is on the line.
    """
    # Unpack line segment
    x1, y1 = line_segment[0]
    x2, y2 = line_segment[1]

    # # Unpack the point
    x, y = point

    # cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # Determine the position of the point relative to the line
    # return torch.sign(cross_product)

    diff_x = x - (x1 + x2) / 2
    return torch.sign(diff_x)


def get_line(
    game_config: Dict[str, Any], key, dflt: List[Tuple[int, int]] = []
) -> List[Tuple[int, int]]:
    line = get_nested_value(game_config, key)
    if line is None:
        return copy.deepcopy(dflt)
    assert len(line) == 2
    return line


def load_lines_from_config(config: Dict[str, Any]) -> List[Tuple[int, int]]:
    lines: Dict[str, List[Tuple[int, int]]] = {}
    labels = ["left_start", "left_stop", "right_start", "right_stop"]
    for label in labels:
        line = get_line(config, f"rink.end_zones.{label}")
        if line:
            lines[label] = line
    return lines
