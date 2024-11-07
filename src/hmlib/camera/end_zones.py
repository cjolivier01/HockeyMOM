import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from hmlib.bbox.box_functions import center
from hmlib.config import get_nested_value
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis
from hmlib.ui.show import show_image
from hmlib.utils.image import image_height, image_width
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
        box_key: str = "current_fast_box_list",
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

        self._exposure_ratio: Dict[str, torch.Tensor] = {}

        self.line_equations: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for name, l in self._lines.items():
            # l = make_y_up(l)
            self.line_equations[name] = find_line_equation(
                x1=l[0][0], y1=l[0][1], x2=l[1][0], y2=l[1][1]
            )

    def draw(self, img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        was_torch = isinstance(img, torch.Tensor)
        if was_torch:
            was_dtype = img.dtype
            was_device = img.device
            img = img.cpu()

        for label, line in self._lines.items():
            if "start" in label:
                color = (0, 255, 0)  # green
            else:
                color = (255, 255, 0)  # yellow
            pt1 = line[0]
            pt2 = line[1]
            img = vis.plot_line(img, [pt1[0], pt1[1]], [pt2[0], pt2[1]], color=color, thickness=2)
        if was_torch:
            img = torch.from_numpy(img).to(device=was_device).to(dtype=was_dtype)
        return img

    def get_ez_image(self, data: Dict[str, Any], dtype: torch.dtype) -> Optional[torch.Tensor]:
        ez_image = data.get(self._image_key)
        if ez_image is not None and ez_image.dtype != self._output_dtype:
            if torch.is_floating_point(ez_image) and self._output_dtype == torch.uint8:
                ez_image = ez_image.clamp(0, 255)
            ez_image = ez_image.to(dtype=self._output_dtype, non_blocking=True)
        return ez_image

    def put_ez_image(self, data: Dict[str, Any], img: torch.Tensor) -> Dict[str, Any]:
        if img is None:
            if self._image_key in data:
                del data[self._image_key]
        else:
            data[self._image_key] = img
        return data

    def forward(self, data: Dict[str, Any], show: bool = False) -> Dict[str, Any]:
        dataset_data = data["dataset_results"]
        if "far_left" not in dataset_data and "far_right" not in dataset_data:
            return data
        bboxes = data.get(self._box_key)
        if bboxes is None:
            return data
        replacement_images: List[torch.Tensor] = []
        # We look at the average of the boxes
        bbox = torch.mean(bboxes, dim=0)
        cc = center(bbox)
        pano_size_height = data["pano_size_wh"][1]
        # See if we're out of the current left or right zone
        if self._current_zone == ZONE_LEFT:
            # Make sure we aren't to the right of the stop line
            pos = point_line_position(self._lines["left_stop"], cc, image_height=pano_size_height)
            if pos > 0:
                self._current_zone = ZONE_MIDDLE
                logger.info("END-ZONE: MIDDLE")
        elif self._current_zone == ZONE_RIGHT:
            # Make sure we aren't to the right of the stop line
            pos = point_line_position(self._lines["right_stop"], cc, image_height=pano_size_height)
            if pos < 0:
                self._current_zone = ZONE_MIDDLE
                logger.info("END-ZONE: MIDDLE")
        # See if we're in the left or right zone
        if self._current_zone == ZONE_MIDDLE:
            if (
                point_line_position(self._lines["left_start"], cc, image_height=pano_size_height)
                < 0
            ):
                self._current_zone = ZONE_LEFT
                logger.info("END-ZONE: LEFT")
            elif (
                point_line_position(self._lines["right_start"], cc, image_height=pano_size_height)
                > 0
            ):
                self._current_zone = ZONE_RIGHT
                logger.info("END-ZONE: RIGHT")

        replacement_image = None
        side_name = None
        if self._current_zone == ZONE_LEFT and "far_left" in dataset_data:
            replacement_image = dataset_data["far_left"][0]
            side_name = "far_left"
        elif self._current_zone == ZONE_RIGHT and "far_right" in dataset_data:
            replacement_image = dataset_data["far_right"][0]
            side_name = "far_right"
        if replacement_image is not None:
            replacement_image, _, _, _, _ = py_letterbox(
                img=replacement_image.get(),
                height=self._output_height,
                width=self._output_width,
                color=0,
            )
            if show:
                show_image("ez", replacement_image, wait=False)
            if side_name not in self._exposure_ratio:
                # We calc for the exact corresponding frame on the first frame
                img = data["img"]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img_mean = torch.mean(img.to(torch.float).clamp(0, 255))
                repl_mean = torch.mean(replacement_image.clamp(0, 255))
                if img_mean > 0 and repl_mean > 0:
                    self._exposure_ratio[side_name] = float(img_mean) / float(repl_mean)
            if side_name in self._exposure_ratio and self._exposure_ratio[side_name] != 1:
                replacement_image = replacement_image * self._exposure_ratio[side_name]
            self.put_ez_image(data, replacement_image)

        return data


def make_y_up(point: List[Tuple[float, float]], image_height: float) -> List[Tuple[float, float]]:
    return [point[0][0], image_height - point[0][1], point[1][0], image_height - point[1][1]]


def find_line_equation(x1, y1, x2, y2) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the slope and y-intercept of the line given two points.

    Args:
    x1 (float): x-coordinate of the first point.
    y1 (float): y-coordinate of the first point.
    x2 (float): x-coordinate of the second point.
    y2 (float): y-coordinate of the second point.

    Returns:
    tuple: (slope, y_intercept) if the line is not vertical, otherwise (None, None)
    """
    if x2 != x1:  # Check to prevent division by zero in case of vertical line
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
        return (slope, y_intercept)
    else:
        return (None, None)  # This is the case for a vertical line which has no defined slope


def point_line_position(
    line_segment: Union[torch.Tensor, List[List[int]]],
    point: Union[torch.Tensor, List[int]],
    image_height: int,
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

    # y1 = image_height - y1
    # y2 = image_height - y2

    assert x1 <= x2

    # TODO: Precalculate as much as possible
    leq = find_line_equation(x1, y1, x2, y2)

    # # Unpack the point
    x, y = point

    # y = image_height - y

    # y = mx + b
    # mx = y - b
    # x = (y - b) / m
    b = leq[1]
    m = leq[0]
    xx = (y - b) / m

    # # cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    if x < xx:
        # Is to the left of the line @ y
        return -1
    elif x > xx:
        # Is to the right of the line @ y
        return 1
    # Is on the line
    return 0

    # # Determine the position of the point relative to the line
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
