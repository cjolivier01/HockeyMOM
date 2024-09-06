import copy
from typing import Any, Dict, List, Tuple

import torch

from hmlib.config import get_nested_value
from hmlib.utils.image import (
    ImageColorScaler,
    ImageHorizontalGaussianDistribution,
    crop_image,
    image_height,
    image_width,
    make_channels_last,
    make_visible_image,
    resize_image,
)
from hmlib.utils.letterbox import py_letterbox
from hmlib.utils.progress_bar import ProgressBar

ZONE_LEFT: str = "LEFT"
ZONE_MIDDLE: str = "MIDDLE"
ZONE_RIGHT: str = "RIGHT"


class EndZones(torch.nn.Module):

    def __init__(
        self,
        lines: Dict[str, List[Tuple[int, int]]],
        output_width: int,
        output_height: int,
        *args,
        **kwargs,
    ):
        super(EndZones, self).__init__(*args, **kwargs)
        self._lines = lines
        self._output_width: int = output_width
        self._output_height: int = output_height
        self._args = args
        self._current_zone = "MIDDLE"

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def check_for_replacement(self):
        # replacement_image = None
        # if self.has_args() and self._args.end_zones:
        #     pano_width = image_width(online_im)
        #     current_box_x = center(current_fast_box)[0]
        #     current_box_left = current_fast_box[0]
        #     current_box_right = current_fast_box[2]
        #     if current_box_right - current_box_left < pano_width / 1.5:
        #         other_data = imgproc_data["data"]
        #         # print(int(current_box_x))
        #         width_ratio = 4
        #         if current_box_x <= pano_width / width_ratio and "far_left" in other_data:
        #             if current_zone != "LEFT":
        #                 current_zone = "LEFT"
        #                 print(f"{current_zone=}")
        #             replacement_image = other_data["far_left"]
        #             if replacement_image is not None:
        #                 replacement_image = replacement_image[0]
        #         elif (
        #             pano_width - current_box_x <= pano_width / width_ratio
        #             and "far_right" in other_data
        #         ):
        #             if current_zone != "RIGHT":
        #                 current_zone = "RIGHT"
        #                 print(f"{current_zone=}")
        #             replacement_image = other_data["far_right"]
        #             if replacement_image is not None:
        #                 replacement_image = replacement_image[0]
        #         else:
        #             if current_zone != "MIDDLE":
        #                 current_zone = "MIDDLE"
        #                 print(f"{current_zone=}")

        #     else:
        #         if current_zone != "MIDDLE":
        #             current_zone = "MIDDLE"
        #             print(f"{current_zone=}")

        #     if replacement_image is not None:
        #         replacement_image, _, _, _, _ = py_letterbox(
        #             img=replacement_image.get(),
        #             height=self._output_frame_height,
        #             width=self._output_frame_width,
        #             color=0,
        #         )
        #         assert image_width(replacement_image) == self._output_frame_width
        #         assert image_height(replacement_image) == self._output_frame_height
        pass


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
