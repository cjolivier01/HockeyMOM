from typing import Any, Dict, List, Tuple

import torch

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


class EndZones(torch.nn.Module):
    def __init__(
        self,
        lines: Dict[str, List[Tuple[int, int]]],
        pano_width: int,
        pano_height: int,
        output_width: int,
        output_height: int,
        *args,
        **kwargs
    ):
        super(self, EndZones).__init__(*args, **kwargs)
        self._lines = lines
        self._pano_width: int = pano_width
        self._pano_height: int = pano_height
        self._output_width: int = output_width
        self._output_height: int = output_height
        self._args = args

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
