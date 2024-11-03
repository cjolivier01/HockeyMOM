from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

import hmlib.tracking_utils.visualization as vis
from hmlib.builder import HM, PIPELINES
from hmlib.camera.camera import HockeyMOM
from hmlib.camera.clusters import ClusterMan
from hmlib.camera.moving_box import MovingBox
from hmlib.config import get_nested_value
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.box_functions import (
    center,
    center_batch,
    clamp_box,
    get_enclosing_box,
    height,
    make_box_at_center,
    remove_largest_bbox,
    scale_box,
    tlwh_centers,
    tlwh_to_tlbr_single,
    width,
)
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_channels_last
from hmlib.utils.progress_bar import ProgressBar

from .number_classifier import TrackJerseyInfo


@dataclass
class TrackingIdNumberInfo:
    tracking_id: int = -1
    current_number: int = -1
    max_score: float = 0.0
    # frame_id -> TrackJerseyInfo
    occurrences: OrderedDict[int, TrackJerseyInfo] = None
    # number -> # occurrences of the number
    number_occurrences: OrderedDict[int, int] = None


class JerseyTracker:
    def __init__(self, show: bool = False) -> None:
        self._show = show
        self._tracking_id_jersey: Dict[int, TrackingIdNumberInfo] = {}
        self._jersey_number_to_tracking_id: Dict[int, int] = {}

    def draw(
        self, image: torch.Tensor, tracking_ids: torch.Tensor, bboxes: torch.Tensor
    ) -> torch.Tensor:
        if not self._show:
            return image
        assert len(tracking_ids) == len(bboxes)
        text_scale = max(2, image.shape[1] / 2500.0)
        for (x1, y1, w, _), tracking_id in zip(bboxes, tracking_ids):
            tracking_id = int(tracking_id)
            info: TrackingIdNumberInfo = self._tracking_id_jersey.get(tracking_id)
            if info is not None:
                xc = int(x1 + w // 2)
                yc = y1
                image = vis.my_put_text(
                    image,
                    str(info.current_number),
                    (xc, yc),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (200, 0, 0),  # Blue
                    thickness=3,
                )
        return image

    def observe_tracking_id_number_info(self, frame_id: int, info: TrackingIdNumberInfo) -> None:
        current_tracking_id = info.tracking_id

        prev_info: TrackingIdNumberInfo = self._tracking_id_jersey.get(current_tracking_id)

        # Base case, it's a new tracking id
        if prev_info is None:
            self._tracking_id_jersey[current_tracking_id] = TrackingIdNumberInfo(
                tracking_id=current_tracking_id,
                current_number=info.number,
                max_score=info.score,
                occurrences={frame_id: info},
                number_occurrences={info.number: 1},
            )
            return
        # Next base case, everything is the same
        if prev_info.current_number == info.number:
            prev_info.number_occurrences[prev_info.current_number] += 1
            # print(f"Recurrence: ID: {current_tracking_id} -> # {prev_info.current_number}")
            return

        # Something changed
        logger.info(
            f"Conflict: ID: {current_tracking_id} # {prev_info.current_number} -> # {info.number}"
        )
        # Record the occurence count
        if info.number not in prev_info.number_occurrences:
            prev_info.occurrences[frame_id] = info
            prev_info.number_occurrences[info.number] = 1
        else:
            prev_info.number_occurrences[info.number] += 1

        # Decide whether to override
        if info.score > prev_info.max_score:
            # Higher score, so maybe replace the number
            prev_info.occurrences[frame_id] = info
            # If higher occurrence count
            if (
                prev_info.number_occurrences[info.number]
                >= prev_info.number_occurrences[prev_info.current_number]
            ):
                logger.info(
                    f"Reassigning ID {current_tracking_id} from # {prev_info.current_number} "
                    f"to # {info.number} (score {prev_info.max_score} -> {info.score})"
                )
                prev_info.current_number = info.number
                prev_info.max_score = info.score
        else:
            # Lower score, but we need to record it anyway
            pass
