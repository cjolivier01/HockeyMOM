from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from numba import njit

import time
import os
import cv2
import argparse
import numpy as np
import traceback
from typing import Tuple
import multiprocessing
import queue

from pathlib import Path

import torch
import torchvision as tv

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer

from hmlib.utils.box_functions import (
    width,
    height,
    center,
    center_x_distance,
    center_distance,
    # clamp_value,
    aspect_ratio,
    make_box_at_center,
)

from hmlib.utils.box_functions import tlwh_centers

from hockeymom import core


class MovingBox:
    def __init__(
        self,
        label: str,
        bbox: torch.Tensor,
        max_speed_x: torch.Tensor,
        max_speed_y: torch.Tensor,
        max_accel_x: torch.Tensor,
        max_accel_y: torch.Tensor,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        device: str = None,
    ):
        self._label = label
        self._color = color
        self._thickness = thickness
        self._bbox = bbox
        self._device = bbox.device if device is None else device
        self._zero_tensor = torch.tensor([0], dtype=torch.float32, device=self._device)
        self._current_speed_x = self._zero
        self._current_speed_y = self._zero
        self._dest_center = center(bbox)
        # Constraints
        self._max_speed_x = max_speed_x
        self._max_speed_y = max_speed_y
        self._max_accel_x = max_accel_x
        self._max_accel_y = max_accel_y
        self._min_width = 0
        self._min_height = 0
        self._max_width = -1
        self._max_height = -1

    @property
    def _zero(self):
        return self._zero_tensor.clone()

    def draw(self, img: np.array):
        vis.plot_rectangle(
            img,
            self._bbox,
            color=self._color,
            thickness=self._thickness,
            label=self._make_label(),
            text_scale=2,
        )

    def _make_label(self):
        return f"dx={self._current_speed_x.item():.1f}, dy={self._current_speed_y.item()}, {self._label}"

    def _clamp_speed(self):
        self._current_speed_x = torch.clamp(
            self._current_speed_x, min=-self._max_speed_x, max=self._max_speed_x
        )
        self._current_speed_y = torch.clamp(
            self._current_speed_y, min=-self._max_speed_y, max=self._max_speed_y
        )

    def set_speed(
        self,
        speed_x: torch.Tensor,
        speed_y: torch.Tensor,
        use_constraints: bool = True,
    ):
        if speed_x is not None:
            self._current_speed_x = speed_x
        if speed_y is not None:
            self._current_speed_y = speed_y
        if use_constraints:
            self._clamp_speed()

    def adjust_speed(
        self,
        accel_x: torch.Tensor = None,
        accel_y: torch.Tensor = None,
        use_constraints: bool = True,
    ):
        if use_constraints:
            if accel_x is not None:
                accel_x = torch.clamp(
                    accel_x, min=-self._max_accel_x, max=self._max_accel_x
                )
            if accel_y is not None:
                accel_y = torch.clamp(
                    accel_x, min=-self._max_accel_y, max=self._max_accel_y
                )
        if accel_x is not None:
            self._current_speed_x += accel_x
        if accel_y is not None:
            self._current_speed_y += accel_y
        if use_constraints:
            self._clamp_speed()

    def stop(self, stop_x: bool = True, stop_y: bool = True):
        if stop_x:
            self._current_speed_x = self._zero
        if stop_y:
            self._current_speed_y = self._zero

    def set_destination(self, dest_box: torch.Tensor, stop_on_dir_change: bool = True):
        """
        We try to go to the given box's position, given
        our current velocity and constraints
        """
        center_current = center(self._bbox)
        center_dest = center(dest_box)
        total_diff = center_dest - center_current

        if different_directions(total_diff[0], self._current_speed_x):
            self._current_speed_x = self._zero
            if stop_on_dir_change:
                total_diff[0] = self._zero
        if different_directions(total_diff[1], self._current_speed_y):
            self._current_speed_y = self._zero
            if stop_on_dir_change:
                total_diff[1] = self._zero
        self.adjust_speed(
            accel_x=total_diff[0], accel_y=total_diff[1], use_constraints=True
        )
        self._dest_center = center_dest

    def next_position(self):
        self._bbox += torch.tensor(
            [
                self._current_speed_x,
                self._current_speed_y,
                self._current_speed_x,
                self._current_speed_y,
            ],
            dtype=self._bbox.dtype,
            device=self._bbox.device,
        )
        return self._bbox

    def __iter__(self):
        return self

    def __next__(self):
        self.next_position()
        return self

    @property
    def bbox(self):
        return self._bbox


def different_directions(d1: torch.Tensor, d2: torch.Tensor):
    return torch.sign(d1) * torch.sign(d2) == -1
