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
        max_width: torch.Tensor,
        max_height: torch.Tensor,
        fixed_aspect_ratio: torch.Tensor = None,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        device: str = None,
    ):
        self._label = label
        self._color = color
        self._thickness = thickness
        self._bbox = bbox
        self._fixed_aspect_ratio = fixed_aspect_ratio
        self._device = bbox.device if device is None else device
        self._zero_tensor = torch.tensor([0], dtype=torch.float32, device=self._device)
        self._current_speed_x = self._zero
        self._current_speed_y = self._zero
        self._current_speed_w = self._zero
        self._current_speed_h = self._zero
        self._nonstop_delay = self._zero
        self._nonstop_delay_counter = self._zero
        self._dest_center = center(bbox)
        # Constraints
        self._max_speed_x = max_speed_x
        self._max_speed_y = max_speed_y
        self._max_accel_x = max_accel_x
        self._max_accel_y = max_accel_y
        self._min_width = 0
        self._min_height = 0
        self._max_width = max_width
        self._max_height = max_height
        self._max_speed_w = max_speed_x / 2
        self._max_speed_h = max_speed_y / 2
        self._max_accel_w = max_accel_x
        self._max_accel_h = max_accel_y

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

    def _clamp_resizing(self):
        self._current_speed_w = torch.clamp(
            self._current_speed_w, min=-self._max_speed_w, max=self._max_speed_w
        )
        self._current_speed_h = torch.clamp(
            self._current_speed_h, min=-self._max_speed_h, max=self._max_speed_h
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
        nonstop_delay: torch.Tensor = None,
    ):
        if use_constraints:
            if accel_x is not None:
                accel_x = torch.clamp(
                    accel_x, min=-self._max_accel_x, max=self._max_accel_x
                )
            if accel_y is not None:
                accel_y = torch.clamp(
                    accel_y, min=-self._max_accel_y, max=self._max_accel_y
                )
        if accel_x is not None:
            self._current_speed_x += accel_x

        if accel_y is not None:
            self._current_speed_y += accel_y
        if nonstop_delay is not None:
            self._nonstop_delay = nonstop_delay
            self._nonstop_delay_counter = self._zero
        # if use_constraints:
        #     self._clamp_speed()

    def adjust_size(
        self,
        accel_w: torch.Tensor = None,
        accel_h: torch.Tensor = None,
        use_constraints: bool = True,
    ):
        if use_constraints:
            if accel_w is not None:
                accel_w = torch.clamp(
                    accel_w, min=-self._max_accel_w, max=self._max_accel_w
                )
            if accel_h is not None:
                accel_h = torch.clamp(
                    accel_h, min=-self._max_accel_h, max=self._max_accel_h
                )
        if accel_w is not None:
            self._current_speed_w += accel_w

        if accel_h is not None:
            self._current_speed_h += accel_h

        if use_constraints:
            self._clamp_resizing()

    def stop_translation(self, stop_x: bool = True, stop_y: bool = True):
        if stop_x:
            self._current_speed_x = self._zero
        if stop_y:
            self._current_speed_y = self._zero

    def stop_resizing(self, stop_w: bool = True, stop_h: bool = True):
        if stop_w:
            self._current_speed_w = self._zero
        if stop_h:
            self._current_speed_h = self._zero

    def set_destination(self, dest_box: torch.Tensor, stop_on_dir_change: bool = True):
        """
        We try to go to the given box's position, given
        our current velocity and constraints
        """
        if isinstance(dest_box, MovingBox):
            dest_box = dest_box._bbox

        center_current = center(self._bbox)
        center_dest = center(dest_box)
        total_diff = center_dest - center_current

        if not self.is_nonstop():
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
        self.set_size(dest_width=width(dest_box), dest_height=height(dest_box))

    def set_size(
        self,
        dest_width: torch.Tensor,
        dest_height: torch.Tensor,
        stop_on_dir_change: bool = True,
    ):
        current_w = width(self._bbox)
        current_h = height(self._bbox)
        dw = dest_width - current_w
        dh = dest_height - current_h
        # print(f"dw={dw.item()}, dh={dh.item()}")
        if different_directions(dw, self._current_speed_w):
            self._current_speed_w = self._zero
            if stop_on_dir_change:
                dw = self._zero
        if different_directions(dh, self._current_speed_h):
            self._current_speed_h = self._zero
            if stop_on_dir_change:
                dh = self._zero
        self.adjust_size(accel_w=dw, accel_h=dh, use_constraints=True)

    def next_position(self):
        self._bbox += torch.tensor(
            [
                self._current_speed_x - self._current_speed_w / 2,
                self._current_speed_y - self._current_speed_h / 2,
                self._current_speed_x + self._current_speed_w / 2,
                self._current_speed_y + self._current_speed_h / 2,
            ],
            dtype=self._bbox.dtype,
            device=self._bbox.device,
        )
        if self._nonstop_delay != self._zero:
            print(f"self._nonstop_delay_counter={self._nonstop_delay_counter.item()}")
            self._nonstop_delay_counter += 1
            if self._nonstop_delay_counter >= self._nonstop_delay:
                self._nonstop_delay = self._zero
                self._nonstop_delay_counter = self._zero
        if self._fixed_aspect_ratio is not None:
            self.set_aspect_ratio(self._fixed_aspect_ratio)
        return self._bbox

    def set_aspect_ratio(self, aspect_ratio: torch.Tensor):
        w = width(self._bbox)
        h = height(self._bbox)
        if w / h < aspect_ratio:
            # Constrain by height
            new_h = h
            new_w = new_h * aspect_ratio
        else:
            # Constrain by width
            new_w = w
            new_h = new_w / aspect_ratio
        self._bbox = make_box_at_center(
            center_point=center(self._bbox), w=new_w, h=new_h
        )

    def is_nonstop(self):
        return self._nonstop_delay != self._zero

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
