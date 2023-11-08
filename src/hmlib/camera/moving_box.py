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
    clamp_box,
    aspect_ratio,
    make_box_at_center,
    shift_box_to_edge,
)

from hmlib.utils.box_functions import tlwh_centers

from pt_autograph import pt_function

from hockeymom import core


class BasicMovingBox:
    def bounding_box(self):
        return self._bbox.clone()


class MovingBox(BasicMovingBox):
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
        scale_width: torch.Tensor = None,
        scale_height: torch.Tensor = None,
        arena_box: torch.Tensor = None,
        fixed_aspect_ratio: torch.Tensor = None,
        translation_threshold: torch.Tensor = None,
        translation_threshold_low: torch.Tensor = None,
        width_change_threshold: torch.Tensor = None,
        width_change_threshold_low: torch.Tensor = None,
        height_change_threshold: torch.Tensor = None,
        height_change_threshold_low: torch.Tensor = None,
        color: Tuple[int, int, int] = (255, 0, 0),
        frozen_color: Tuple[int, int, int] = (64, 64, 64),
        thickness: int = 2,
        device: str = None,
    ):
        self._label = label
        self._color = color
        self._frozen_color = frozen_color
        # self._current_color = self._color
        self._thickness = thickness

        if isinstance(bbox, BasicMovingBox):
            self._following_box = bbox
            bbox = self._following_box.bounding_box()
        else:
            self._following_box = None

        self._device = bbox.device if device is None else device
        self._zero_float_tensor = torch.tensor(
            0, dtype=torch.float32, device=self._device
        )
        self._zero_int_tensor = torch.tensor(0, dtype=torch.int64, device=self._device)
        self._one_float_tensor = torch.tensor(1, dtype=torch.int64, device=self._device)
        self._line_thickness_tensor = torch.tensor(
            [2, 2, -1, -2], dtype=torch.float32, device=self._device
        )
        self._true = torch.tensor(True, dtype=torch.bool, device=self._device)
        self._false = torch.tensor(False, dtype=torch.bool, device=self._device)

        self._scale_width = (
            self._one_float_tensor if scale_width is None else scale_width
        )
        self._scale_height = (
            self._one_float_tensor if scale_height is None else scale_height
        )
        self._bbox = make_box_at_center(
            center(bbox),
            w=width(bbox) * self._scale_width,
            h=height(bbox) * self._scale_height,
        )

        self._arena_box = arena_box
        self._fixed_aspect_ratio = fixed_aspect_ratio
        self._current_speed_x = self._zero
        self._current_speed_y = self._zero
        self._current_speed_w = self._zero
        self._current_speed_h = self._zero
        self._nonstop_delay = self._zero
        self._nonstop_delay_counter = self._zero

        self._size_is_frozen = False
        self._translation_is_frozen = False

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
        # Change threshholds
        self._width_change_threshold = width_change_threshold
        self._width_change_threshold_low = width_change_threshold_low
        self._height_change_threshold = height_change_threshold
        self._height_change_threshold_low = height_change_threshold_low
        self._translation_threshold = translation_threshold
        self._translation_threshold_low = translation_threshold_low

    @property
    def _zero(self):
        return self._zero_float_tensor.clone()

    @property
    def _zero_int(self):
        return self._zero_int_tensor.clone()

    def draw(self, img: np.array):
        draw_box = self._bbox.clone()
        vis.plot_rectangle(
            img,
            draw_box,
            color=self._color,
            thickness=self._thickness,
            label=self._make_label(),
            text_scale=2,
        )
        if self._size_is_frozen:
            draw_box += self._line_thickness_tensor
            vis.plot_rectangle(
                img,
                draw_box,
                color=self._frozen_color,
                thickness=self._thickness,
                label=self._make_label(),
                text_scale=2,
            )
        if self._translation_is_frozen:
            draw_box += self._line_thickness_tensor
            vis.plot_rectangle(
                img,
                draw_box,
                color=(255, 255, 255),
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

        diff_magnitude = torch.linalg.norm(total_diff)

        # Check if the new center is in a direction opposed to our current velocity
        velocity = torch.tensor(
            [self._current_speed_x, self._current_speed_y],
            device=self._current_speed_x.device,
        )
        s1 = torch.sign(total_diff)
        s2 = torch.sign(velocity)
        changed_direction = s1 * s2
        # Reduce velocity on axes that changed direction
        #velocity = torch.where(changed_direction < 0, velocity / 6, velocity)
        velocity = torch.where(changed_direction < 0, self._zero, velocity)

        # BEGIN Sticky
        if self._translation_threshold_low is not None:
            assert self._translation_threshold is not None
            if (
                not self._translation_is_frozen
                and diff_magnitude <= self._translation_threshold_low
            ):
                self._translation_is_frozen = True
                self._current_speed_x = self._zero
                self._current_speed_y = self._zero
            elif (
                self._translation_is_frozen
                and diff_magnitude >= self._translation_threshold
            ):
                self._translation_is_frozen = False
        else:
            assert self._translation_threshold is None
        # END Sticky

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
        self.set_destination_size(
            dest_width=width(dest_box), dest_height=height(dest_box)
        )

    def set_destination_size(
        self,
        dest_width: torch.Tensor,
        dest_height: torch.Tensor,
        stop_on_dir_change: bool = True,
    ):
        scale_w = (
            self._one_float_tensor if self._scale_width is None else self._scale_width
        )
        scale_h = (
            self._one_float_tensor if self._scale_height is None else self._scale_height
        )

        current_w = width(self._bbox)
        current_h = height(self._bbox)

        dest_width *= scale_w
        dest_height *= scale_h

        if self._fixed_aspect_ratio is not None:
            # Apply aspect ratio
            if dest_width / dest_height < self._fixed_aspect_ratio:
                # Constrain by height
                dest_width = dest_height * self._fixed_aspect_ratio
            else:
                # Constrain by width
                dest_height = dest_width / self._fixed_aspect_ratio

        dw = dest_width - current_w
        dh = dest_height - current_h

        #
        # BEGIN size threshhold
        #
        stopped_count = 0
        if (
            self._width_change_threshold_low is not None
            and abs(dw) < self._width_change_threshold_low
        ):
            stopped_count += 1
            self._current_speed_w = self._zero
        if (
            self._height_change_threshold_low is not None
            and abs(dh) < self._height_change_threshold_low
        ):
            stopped_count += 1
            self._current_speed_h = self._zero
        if stopped_count == 2:
            self._size_is_frozen = True

        if (
            self._width_change_threshold is not None
            and abs(dw) >= self._width_change_threshold
        ):
            self._size_is_frozen = False
            # start from zero change speed so as not to jerk
            self._current_speed_w = self._zero
            self._current_speed_h = self._zero
        elif (
            self._height_change_threshold is not None
            and abs(dh) >= self._height_change_threshold
        ):
            self._size_is_frozen = False
            # start from zero change speed so as not to jerk
            self._current_speed_w = self._zero
            self._current_speed_h = self._zero
        #
        # END size threshhold
        #
        if different_directions(dw, self._current_speed_w):
            self._current_speed_w = self._zero
            if stop_on_dir_change:
                dw = self._zero
        if different_directions(dh, self._current_speed_h):
            self._current_speed_h = self._zero
            if stop_on_dir_change:
                dh = self._zero
        self.adjust_size(accel_w=dw, accel_h=dh, use_constraints=True)

    def next_position(self, arena_box: torch.Tensor = None):
        if arena_box is None:
            arena_box = self._arena_box
        if self._following_box is not None:
            self.set_destination(
                dest_box=self._following_box.bounding_box(), stop_on_dir_change=True
            )

        # BEGIN Sticky
        if self._translation_is_frozen:
            dx = self._zero
            dy = self._zero
        else:
            dx = self._current_speed_x
            dy = self._current_speed_y

        if self._size_is_frozen:
            dw = self._zero
            dh = self._zero
        else:
            dw = self._current_speed_w / 2
            dh = self._current_speed_h
        # END Sticky

        self._bbox += torch.tensor(
            [
                dx - dw,
                dy - dh,
                dx + dw,
                dy + dh,
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
        self.stop_if_out_of_arena()
        self.clamp_to_arena()
        if self._fixed_aspect_ratio is not None:
            self._bbox = self.set_aspect_ratio(self._bbox, self._fixed_aspect_ratio)
        self.clamp_size_scaled()
        if arena_box is not None:
            self._bbox, was_shifted_x, was_shifted_y = shift_box_to_edge(
                self._bbox, arena_box
            )
            if was_shifted_x:
                # We show down X velocity if we went off the edge
                self._current_speed_x /= 2
            if was_shifted_y:
                # We show down X velocity if we went off the edge
                self._current_speed_y /= 2

        if self._fixed_aspect_ratio is not None:
            assert torch.isclose(aspect_ratio(self._bbox), self._fixed_aspect_ratio)
        return self._bbox

    def clamp_size_scaled(self):
        w = width(self._bbox).unsqueeze(0)
        h = height(self._bbox).unsqueeze(0)
        wscale = self._zero
        hscale = self._zero
        if w > self._max_width:
            wscale = self._max_width / w
        if h > self._max_height:
            hscale = self._max_height / h
        final_scale = torch.max(wscale, hscale)
        if final_scale != self._zero:
            w *= final_scale
            h *= final_scale
            self._bbox = make_box_at_center(center(self._bbox), w=w, h=h)

    def clamp_to_arena(self):
        if self._arena_box is None:
            return
        self._bbox = clamp_box(box=self._bbox, clamp_box=self._arena_box)

    def stop_if_out_of_arena(self):
        if self._arena_box is None:
            return
        out_x = (
            self._bbox[0] <= self._arena_box[0] or self._bbox[2] >= self._arena_box[2]
        )
        out_y = (
            self._bbox[1] <= self._arena_box[1] or self._bbox[3] >= self._arena_box[3]
        )
        self.stop_translation(stop_x=out_x, stop_y=out_y)

    def set_aspect_ratio(self, setting_box: torch.Tensor, aspect_ratio: torch.Tensor):
        if self._arena_box is not None:
            setting_box = clamp_box(setting_box, self._arena_box)
        w = width(setting_box)
        h = height(setting_box)
        if w / h < aspect_ratio:
            # Constrain by height
            new_h = h
            new_w = new_h * aspect_ratio
        else:
            # Constrain by width
            new_w = w
            new_h = new_w / aspect_ratio
        return make_box_at_center(center_point=center(setting_box), w=new_w, h=new_h)

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
