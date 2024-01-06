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
from hmlib.utils.image import ImageHorizontalGaussianDistribution

from hmlib.utils.box_functions import (
    width,
    height,
    center,
    # center_x_distance,
    # center_distance,
    clamp_box,
    aspect_ratio,
    make_box_at_center,
    shift_box_to_edge,
    scale_box,
    is_box_edge_on_or_outside_other_box_edge,
    check_for_box_overshoot,
    move_box_to_center,
)

from hmlib.utils.box_functions import tlwh_centers

from pt_autograph import pt_function

from hockeymom import core


class BasicBox:
    def __init__(self, bbox: torch.Tensor, device: str = None):
        self.device = bbox.device if device is None else device
        self._zero_float_tensor = torch.tensor(
            0, dtype=torch.float32, device=self.device
        )
        self._zero_int_tensor = torch.tensor(0, dtype=torch.int64, device=self.device)
        self._one_float_tensor = torch.tensor(1, dtype=torch.int64, device=self.device)
        self._true = torch.tensor(True, dtype=torch.bool, device=self.device)
        self._false = torch.tensor(False, dtype=torch.bool, device=self.device)

    def set_bbox(self, bbox: torch.Tensor):
        self._bbox = bbox

    @property
    def _zero(self):
        return self._zero_float_tensor.clone()

    @property
    def _zero_int(self):
        return self._zero_int_tensor.clone()

    def bounding_box(self):
        return self._bbox.clone()


class ResizingBox(BasicBox):
    # TODO: move resizing stuff here
    def __init__(
        self,
        bbox: torch.Tensor,
        max_speed_w: torch.Tensor,
        max_speed_h: torch.Tensor,
        max_accel_w: torch.Tensor,
        max_accel_h: torch.Tensor,
        min_width: torch.Tensor,
        min_height: torch.Tensor,
        max_width: torch.Tensor,
        max_height: torch.Tensor,
        sticky_sizing: bool = False,
        width_change_threshold: torch.Tensor = None,
        width_change_threshold_low: torch.Tensor = None,
        height_change_threshold: torch.Tensor = None,
        height_change_threshold_low: torch.Tensor = None,
        device: str = None,
    ):
        super(ResizingBox, self).__init__(bbox=bbox, device=device)
        self._sticky_sizing = sticky_sizing

        self._max_speed_w = max_speed_w
        self._max_speed_h = max_speed_h
        self._max_accel_w = max_accel_w
        self._max_accel_h = max_accel_h

        # Change threshholds
        self._width_change_threshold = width_change_threshold
        self._width_change_threshold_low = width_change_threshold_low
        self._height_change_threshold = height_change_threshold
        self._height_change_threshold_low = height_change_threshold_low

        self._min_width = min_width
        self._min_height = min_height
        self._max_width = max_width
        self._max_height = max_height

        self._size_is_frozen = True

    def draw(self, img: np.array, draw_threasholds: bool = False):
        pass

    def _clamp_resizing(self):
        self._current_speed_w = torch.clamp(
            self._current_speed_w, min=-self._max_speed_w, max=self._max_speed_w
        )
        self._current_speed_h = torch.clamp(
            self._current_speed_h, min=-self._max_speed_h, max=self._max_speed_h
        )

    def _adjust_size(
        self,
        accel_w: torch.Tensor = None,
        accel_h: torch.Tensor = None,
        use_constraints: bool = True,
    ):
        if self._size_is_frozen:
            return

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

    def set_destination(self, dest_box: torch.Tensor, stop_on_dir_change: bool = True):
        self._set_destination_size(
            dest_width=width(dest_box),
            dest_height=height(dest_box),
            stop_on_dir_change=stop_on_dir_change,
        )

    def _set_destination_size(
        self,
        dest_width: torch.Tensor,
        dest_height: torch.Tensor,
        stop_on_dir_change: bool = True,
    ):
        bbox = self.bounding_box()
        current_w = width(bbox)
        current_h = height(bbox)

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
        if self._sticky_sizing:
            scale_amount = 0.1
            req_w_diff = current_w * scale_amount
            req_h_diff = current_h * scale_amount
            dw_thresh = False
            dh_thresh = False
            want_bigger = False

            zero = self._zero_float_tensor.clone()
            if dw < zero and dw < -req_w_diff:
                dw_thresh = True
            elif dw > zero and dw > req_w_diff:
                dw_thresh = True
                want_bigger = True
            else:
                dw = zero.clone()

            if dh < zero and dh < -req_h_diff:
                dh_thresh = True
            elif dh > zero and dh > req_h_diff:
                dh_thresh = True
                want_bigger = True
            else:
                dh = zero.clone()

            if not dw_thresh and not dh_thresh:
                self._size_is_frozen = True
            elif (dw_thresh and dh_thresh) or (
                want_bigger and (dw_thresh or dh_thresh)
            ):
                self._size_is_frozen = False

            # print(f"frozen size={self._size_is_frozen}")

        #
        # END size threshhold
        #

        assert self._zero.item() == 0

        if different_directions(dw, self._current_speed_w):
            self._current_speed_w = self._zero.clone()
            if stop_on_dir_change:
                dw = self._zero.clone()
                # self._size_is_frozen = True
        if different_directions(dh, self._current_speed_h):
            self._current_speed_h = self._zero.clone()
            if stop_on_dir_change:
                dh = self._zero.clone()
                # self._size_is_frozen = True
        self._adjust_size(accel_w=dw, accel_h=dh, use_constraints=True)


class MovingBox(ResizingBox):
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
        sticky_translation: bool = False,
        sticky_sizing: bool = False,
        width_change_threshold: torch.Tensor = None,
        width_change_threshold_low: torch.Tensor = None,
        height_change_threshold: torch.Tensor = None,
        height_change_threshold_low: torch.Tensor = None,
        color: Tuple[int, int, int] = (255, 0, 0),
        frozen_color: Tuple[int, int, int] = (64, 64, 64),
        thickness: int = 2,
        device: str = None,
    ):
        super().__init__(
            bbox=bbox,
            device=device,
            max_speed_w=max_speed_x / 2,
            max_speed_h=max_speed_y / 2,
            max_accel_w=max_accel_x,
            max_accel_h=max_accel_y,
            sticky_sizing=sticky_sizing,
            width_change_threshold=width_change_threshold,
            width_change_threshold_low=width_change_threshold_low,
            height_change_threshold=height_change_threshold,
            height_change_threshold_low=height_change_threshold_low,
            min_width=0,
            min_height=0,
            max_width=max_width,
            max_height=max_height,
        )
        self._label = label
        self._color = color
        self._frozen_color = frozen_color
        self._thickness = thickness
        self._sticky_translation = sticky_translation

        self._line_thickness_tensor = torch.tensor(
            [2, 2, -1, -2], dtype=torch.float32, device=self.device
        )
        self._inflate_arena_for_unsticky_edges = torch.tensor(
            [1, 1, -1, -1], dtype=torch.float32, device=self.device
        )

        if isinstance(bbox, BasicBox):
            self._following_box = bbox
            bbox = self._following_box.bounding_box()
        else:
            self._following_box = None
            self._size_is_frozen = False

        self._scale_width = (
            self._one_float_tensor if scale_width is None else scale_width
        )
        self._scale_height = (
            self._one_float_tensor if scale_height is None else scale_height
        )
        self.set_bbox(
            make_box_at_center(
                center(bbox),
                w=width(bbox) * self._scale_width,
                h=height(bbox) * self._scale_height,
            )
        )

        self._arena_box = arena_box
        self._fixed_aspect_ratio = fixed_aspect_ratio
        self._current_speed_x = self._zero
        self._current_speed_y = self._zero
        self._current_speed_w = self._zero
        self._current_speed_h = self._zero
        self._nonstop_delay = self._zero
        self._nonstop_delay_counter = self._zero
        self._previous_area = width(bbox) * height(bbox)

        if self._arena_box is not None:
            self._horizontal_image_gaussian_distribution = (
                ImageHorizontalGaussianDistribution(width(self._arena_box))
            )
        else:
            self._horizontal_image_gaussian_distribution = None

        self._translation_is_frozen = False

        # Constraints
        self._max_speed_x = max_speed_x
        self._max_speed_y = max_speed_y
        self._max_accel_x = max_accel_x
        self._max_accel_y = max_accel_y

    def draw(self, img: np.array, draw_threasholds: bool = False):
        super().draw(img=img, draw_threasholds=draw_threasholds)
        draw_box = self.bounding_box()
        img = vis.plot_rectangle(
            img,
            draw_box,
            color=self._color if not self._translation_is_frozen else (128, 128, 128),
            thickness=self._thickness,
            label=self._make_label(),
            text_scale=2,
        )
        if draw_threasholds and self._sticky_translation:
            sticky, unsticky = self._get_sticky_translation_sizes()
            cl = [int(i) for i in center(self.bounding_box())]
            cv2.circle(
                img,
                cl,
                radius=int(sticky),
                color=(255, 0, 0),
                thickness=3,
            )
            cv2.circle(
                img,
                cl,
                radius=int(unsticky),
                color=(255, 0, 255),
                thickness=3,
            )
            if self._following_box is not None:
                following_bbox = self._following_box.bounding_box()
                following_bbox_center = center(following_bbox)
                # dashed box representing the following box inscribed at our center

                scaled_following_box = scale_box(
                    following_bbox.clone(),
                    scale_width=self._scale_width,
                    scale_height=self._scale_height,
                )
                inscribed = move_box_to_center(
                    scaled_following_box.clone(), center(self.bounding_box())
                )
                img = vis.draw_dashed_rectangle(
                    img, box=inscribed, color=(255, 255, 255), thickness=1
                )

                # Line from center of this box to the center of the box that it is following,
                # with little circle nubs at each end.
                co = [int(i) for i in following_bbox_center]
                cv2.circle(
                    img,
                    cl,
                    radius=5,
                    color=(255, 255, 0),
                    thickness=cv2.FILLED,
                )
                cv2.circle(
                    img,
                    co,
                    radius=5,
                    color=(0, 255, 128),
                    thickness=cv2.FILLED,
                )
                vis.plot_line(img, cl, co, color=(255, 255, 0), thickness=1)
                # X
                vis.plot_line(img, cl, [co[0], cl[1]], color=(255, 255, 0), thickness=3)
                # Y
                vis.plot_line(img, cl, [cl[0], co[1]], color=(255, 255, 0), thickness=3)

        return img

    def get_gaussian_y_about_width_center(self, x):
        if self._horizontal_image_gaussian_distribution is None:
            return 1.0
        else:
            return self._horizontal_image_gaussian_distribution.get_gaussian_y_from_image_x_position(
                x
            )

    def _get_sticky_translation_sizes(self):
        gaussian_factor = self.get_gaussian_y_about_width_center(
            center(self.bounding_box())[0]
        )
        gaussian_mult = 6
        gaussian_add = gaussian_factor * gaussian_mult
        # print(f"gaussian_factor={gaussian_factor}, gaussian_add={gaussian_add}")
        sticky_size = self._max_speed_x * 6 + gaussian_add
        unsticky_size = sticky_size * 3 / 4
        return sticky_size, unsticky_size

    def _make_label(self):
        return f"dx={self._current_speed_x.item():.1f}, dy={self._current_speed_y.item()}, {self._label}"

    def _clamp_speed(self):
        self._current_speed_x = torch.clamp(
            self._current_speed_x, min=-self._max_speed_x, max=self._max_speed_x
        )
        self._current_speed_ = torch.clamp(
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

    def set_destination(self, dest_box: torch.Tensor, stop_on_dir_change: bool = True):
        """
        We try to go to the given box's position, given
        our current velocity and constraints
        """
        if isinstance(dest_box, BasicBox):
            dest_box = dest_box.bounding_box()

        bbox = self.bounding_box()
        center_current = center(bbox)
        center_dest = center(dest_box)
        total_diff = center_dest - center_current

        # If both the dest box and our current box are on an edge, we zero-out
        # the magnitude in the direction fo that edge so that the size
        # differences of the box don't keep us in the un-stuck mode,
        # even though we can't move anymore in that direction
        # TODO: do cleverly with pytorch tensors
        if self._arena_box is not None:
            edge_ok = torch.logical_not(
                check_for_box_overshoot(
                    box=bbox,
                    bounding_box=self._arena_box
                    + self._inflate_arena_for_unsticky_edges,
                    movement_directions=total_diff,
                    epsilon=0.1,
                )
            )
            self._current_speed_x *= edge_ok[0]
            self._current_speed_y *= edge_ok[1]
            total_diff *= edge_ok
            # print(total_diff)

        # BEGIN Sticky Translation
        if self._sticky_translation:
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
            # velocity = torch.where(changed_direction < 0, velocity / 6, velocity)
            velocity = torch.where(changed_direction < 0, self._zero, velocity)

            sticky, unsticky = self._get_sticky_translation_sizes()
            if not self._translation_is_frozen and diff_magnitude <= sticky:
                self._translation_is_frozen = True
                self._current_speed_x = self._zero.clone()
                self._current_speed_y = self._zero.clone()
            elif self._translation_is_frozen and diff_magnitude >= unsticky:
                self._translation_is_frozen = False
                # Unstick at zero velocity
                self._current_speed_x = self._zero.clone()
                self._current_speed_y = self._zero.clone()

            if self._translation_is_frozen:
                print("Translation FROZEN")
            else:
                print("Translation unfrozen")

                # clamp to max velocities
                # self._current_speed_x = torch.clamp(self._current_speed_x, -self._max_speed_x, self._max_speed_x)
                # self._current_speed_xy= torch.clamp(self._current_speed_y, -self._max_speed_y, self._max_speed_y)
        # END Sticky Translation

        if not self.is_nonstop():
            if different_directions(total_diff[0], self._current_speed_x):
                self._current_speed_x = self._zero.clone()
                if stop_on_dir_change:
                    total_diff[0] = self._zero.clone()
            if different_directions(total_diff[1], self._current_speed_y):
                self._current_speed_y = self._zero.clone()
                if stop_on_dir_change:
                    total_diff[1] = self._zero.clone()
        self.adjust_speed(
            accel_x=total_diff[0], accel_y=total_diff[1], use_constraints=True
        )
        super(MovingBox, self).set_destination(
            dest_box=scale_box(
                dest_box, scale_width=self._scale_width, scale_height=self._scale_height
            ),
            stop_on_dir_change=stop_on_dir_change,
        )

    def next_position(self, arena_box: torch.Tensor = None):
        if arena_box is None:
            arena_box = self._arena_box
        if self._following_box is not None:
            self.set_destination(
                dest_box=self._following_box.bounding_box(),

                stop_on_dir_change=True,
            )

        # BEGIN Sticky Translation
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
            dh = self._current_speed_h / 2
        # END Sticky Translation

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
            # print(f"self._nonstop_delay_counter={self._nonstop_delay_counter.item()}")
            self._nonstop_delay_counter += 1
            if self._nonstop_delay_counter >= self._nonstop_delay:
                self._nonstop_delay = self._zero
                self._nonstop_delay_counter = self._zero
        self.stop_translation_if_out_of_arena()
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

    def stop_translation_if_out_of_arena(self):
        if self._arena_box is None:
            return
        edge_ok = torch.logical_not(
            check_for_box_overshoot(
                box=self._bbox,
                bounding_box=self._arena_box + self._inflate_arena_for_unsticky_edges,
                movement_directions=torch.tensor(
                    [self._current_speed_x, self._current_speed_y]
                ),
                epsilon=0.1,
            )
        )
        self._current_speed_x *= edge_ok[0]
        self._current_speed_y *= edge_ok[1]

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
    return torch.sign(d1) * torch.sign(d2) < 0
