import cv2
import numpy as np
from typing import Tuple, Union
import torch

from hmlib.builder import HM
from hmlib.tracking_utils import visualization as vis
from hmlib.utils.image import ImageHorizontalGaussianDistribution
from hmlib.utils.box_functions import (
    width,
    height,
    center,
    clamp_box,
    aspect_ratio,
    make_box_at_center,
    shift_box_to_edge,
    scale_box,
    check_for_box_overshoot,
    move_box_to_center,
)


class BasicBox(torch.nn.Module):
    def __init__(
        self, bbox: torch.Tensor, device: Union[torch.device, str, None] = None
    ):
        super(BasicBox, self).__init__()
        if device and isinstance(device, str):
            device = torch.device(device)
        self.device = bbox.device if device is None else device
        self._zero_float_tensor = torch.tensor(0, dtype=torch.float, device=self.device)
        self._zero_int_tensor = torch.tensor(0, dtype=torch.int64, device=self.device)
        self._one_float_tensor = torch.tensor(1, dtype=torch.int64, device=self.device)
        self._true = torch.tensor(True, dtype=torch.bool, device=self.device)
        self._false = torch.tensor(False, dtype=torch.bool, device=self.device)

    def set_bbox(self, bbox: torch.Tensor):
        self._bbox = bbox

    @property
    def one(self):
        return self._one_float_tensor.clone()

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
        self._size_constrained = False

        #
        # Sticky sizing thresholds
        #
        # Threshold to grow width (ratio of bbox)
        self._size_ratio_thresh_grow_dw = 0.05
        # Threshold to grow height (ratio of bbox)
        self._size_ratio_thresh_grow_dy = 0.1
        # Threshold to shrink width (ratio of bbox)
        self._size_ratio_thresh_shrink_dw = 0.08
        # Threshold to shrink height (ratio of bbox)
        self._size_ratio_thresh_shrink_dy = 0.1

        self._size_is_frozen = True

    def draw(
        self,
        img: np.array,
        draw_thresholds: bool = True,
        following_box: BasicBox = None,
    ):
        if self._sticky_sizing:
            assert following_box is not None  # why?
            my_bbox = self.bounding_box()
            center_tensor = center(my_bbox)
            my_width = width(my_bbox)
            my_height = height(my_bbox)
            following_bbox = following_box.bounding_box()
            # dashed box representing the following box inscribed at our center

            if self._size_is_frozen:
                corner_box = scale_box(
                    box=my_bbox.clone(), scale_width=0.98, scale_height=0.98
                )
                img = vis.draw_corner_boxes(
                    image=img, bbox=corner_box, color=(255, 255, 255), thickness=1
                )

            scaled_following_box = scale_box(
                following_bbox,
                scale_width=self._scale_width,
                scale_height=self._scale_height,
            )

            inscribed = move_box_to_center(
                scaled_following_box.clone(), center(my_bbox)
            )
            # print(
            #     f"inscribed: {width(scaled_following_box)} x {height(scaled_following_box)}"
            # )
            img = vis.draw_dashed_rectangle(
                img,
                box=inscribed,
                color=(255, 255, 255) if not self._size_constrained else (0, 0, 255),
                thickness=2,
            )

            # Sizing thresholds
            if draw_thresholds:
                grow_width, grow_height, shrink_width, shrink_height = (
                    self._get_grow_wh_and_shrink_wh(bbox=my_bbox)
                )
                grow_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width + grow_width,
                    h=my_height + grow_height,
                )
                shrink_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width - shrink_width,
                    h=my_height - shrink_height,
                )

                img = vis.draw_centered_lines(
                    img, bbox=grow_box, thickness=4, color=(0, 255, 0)
                )
                img = vis.draw_centered_lines(
                    img, bbox=shrink_box, thickness=4, color=(0, 0, 255)
                )
        return img

    def _get_grow_wh_and_shrink_wh(self, bbox: torch.Tensor):
        my_width = width(bbox)
        my_height = height(bbox)
        grow_width = my_width * self._size_ratio_thresh_grow_dw
        grow_height = my_height * self._size_ratio_thresh_grow_dy
        shrink_width = my_width * self._size_ratio_thresh_shrink_dw
        shrink_height = my_height * self._size_ratio_thresh_shrink_dy
        return grow_width, grow_height, shrink_width, shrink_height

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
            # Growing is allowed at a higher rate than shrinking
            resize_larger_scale = 2.0
            max_accel_wh = torch.tensor([self._max_accel_w, self._max_accel_h])
            max_accel_wh = torch.where(
                torch.tensor([accel_w, accel_h]) > 0,
                max_accel_wh * resize_larger_scale,
                max_accel_wh,
            )

            if accel_w is not None:
                accel_w = torch.clamp(
                    accel_w, min=-max_accel_wh[0], max=max_accel_wh[0]
                )
            if accel_h is not None:
                accel_h = torch.clamp(
                    accel_h, min=-max_accel_wh[1], max=max_accel_wh[1]
                )
        if accel_w is not None:
            self._current_speed_w += accel_w

        if accel_h is not None:
            self._current_speed_h += accel_h

        if use_constraints:
            self._clamp_resizing()

    def set_destination(self, dest_box: torch.Tensor):
        self._set_destination_size(
            dest_width=width(dest_box),
            dest_height=height(dest_box),
        )

    def _set_destination_size(
        self,
        dest_width: torch.Tensor,
        dest_height: torch.Tensor,
    ):
        bbox = self.bounding_box()
        current_w = width(bbox)
        current_h = height(bbox)

        dw = dest_width - current_w
        dh = dest_height - current_h

        #
        # BEGIN size threshhold
        #
        if self._sticky_sizing:

            grow_width, grow_height, shrink_width, shrink_height = (
                self._get_grow_wh_and_shrink_wh(bbox=bbox)
            )

            dw_thresh = torch.logical_and(dw < 0, dw < -shrink_width)
            want_bigger_w = torch.logical_and(dw > 0, dw > grow_width)
            dw_thresh = torch.logical_or(dw_thresh, want_bigger_w)
            dw = torch.where(dw_thresh, dw, 0)

            dh_thresh = torch.logical_and(dh < 0, dh < -shrink_height)
            want_bigger_h = torch.logical_and(dh > 0, dh > grow_height)
            dh_thresh = torch.logical_or(dh_thresh, want_bigger_h)
            dh = torch.where(dw_thresh, dh, 0)

            self._size_is_frozen = torch.logical_and(
                torch.logical_not(dw_thresh), torch.logical_not(dh_thresh)
            )
            both_thresh = torch.logical_and(dw_thresh, dh_thresh)

            # prioritize width thresh
            if True:
                both_thresh = torch.logical_or(dw_thresh, both_thresh)

            any_thresh = torch.logical_or(dw_thresh, dh_thresh)
            want_bigger = torch.logical_and(
                torch.logical_or(want_bigger_w, want_bigger_h), any_thresh
            )
            self._size_is_frozen = torch.logical_or(
                self._size_is_frozen,
                torch.logical_not(torch.logical_or(both_thresh, want_bigger)),
            )
        #
        # END size threshhold
        #

        assert self._zero.item() == 0

        if different_directions(dw, self._current_speed_w):
            # self._current_speed_w = self._zero.clone()
            self._current_speed_w = torch.where(
                torch.abs(self._current_speed_w) < self._max_speed_w / 6,
                0,
                self._current_speed_w / 2,
            )
            # if stop_on_dir_change:
            #     dw = self._zero.clone()
            # self._size_is_frozen = True
        if different_directions(dh, self._current_speed_h):
            # self._current_speed_h = self._zero.clone()
            self._current_speed_h = torch.where(
                torch.abs(self._current_speed_h) < self._max_speed_h / 6,
                0,
                self._current_speed_w / 2,
            )
            # if stop_on_dir_change:
            #     dh = self._zero.clone()
            # self._size_is_frozen = True

        # # Growing is allowed at a higher speed than shrinking
        # resize_larger_scale = 2.0
        # dw_dh = torch.tensor([dw, dh])
        # dw_dh = torch.where(dw_dh > 0, dw_dh * resize_larger_scale, dw_dh)
        # self._adjust_size(accel_w=dw_dh[0], accel_h=dw_dh[1], use_constraints=True)

        self._adjust_size(accel_w=dw, accel_h=dh, use_constraints=True)


# @HM.register_module()
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
        min_width: int = 10,
        min_height: int = 10,
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
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
        )
        self._label = label
        self._color = color
        self._frozen_color = frozen_color
        self._thickness = thickness
        self._sticky_translation = sticky_translation

        self._line_thickness_tensor = torch.tensor(
            [2, 2, -1, -2], dtype=torch.float, device=self.device
        )
        self._inflate_arena_for_unsticky_edges = torch.tensor(
            [1, 1, -1, -1], dtype=torch.float, device=self.device
        )

        # if isinstance(bbox, BasicBox):
        #     self._following_box = bbox
        #     bbox = self._following_box.bounding_box()
        # else:
        #     self._following_box = None
        #     self._size_is_frozen = False

        # self._following_box = None
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

        if self._arena_box is not None:
            self._gaussian_x_clamp = torch.tensor(
                [self._arena_box[0], self._arena_box[2]]
            )
        else:
            self._gaussian_x_clamp = None

        # Set up some values to help us to efficiently calculate the current zoom level
        # as well as other rules based upon zoom level (i.e. gaussian wrt center position)
        self._full_aspect_ratio_size = None
        if self._arena_box is not None and self._fixed_aspect_ratio:
            ah = width(self._arena_box)
            aw = height(self._arena_box)
            # Figure out which is the constraining edge
            ar_w = ah * self._fixed_aspect_ratio
            if ar_w <= aw:
                ww = ar_w
                hh = ah
            else:
                ww = aw
                hh = aw / self._fixed_aspect_ratio
                assert hh <= ah
            self._full_aspect_ratio_size = torch.tensor([ww, hh])
            self._full_aspect_ratio_sqrt_area = torch.sqrt(
                torch.square(ww) + torch.square(hh)
            )
            # Gaussian clamp to center x of max aspect ratio box
            self._gaussian_x_clamp[0] += ww / 2
            self._gaussian_x_clamp[1] -= ww / 2

    def get_zoom_level(self):
        """
        We calculate zoom level as the ratio of sqrt area of
        ourselves vs that of a max-sized box on the arena for a fixed aspect ratio
        """
        if self._full_aspect_ratio_size is None:
            return self.one
        bbox = self.bounding_box()
        zl = torch.sqrt(torch.square(width(bbox) + torch.square(height(bbox))))
        return zl

    def draw(
        self,
        img: np.array,
        draw_thresholds: bool = False,
        following_box: BasicBox = None,
    ):
        super().draw(
            img=img, draw_thresholds=draw_thresholds, following_box=following_box
        )
        draw_box = self.bounding_box()
        img = vis.plot_rectangle(
            img,
            draw_box,
            color=self._color if not self._translation_is_frozen else (128, 128, 128),
            thickness=self._thickness,
            label=self._make_label(),
            text_scale=2,
        )
        # img = vis.draw_arrows(img, bbox=draw_box, horizontal=True, vertical=True)
        if draw_thresholds and self._sticky_translation:
            sticky, unsticky = self._get_sticky_translation_sizes()
            center_tensor = center(self.bounding_box())
            my_center = [int(i) for i in center_tensor]
            my_width = width(self.bounding_box())
            my_height = height(self.bounding_box())
            cv2.circle(
                img,
                my_center,
                radius=int(sticky),
                color=(255, 0, 0),
                thickness=3,
            )
            cv2.circle(
                img,
                my_center,
                radius=int(unsticky),
                color=(255, 0, 255),
                thickness=3,
            )
            if following_box is not None:
                following_bbox = following_box.bounding_box()
                following_bbox_center = center(following_bbox)

                # Line from center of this box to the center of the box that it is following,
                # with little circle nubs at each end.
                co = [int(i) for i in following_bbox_center]
                cv2.circle(
                    img,
                    my_center,
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
                img = vis.plot_line(
                    img, my_center, co, color=(255, 255, 0), thickness=1
                )
                # X
                img = vis.plot_line(
                    img,
                    my_center,
                    [co[0], my_center[1]],
                    color=(255, 255, 0),
                    thickness=3,
                )
                # Y
                img = vis.plot_line(
                    img,
                    my_center,
                    [my_center[0], co[1]],
                    color=(255, 255, 0),
                    thickness=3,
                )

        return img

    def get_gaussian_y_about_width_center(self, x):
        if self._horizontal_image_gaussian_distribution is None:
            return 1.0
        else:
            if self._gaussian_x_clamp is not None:
                x = torch.clamp(
                    x, min=self._gaussian_x_clamp[0], max=self._gaussian_x_clamp[1]
                )
            return self._horizontal_image_gaussian_distribution.get_gaussian_y_from_image_x_position(
                x
            )

    def _get_sticky_translation_sizes(self):
        gaussian_factor = 1 - self.get_gaussian_y_about_width_center(
            center(self.bounding_box())[0]
        )
        gaussian_mult = 6
        gaussian_add = gaussian_factor * gaussian_mult
        # print(f"gaussian_factor={gaussian_factor}, gaussian_add={gaussian_add}")

        max_sticky_size = self._max_speed_x * 5 + gaussian_add
        sticky_size = width(self.bounding_box()) / 10
        sticky_size = min(sticky_size, max_sticky_size)

        unsticky_size = sticky_size * 3 / 4

        return sticky_size, unsticky_size

    def _make_label(self):
        return f"dx={self._current_speed_x.item():.1f}, dy={self._current_speed_y.item()}, {self._label}"

    def _clamp_speed(self, scale: float = 1.0):
        mx = self._max_speed_x * scale
        my = self._max_speed_y * scale
        self._current_speed_x = torch.clamp(self._current_speed_x, min=-mx, max=mx)
        self._current_speed_ = torch.clamp(self._current_speed_y, min=-my, max=my)

    def adjust_speed(
        self,
        accel_x: torch.Tensor = None,
        accel_y: torch.Tensor = None,
        scale_constraints: float = None,
        nonstop_delay: torch.Tensor = None,
    ):
        if scale_constraints is not None:
            mult = scale_constraints
            if accel_x is not None:
                accel_x = torch.clamp(
                    accel_x, min=-self._max_accel_x * mult, max=self._max_accel_x * mult
                )
            if accel_y is not None:
                accel_y = torch.clamp(
                    accel_y, min=-self._max_accel_y * mult, max=self._max_accel_y * mult
                )
        if accel_x is not None:
            self._current_speed_x += accel_x

        if accel_y is not None:
            self._current_speed_y += accel_y

        if scale_constraints is not None:
            self._clamp_speed(scale=scale_constraints)

        if nonstop_delay is not None:
            self._nonstop_delay = nonstop_delay
            self._nonstop_delay_counter = self._zero.clone()

    def scale_speed(
        self,
        ratio_x: torch.Tensor = None,
        ratio_y: torch.Tensor = None,
    ):
        if ratio_x is not None:
            self._current_speed_x *= ratio_x

        if ratio_y is not None:
            self._current_speed_y += ratio_y

    def set_destination(self, dest_box: torch.Tensor, stop_on_dir_change: bool = True):
        """
        We try to go to the given box's position, given
        our current velocity and constraints
        """
        if isinstance(dest_box, BasicBox):
            dest_box = scale_box(
                box=dest_box.bounding_box(),
                scale_width=self._scale_width,
                scale_height=self._scale_height,
            )

        bbox = self.bounding_box()
        center_current = center(bbox)
        center_dest = center(dest_box)
        total_diff = center_dest - center_current

        # If both the dest box and our current box are on an edge, we zero-out
        # the magnitude in the direction of that edge so that the size
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
        if self._sticky_translation and not self.is_nonstop():
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
        # END Sticky Translation

        if not self.is_nonstop():
            if different_directions(total_diff[0], self._current_speed_x):
                # self._current_speed_x = self._zero.clone()
                # self._current_speed_x /= 2

                self._current_speed_x = torch.where(
                    torch.abs(self._current_speed_x) < self._max_speed_x / 6,
                    0,
                    self._current_speed_x / 2,
                )

                if stop_on_dir_change:
                    total_diff[0] = self._zero.clone()
            if different_directions(total_diff[1], self._current_speed_y):
                # self._current_speed_y = self._zero.clone()
                # self._current_speed_y /= 2

                self._current_speed_y = torch.where(
                    torch.abs(self._current_speed_y) < self._max_speed_y / 6,
                    0,
                    self._current_speed_y / 2,
                )

                if stop_on_dir_change:
                    total_diff[1] = self._zero.clone()
        self.adjust_speed(
            accel_x=total_diff[0],
            accel_y=total_diff[1],
            scale_constraints=1.0,
        )

        super(MovingBox, self).set_destination(dest_box=dest_box)

    def forward(self, dest_box: torch.Tensor, stop_on_dir_change: bool):
        if self._scale_width is not None or self._scale_height is not None:
            dest_box = scale_box(
                dest_box, scale_width=self._scale_width, scale_height=self._scale_height
            )
        self.set_destination(dest_box=dest_box, stop_on_dir_change=stop_on_dir_change)
        return self.next_position()

    def next_position(self):
        # if arena_box is None:
        arena_box = self._arena_box

        # BEGIN Sticky Translation
        if self._translation_is_frozen and not self.is_nonstop():
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

        new_box = self._bbox
        new_box += torch.tensor(
            [
                dx - dw,
                dy - dh,
                dx + dw,
                dy + dh,
            ],
            dtype=self._bbox.dtype,
            device=self._bbox.device,
        )

        # Constrain size
        new_ww = width(new_box)
        new_hh = height(new_box)
        ww = max(new_ww, self._min_width)
        hh = max(new_hh, self._min_height)
        self._size_constrained = ww != new_ww or hh != new_hh
        new_box = make_box_at_center(center_point=center(new_box), w=ww, h=hh)

        # Assign new bounding box
        self._bbox = new_box

        if self._nonstop_delay != self._zero:
            # print(f"self._nonstop_delay_counter={self._nonstop_delay_counter.item()}")
            self._nonstop_delay_counter += 1
            if self._nonstop_delay_counter > self._nonstop_delay:
                self._nonstop_delay = self._zero
                self._nonstop_delay_counter = self._zero.clone()
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
        assert new_w >= w
        return make_box_at_center(center_point=center(setting_box), w=new_w, h=new_h)

    def is_nonstop(self):
        return self._nonstop_delay != self._zero


def different_directions(d1: torch.Tensor, d2: torch.Tensor):
    return torch.sign(d1) * torch.sign(d2) < 0
