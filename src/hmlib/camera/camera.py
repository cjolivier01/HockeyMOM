from typing import Dict, List, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from hmlib.tracking_utils.log import logger
from hmlib.utils.box_functions import (
    center,
    clamp_box,
    height,
    tlwh_to_tlbr_single,
    width,
)

# nosec B101

X_SPEED_HISTORY_LENGTH = 5


def _as_scalar(tensor):
    return tensor


def get_image_top_view(
    image_file: str,
    camera: ct.Camera,
    dimensions_2d_m: Tuple[float, float, float, float],
):
    im = plt.imread(image_file)
    top_im = camera.getTopViewOfImage(image=im, extent=dimensions_2d_m, do_plot=False)
    cv2.imshow("online_im", top_im)
    cv2.waitKey(0)


def is_equal(f1, f2):
    return torch.isclose(f1, f2, rtol=0.01, atol=0.01)


class VideoFrame(object):
    def __init__(self, image_width: int, image_height: int):
        # Make sure shape indexes haven't been screwed up
        assert bool(image_width > 10)
        assert bool(image_height > 10)
        assert bool(image_height < image_width)  # Usually the case, but not required
        self._image_width = _as_scalar(image_width)
        self._image_height = _as_scalar(image_height)
        self._vertical_center = self._image_height / 2
        self._horizontal_center = self._image_width / 2
        self._bbox = torch.tensor(
            (0, 0, self._image_width - 1, self._image_height - 1),
            dtype=torch.float,
            device=self._image_width.device,
        )

    def bounding_box(self):
        return self._bbox.clone()

    @property
    def width(self):
        return self._image_width.clone()

    @property
    def height(self):
        return self._image_height.clone()


class TlwhHistory(object):
    def __init__(self, id: int, video_frame: VideoFrame, max_history_length: int = 25):
        self.id_ = id
        self._max_history_length = max_history_length
        self._image_position_history: List[torch.Tensor] = list()
        self._spatial_distance_sum = 0.0
        self._current_spatial_x_speed = 0.0
        self._image_distance_sum = 0.0
        self._current_image_speed = 0.0
        self._current_image_x_speed = 0.0

    @property
    def id(self):
        return self.id_

    def append(self, image_position: torch.Tensor, spatial_position: torch.Tensor = None):
        current_historty_length = len(self._image_position_history)
        if current_historty_length >= self._max_history_length - 1:
            assert current_historty_length == self._max_history_length - 1
            removing_image_point = self._image_position_history[0]
            self._image_position_history = self._image_position_history[1:]
            old_image_distance = torch.linalg.norm(
                removing_image_point - self._image_position_history[0]
            )
            self._image_distance_sum -= old_image_distance
        if len(self._image_position_history) > 0:
            new_image_distance = torch.linalg.norm(
                image_position - self._image_position_history[-1]
            )
            self._image_distance_sum += new_image_distance
        self._image_position_history.append(image_position)
        if len(self._image_position_history) > 1:
            self._current_image_speed = self._image_distance_sum / (
                len(self._image_position_history) - 1
            )
            if len(self._image_position_history) >= X_SPEED_HISTORY_LENGTH:
                self._current_image_x_speed = (
                    self._image_position_history[-1][0]
                    - self._image_position_history[-X_SPEED_HISTORY_LENGTH][0]
                ) / float(
                    X_SPEED_HISTORY_LENGTH
                )  # type: ignore
            else:
                self._current_image_x_speed = 0.0
        else:
            self._current_image_speed = 0.0
            self._current_image_x_speed = 0.0

    @property
    def current_image_x_speed(self):
        return self._current_image_x_speed

    @property
    def image_position_history(self):
        return self._image_position_history

    @staticmethod
    def center_point(tlwh: torch.Tensor):
        assert tlwh.ndim == 1  # Not a batch
        top = tlwh[0]
        left = tlwh[1]
        width = tlwh[2]
        height = tlwh[3]
        x_center = left + width / 2
        y_center = top + height / 2
        return torch.tensor((x_center, y_center), dtype=torch.float)

    def __len__(self):
        length = len(self._image_position_history)
        return length

    @property
    def image_speed(self):
        return self._current_image_speed


class Clustering:
    def __init__(self):
        pass


CAMERA_TYPE_MAX_SPEEDS = {
    "GoPro": 200.0,
    "Zhiwei": 200.0,
    "LiveBarn": 150.0,
}


class HockeyMOM:

    def __init__(
        self,
        image_width: int,
        image_height: int,
        device,
        camera_name: str,
        max_history: int = 26,
        speed_history: int = 26,
    ):
        self._device = device
        self._video_frame = VideoFrame(
            image_width=self._to_scalar_float(image_width),
            image_height=self._to_scalar_float(image_height),
        )
        self._clamp_box = self._video_frame.bounding_box()
        self._online_tlwhs_history: List[torch.Tensor] = list()
        self._max_history = max_history
        self._online_ids: Set[int] = set()
        self._id_to_tlwhs_history_map = dict()

        self._kmeans_objects = dict()
        self._cluster_counts = dict()
        self._cluster_label_ids = dict()
        self._largest_cluster_label = dict()
        self._online_image_center_points = []

        self._epsilon = self._to_scalar_float(0.00001)

        #
        # The camera's box speed itself
        #
        self._current_camera_box_speed_x = self._to_scalar_float(0)
        self._current_camera_box_speed_y = self._to_scalar_float(0)

        self._current_camera_box_speed_reversed_x = False
        self._current_camera_box_speed_reversed_y = False

        self._camera_box_max_speed_x = self._to_scalar_float(
            max(image_width / CAMERA_TYPE_MAX_SPEEDS[camera_name], 12.0)
        )
        self._camera_box_max_speed_y = self._to_scalar_float(
            max(image_height / CAMERA_TYPE_MAX_SPEEDS[camera_name], 12.0)
        )
        logger.info(
            f"Camera Max speeds: x={self._camera_box_max_speed_x}, y={self._camera_box_max_speed_y}"
        )

        self._camera_box_max_accel_x = self._to_scalar_float(1)
        self._camera_box_max_accel_y = self._to_scalar_float(1)
        logger.info(
            f"Camera Max acceleration: dx={self._camera_box_max_accel_x}, dy={self._camera_box_max_accel_y}"
        )

        self._last_acceleration_dx = self._to_scalar_float(0)
        self._last_acceleration_dy = self._to_scalar_float(0)

        #
        # Zoom velocity
        #
        self._camera_box_size_change_velocity_x = self._to_scalar_float(0)
        self._camera_box_size_change_velocity_y = self._to_scalar_float(0)

        self._camera_box_max_size_change_velocity_x = self._to_scalar_float(2)
        self._camera_box_max_size_change_velocity_y = self._to_scalar_float(2)

    def _to_scalar_float(self, scalar_float):
        return torch.tensor(scalar_float, dtype=torch.float, device=self._device)

    def is_fast(self, speed: float = 7):
        # return abs(self._current_camera_box_speed_x) > speed or abs(self._current_camera_box_speed_y) > speed
        return self.get_speed() >= speed

    def append_online_objects(self, online_ids, online_tlws):
        # assert isinstance(online_tlwh_map, dict)
        if len(online_ids) == 0:
            return
        assert isinstance(online_ids, torch.Tensor)
        assert isinstance(online_tlws, torch.Tensor)
        self._online_ids = online_ids
        self._online_tlws = online_tlws
        if len(online_tlws) != 0:
            self._online_image_center_points = torch.stack(
                [TlwhHistory.center_point(twls) for twls in online_tlws]
            )
        else:
            self._online_image_center_points = []

        # Add to history
        prev_dict = self._id_to_tlwhs_history_map
        self._id_to_tlwhs_history_map = dict()
        for id, image_pos in zip(self._online_ids, self._online_tlws):
            hist = prev_dict.get(
                id.item(), TlwhHistory(id=id.item(), video_frame=self._video_frame)
            )
            hist.append(image_position=image_pos)
            self._id_to_tlwhs_history_map[id.item()] = hist

    @property
    def online_image_center_points(self):
        return self._online_image_center_points

    @property
    def video(self):
        return self._video_frame

    @property
    def clamp_box(self):
        return self._clamp_box

    def get_image_tracking(self, online_ids):
        results = []
        for id in online_ids:
            results.append(self._id_to_tlwhs_history_map[id.item()].image_position_history)
        return results

    def get_fast_ids(self, min_fast_items: int = 4, max_fast_items: int = 6):
        return self._prune_slow_ids(
            id_to_pos_history=self._id_to_tlwhs_history_map,
            min_fast_items=min_fast_items,
            max_fast_items=max_fast_items,
        )

    # def get_spatial_speed(self, id: int):
    #     return self._id_to_tlwhs_history_map[id].spatial_speed

    def get_image_speed(self, id: int):
        return self._id_to_tlwhs_history_map[id.item()].image_speed

    def get_tlwh(self, id: int):
        position_history = self._id_to_tlwhs_history_map[id.item()]
        return position_history.image_position_history[-1]

    @staticmethod
    def _prune_slow_ids(
        id_to_pos_history: Dict[int, TlwhHistory],
        fast_threshhold: float = 0.5,
        min_fast_items: int = -1,
        max_fast_items: int = -1,
        min_history_length: int = 20,
    ) -> List[int]:
        # Sort by speed
        keys = list(id_to_pos_history.keys())
        # speeds = [hist.spatial_speed for hist in id_to_pos_history.values()]
        speeds = [hist.image_speed for hist in id_to_pos_history.values()]
        sorted_value_index = np.argsort(speeds)
        sorted_ids = reversed([keys[i] for i in sorted_value_index])
        sorted_speeds = reversed([speeds[i] for i in sorted_value_index])
        fast_ids = []
        # trunk-ignore(ruff/B905)
        for sorted_id, sorted_speed in zip(sorted_ids, sorted_speeds):
            if sorted_speed >= fast_threshhold:
                if len(id_to_pos_history[sorted_id]) >= min_history_length:
                    fast_ids.append(sorted_id)
                    if max_fast_items > 0 and len(fast_ids) >= max_fast_items:
                        break

        if min_fast_items > 0:
            if len(fast_ids) < min_fast_items:
                fast_ids = []

        # Now prune to
        return fast_ids

    def get_group_x_velocity(
        self, min_considered_velocity: float = 0.01, group_threshhold: float = 0.6
    ):
        neg_count = 0
        pos_count = 0
        idle_count = 0
        neg_sum = 0.0
        pos_sum = 0.0
        leftmost_center = None
        rightmost_center = None
        for _, hist in self._id_to_tlwhs_history_map.items():
            pos = hist.image_position_history[-1]
            this_pos_center = [pos[0] + pos[2] / 2, pos[1] + pos[3] / 2]
            if leftmost_center is None:
                leftmost_center = this_pos_center
            if rightmost_center is None:
                rightmost_center = this_pos_center
            image_speed = hist.current_image_x_speed
            if image_speed < -min_considered_velocity:
                neg_count += 1
                neg_sum += image_speed
                if this_pos_center[0] < leftmost_center[0]:
                    leftmost_center = this_pos_center
            elif image_speed > min_considered_velocity:
                pos_count += 1
                pos_sum += image_speed
                if this_pos_center[0] > rightmost_center[0]:
                    rightmost_center = this_pos_center
            else:
                idle_count += 1
        total_ids = len(self._id_to_tlwhs_history_map)
        if total_ids and total_ids > 4:  # Don't just consider two players
            if float(pos_count) / total_ids > group_threshhold:
                avg_x_speed = pos_sum / pos_count
                return avg_x_speed, rightmost_center
            elif float(neg_count) / total_ids > group_threshhold:
                avg_x_speed = neg_sum / neg_count
                return avg_x_speed, leftmost_center
        return 0, None

    def add_x_velocity(self, x_velocity_to_add):
        self._current_camera_box_speed_x += x_velocity_to_add

    def clamp(self, box: torch.Tensor):
        return clamp_box(box, self._video_frame.bounding_box())

    @staticmethod
    def _tlwh_to_tlbr(box):
        return torch.tensor(box[0], box[1], box[0] + box[2], box[1] + box[3])

    @staticmethod
    def _tlbr_to_tlwh(box):
        return torch.tensor(box[0], box[1], box[2] - box[0], box[3] - box[1])

    @staticmethod
    def _union(box1, box2):
        box = HockeyMOM._tlwh_to_tlbr(box1)
        box2 = HockeyMOM._tlwh_to_tlbr(box2)
        box[0] = min(box[0], box2[0])
        box[1] = min(box[1], box2[1])
        box[2] = max(box[2], box2[2])
        box[3] = max(box[3], box2[3])
        return HockeyMOM._tlbr_to_tlwh(box1)

    @classmethod
    def union_box(cls, box1, box2):
        assert box1.dtype == box2.dtype
        top_left = torch.min(box1[:2], box2[:2])
        bottom_right = torch.max(box1[2:], box2[2:])
        return torch.cat([top_left, bottom_right])

    @staticmethod
    def _make_pruned_map(map, allowed_keys):
        key_set = set(allowed_keys)
        new_map = dict()
        for map_key in map.keys():
            if map_key in key_set:
                new_map[map_key] = map[map_key]
        return new_map

    @staticmethod
    def _normalize_map(map, reference_map):
        for key in reference_map:
            if key not in map:
                map[key] = reference_map[key]
        return map

    @classmethod
    def box_str(cls, box):
        box_center = center(box)
        return (
            f"[x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}] "
            + f"w={box[2] - box[0]}, h={box[3] - box[1]}, "
            + f"center={box_center}"
        )

    @classmethod
    def print_box(cls, box):
        logger.info(cls.box_str(box))

    def get_current_bounding_box(self, ids=None):
        if ids is None:
            ids = self._online_ids
        if len(ids) == 0:
            return self._video_frame.bounding_box()

        tlwh_list = []
        for id in ids:
            # Can this be a select or gather one day?
            tlwh = self.get_tlwh(id)
            tlwh_list.append(tlwh_to_tlbr_single(tlwh))
        bounding_boxes = torch.stack(tlwh_list)
        min_tl, _ = torch.min(bounding_boxes[:, :2], dim=0)
        max_br, _ = torch.max(bounding_boxes[:, 2:], dim=0)
        # The containing bounding box is then
        containing_box = torch.cat([min_tl, max_br])
        return containing_box

    def shift_box_to_edge(self, box, strict: bool = False):
        """
        If a box is off the edge of the image, translate
        the box to be flush with the edge instead.
        """
        if strict:
            assert width(box) <= self._video_frame.width
            assert height(box) <= self._video_frame.height
        else:
            if width(box) > self._video_frame.width:
                logger.info(
                    f"ERROR: Width {width(box)} is too wide! Larger than video frame width of {self._video_frame.width}"
                )
            if height(box) > self._video_frame.height:
                logger.info(
                    f"ERROR: Height {height(box)} is too tall! Larger than video frame width of {self._video_frame.height}"
                )

        if box[0] < 0:
            box[2] += -box[0]
            box[0] += -box[0]
        elif box[2] >= self._video_frame.width:
            offset = box[2] - (self._video_frame.width - 1)
            box[0] -= offset
            box[2] -= offset

        if box[1] < 0:
            box[3] += -box[1]
            box[1] += -box[1]
        elif box[3] >= self._video_frame.height:
            offset = box[3] - (self._video_frame.height - 1)
            box[1] -= offset
            box[3] -= offset
        return box
