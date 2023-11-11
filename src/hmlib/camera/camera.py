import cameratransform as ct
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from fast_pytorch_kmeans import KMeans
import matplotlib.pyplot as plt
import pt_autograph as ptag

import torch

# import pt_autograph
from pt_autograph import pt_function

# import pt_autograph.flow.runner as runner

from hmlib.utils.box_functions import (
    width,
    height,
    center,
    center_x_distance,
    clamp_box,
    clamp_value,
    aspect_ratio,
    make_box_at_center,
    tlwh_to_tlbr_single,
    #translate_box_to_edge,
    shift_box_to_edge,
)

# nosec B101

X_SPEED_HISTORY_LENGTH = 5


def _as_scalar(tensor):
    # if isinstance(tensor, torch.Tensor):
    #     return tensor.item()
    return tensor


def create_camera(
    elevation_m: float,
    image_size_px=(3264, 2448),
    tilt_degrees: float = 45.0,
    roll_degrees: float = 0.0,
    focal_length: float = 6.2,
    sensor_size_mm: Tuple[float, float] = (6.17, 4.55),
) -> ct.Camera:
    """
    Create a camera object with the given parameters/position.
    """
    # initialize the camera
    cam = ct.Camera(
        ct.RectilinearProjection(
            focallength_mm=focal_length,
            sensor=sensor_size_mm,
            image=image_size_px,
        ),
        ct.SpatialOrientation(
            elevation_m=elevation_m,
            tilt_deg=tilt_degrees,
            roll_deg=roll_degrees,
        ),
    )
    return cam


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
        self._image_width = _as_scalar(image_width)
        self._image_height = _as_scalar(image_height)
        self._vertical_center = self._image_height / 2
        self._horizontal_center = self._image_width / 2
        self._bbox = torch.tensor(
            (0, 0, self._image_width - 1, self._image_height - 1),
            dtype=torch.float32,
            device=self._image_width.device,
        )

    def bounding_box(self):
        return self._bbox

    @property
    def width(self):
        return self._image_width

    @property
    def height(self):
        return self._image_height


class TlwhHistory(object):
    def __init__(self, id: int, video_frame: VideoFrame, max_history_length: int = 25):
        self.id_ = id
        # self._video_frame = video_frame
        self._max_history_length = max_history_length
        self._image_position_history = list()
        # self._spatial_position_history = list()
        self._spatial_distance_sum = 0.0
        # self._current_spatial_speed = 0.0
        self._current_spatial_x_speed = 0.0
        self._image_distance_sum = 0.0
        self._current_image_speed = 0.0
        self._current_image_x_speed = 0.0
        # self._spatial_speed_multiplier = 100.0

    @property
    def id(self):
        return self.id_

    def append(
        self, image_position: torch.Tensor, spatial_position: torch.Tensor = None
    ):
        current_historty_length = len(self._image_position_history)
        if current_historty_length >= self._max_history_length - 1:
            # trunk-ignore(bandit/B101)
            assert current_historty_length == self._max_history_length - 1
            # dist = np.linalg.norm()
            # removing_spatial_point = self._spatial_position_history[0]
            removing_image_point = self._image_position_history[0]
            # self._spatial_position_history = self._spatial_position_history[1:]
            self._image_position_history = self._image_position_history[1:]
            # old_spatial_distance = np.linalg.norm(
            # removing_spatial_point - self._spatial_position_history[0]
            # )
            # self._spatial_distance_sum -= old_spatial_distance
            old_image_distance = torch.linalg.norm(
                removing_image_point - self._image_position_history[0]
            )
            self._image_distance_sum -= old_image_distance
        if len(self._image_position_history) > 0:
            # new_spatial_distance = np.linalg.norm(
            # spatial_position - self._spatial_position_history[-1]
            # )
            # self._spatial_distance_sum += new_spatial_distance
            new_image_distance = torch.linalg.norm(
                image_position - self._image_position_history[-1]
            )
            self._image_distance_sum += new_image_distance
        # self._spatial_position_history.append(spatial_position)
        self._image_position_history.append(image_position)
        if len(self._image_position_history) > 1:
            # self._current_spatial_speed = (
            #     self._spatial_speed_multiplier
            #     * self._spatial_distance_sum
            #     / (len(self._spatial_position_history) - 1)
            # )
            self._current_image_speed = self._image_distance_sum / (
                len(self._image_position_history) - 1
            )
            if len(self._image_position_history) >= X_SPEED_HISTORY_LENGTH:
                self._current_image_x_speed = (
                    self._image_position_history[-1][0]
                    - self._image_position_history[-X_SPEED_HISTORY_LENGTH][0]
                ) / float(X_SPEED_HISTORY_LENGTH)
                # x speed from last X_SPEED_HISTORY_LENGTH frames
                # print(f"id {self.id_} has image x speed {self._current_image_x_speed}")
            else:
                self._current_image_x_speed = 0.0
        else:
            # self._current_spatial_speed = 0.0
            self._current_image_speed = 0.0
            self._current_image_x_speed = 0.0

    @property
    def current_image_x_speed(self):
        return self._current_image_x_speed

    @property
    def image_position_history(self):
        return self._image_position_history

    @staticmethod
    def center_point(tlwh):
        top = tlwh[0]
        left = tlwh[1]
        width = tlwh[2]
        height = tlwh[3]
        x_center = left + width / 2
        y_center = top + height / 2
        return torch.tensor((x_center, y_center), dtype=torch.float32)

    def __len__(self):
        length = len(self._image_position_history)
        # trunk-ignore(bandit/B101)
        # assert length == len(self._spatial_position_history)
        return length

    # @property
    # def spatial_speed(self):
    #     return self._current_spatial_speed

    @property
    def image_speed(self):
        return self._current_image_speed


class Clustering:
    def __init__(self):
        pass


CAMERA_TYPE_MAX_SPEEDS = {
    "gopro": 300.0,
    "zhiwei": 200.0,
}


class HockeyMOM:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        device,
        max_history: int = 26,
        speed_history: int = 26,
    ):
        self._device = device
        self._video_frame = VideoFrame(
            image_width=self._to_scalar_float(image_width),
            image_height=self._to_scalar_float(image_height),
        )
        self._clamp_box = self._video_frame.bounding_box()
        self._online_tlwhs_history = list()
        self._max_history = max_history
        # self._speed_history = speed_history
        self._online_ids = set()
        # self._id_to_speed_map = dict()
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

        # self._camera_type = "zhiwei"
        self._camera_type = "gopro"

        # self._camera_box_max_speed_x = max(image_width / 150.0, 12.0)
        # self._camera_box_max_speed_y = max(image_height / 150.0, 12.0)
        # self._camera_box_max_speed_x = max(image_width / 200.0, 12.0)
        # self._camera_box_max_speed_y = max(image_height / 200.0, 12.0)
        # self._camera_box_max_speed_x = max(image_width / 300.0, 12.0)
        # self._camera_box_max_speed_y = max(image_height / 300.0, 12.0)
        self._camera_box_max_speed_x = self._to_scalar_float(
            max(image_width / CAMERA_TYPE_MAX_SPEEDS[self._camera_type], 12.0)
        )
        self._camera_box_max_speed_y = self._to_scalar_float(
            max(image_height / CAMERA_TYPE_MAX_SPEEDS[self._camera_type], 12.0)
        )
        print(
            f"Camera Max speeds: x={self._camera_box_max_speed_x}, y={self._camera_box_max_speed_y}"
        )

        # Max acceleration in pixels per frame
        # self._camera_box_max_accel_x = 6
        # self._camera_box_max_accel_y = 6
        # self._camera_box_max_accel_x = 3
        # self._camera_box_max_accel_y = 3
        self._camera_box_max_accel_x = self._to_scalar_float(1)
        self._camera_box_max_accel_y = self._to_scalar_float(1)
        # self._camera_box_max_accel_x = max(self._camera_box_max_speed_x / 20.0, 1)
        # self._camera_box_max_accel_y = max(self._camera_box_max_speed_y / 20.0, 1)
        print(
            f"Camera Max acceleration: dx={self._camera_box_max_accel_x}, dy={self._camera_box_max_accel_y}"
        )

        self._last_acceleration_dx = self._to_scalar_float(0)
        self._last_acceleration_dy = self._to_scalar_float(0)

        #
        # Zoom velocity
        #
        self._camera_box_size_change_velocity_x = self._to_scalar_float(0)
        self._camera_box_size_change_velocity_y = self._to_scalar_float(0)

        # self._camera_box_max_size_change_velocity_x = self._to_scalar_float(0)
        # self._camera_box_max_size_change_velocity_y = self._to_scalar_float(0)
        self._camera_box_max_size_change_velocity_x = self._to_scalar_float(2)
        self._camera_box_max_size_change_velocity_y = self._to_scalar_float(2)

        # Create the camera transofrmer
        # self._camera = create_camera(
        #     elevation_m=3,
        #     tilt_degrees=45,
        #     roll_degrees=5,  # <-- looks correct for Vallco left penalty box glass
        #     focal_length=12,
        #     sensor_size_mm=(73, 4.55),
        #     image_size_px=(image_width, image_height),
        # )

    def _to_scalar_float(self, scalar_float):
        return torch.tensor(scalar_float, dtype=torch.float32, device=self._device)

    def get_speed(self):
        return math.sqrt(
            self._current_camera_box_speed_x * self._current_camera_box_speed_x
            + self._current_camera_box_speed_y * self._current_camera_box_speed_y
        )

    def is_fast(self, speed: float = 7):
        # return abs(self._current_camera_box_speed_x) > speed or abs(self._current_camera_box_speed_y) > speed
        return self.get_speed() >= speed

    def control_speed(self, abs_max_x: float, abs_max_y: float, set_speed_x: bool):
        if set_speed_x:
            if self._current_camera_box_speed_x < 0:
                self._current_camera_box_speed_x = -abs_max_x
            elif self._current_camera_box_speed_x > 0:
                self._current_camera_box_speed_x = abs_max_x

        if abs(self._current_camera_box_speed_x) > abs_max_x:
            if self._current_camera_box_speed_x < 0:
                self._current_camera_box_speed_x = -abs_max_x
            else:
                self._current_camera_box_speed_x = abs_max_x
        if abs(self._current_camera_box_speed_y) > abs_max_y:
            if self._current_camera_box_speed_y < 0:
                self._current_camera_box_speed_y = -abs_max_y
            else:
                self._current_camera_box_speed_y = abs_max_y

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
        # Result will be 3D with all Z axis as zeroes, so just trim that dim
        # if len(self._online_image_center_points):
        #     self._online_spatial = self._camera.spaceFromImage(
        #         self._online_image_center_points
        #     )
        # else:
        #     self._online_spatial = np.array([])

        # Add to history
        prev_dict = self._id_to_tlwhs_history_map
        self._id_to_tlwhs_history_map = dict()
        # self._id_to_speed_map = dict()
        # trunk-ignore(ruff/B905)
        for id, image_pos in zip(self._online_ids, self._online_tlws):
            hist = prev_dict.get(
                id.item(), TlwhHistory(id=id.item(), video_frame=self._video_frame)
            )
            hist.append(image_position=image_pos)
            self._id_to_tlwhs_history_map[id.item()] = hist

    def reset_clusters(self):
        self._cluster_label_ids = dict()
        self._largest_cluster_label = dict()
        self._cluster_counts = dict()

    def calculate_clusters(self, n_clusters, device="cuda"):
        self._cluster_label_ids[n_clusters] = dict()
        if len(self._online_image_center_points) < n_clusters:
            # Minimum clusters that we can have is the minimum number of objects
            max_clusters = len(self._online_image_center_points)
            if not max_clusters:
                return
            elif max_clusters < n_clusters:
                return
        self._largest_cluster_label[n_clusters] = None
        labels = None
        if n_clusters not in self._kmeans_objects:
            self._kmeans_objects[n_clusters] = KMeans(
                n_clusters=n_clusters,
                mode="euclidean",
            )
        torch_tensors = []
        for n in self._online_image_center_points:
            # print(n)
            torch_tensors.append(n.to(device))
        tt = torch.cat(torch_tensors, dim=0)
        tt = torch.reshape(tt, (len(torch_tensors), 2))
        # print(tt)
        labels = self._kmeans_objects[n_clusters].fit_predict(tt)
        # print(labels)
        self._cluster_counts[n_clusters] = [0 for i in range(n_clusters)]
        cluster_counts = self._cluster_counts[n_clusters]
        cluster_label_ids = self._cluster_label_ids[n_clusters]
        id_count = len(self._online_ids)
        # trunk-ignore(bandit/B101)
        assert id_count == labels.shape[0]
        for i in range(id_count):
            id = self._online_ids[i].item()
            cluster_label = labels[i].item()
            cluster_counts[cluster_label] += 1
            if cluster_label not in cluster_label_ids:
                cluster_label_ids[cluster_label] = [id]
            else:
                cluster_label_ids[cluster_label].append(id)
        for cluster_label, cluster_id_list in self._cluster_label_ids[
            n_clusters
        ].items():
            if self._largest_cluster_label[n_clusters] is None:
                self._largest_cluster_label[n_clusters] = cluster_label
            elif len(cluster_id_list) > len(
                cluster_label_ids[self._largest_cluster_label[n_clusters]]
            ):
                self._largest_cluster_label[n_clusters] = cluster_label

    def get_largest_cluster_id_set(self, n_clusters):
        if n_clusters not in self._largest_cluster_label:
            return set()
        return set(
            self._cluster_label_ids[n_clusters][self._largest_cluster_label[n_clusters]]
        )

    def prune_not_in_largest_cluster(self, n_clusters, ids):
        # TODO: return a tensor?
        largest_cluster_set = self.get_largest_cluster_id_set(n_clusters)
        result_ids = []
        for id in ids:
            if id.item() in largest_cluster_set:
                result_ids.append(id)
        return result_ids

    @property
    def largest_cluster_label(self, n_clusters):
        return self._largest_cluster_label[n_clusters]

    @property
    def cluster_size(self, n_clusters, cluster_label):
        return len(self._cluster_label_ids[n_clusters][cluster_label])

    @property
    def online_image_center_points(self):
        return self._online_image_center_points

    @property
    def video(self):
        return self._video_frame

    # @property
    # def camera(self):
    #     return self._camera

    @property
    def clamp_box(self):
        return self._clamp_box

    def get_image_tracking(self, online_ids):
        results = []
        for id in online_ids:
            results.append(
                self._id_to_tlwhs_history_map[id.item()].image_position_history
            )
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

    # @classmethod
    # def _box_centers(cls, box):
    #     return torch.tensor(
    #         [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=box.dtype
    #     )

    # @classmethod
    # def _box_centers(cls, bboxes):
    #     """
    #     Compute the centers of multiple bounding boxes.

    #     Parameters:
    #     bboxes (Tensor): A tensor containing multiple bounding boxes, each with the form [x1, y1, x2, y2].

    #     Returns:
    #     Tensor: A tensor containing the center coordinates [cx, cy] for each bounding box.
    #     """
    #     single = False
    #     if len(bboxes.shape) == 1:
    #         bboxes = bboxes.unsqueeze(dim=0)
    #         single = True
    #     x1 = bboxes[:, 0]
    #     y1 = bboxes[:, 1]
    #     x2 = bboxes[:, 2]
    #     y2 = bboxes[:, 3]
    #     cxs = (x1 + x2) / 2
    #     cys = (y1 + y2) / 2
    #     centers = torch.stack((cxs, cys), dim=1)
    #     if single:
    #         centers = centers.squeeze(dim=0)
    #     return centers

    # @classmethod
    # def _clamp(cls, box, clamp_box):
    #     assert box.device == clamp_box.device
    #     clamped_box = torch.empty_like(box)
    #     clamped_box[0] = torch.clamp(box[0], min=clamp_box[0], max=clamp_box[2])
    #     clamped_box[1] = torch.clamp(box[1], min=clamp_box[1], max=clamp_box[3])
    #     clamped_box[2] = torch.clamp(box[2], min=clamp_box[0], max=clamp_box[2])
    #     clamped_box[3] = torch.clamp(box[1], min=clamp_box[1], max=clamp_box[3])
    #     return clamped_box
    #     # return torch.tensor(
    #     #     [
    #     #         torch.max(box[0], clamp_box[0]),
    #     #         torch.max(box[1], clamp_box[1]),
    #     #         torch.min(box[2], clamp_box[2]),
    #     #         torch.min(box[3], clamp_box[3]),
    #     #     ],
    #     #     dtype=box.dtype,
    #     # )

    def clamp(self, box: torch.Tensor):
        return clamp_box(box, self._video_frame.bounding_box())
        # return self._clamp(
        #     box,
        #     torch.tensor(
        #         [0, 0, self._video_frame.width - 1, self._video_frame.height - 1],
        #         dtype=torch.float32,
        #     ),
        # )

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
        # box = box1.clone()
        # box[0] = torch.min(box[0], box2[0])
        # box[1] = torch.min(box[1], box2[1])
        # box[2] = torch.max(box[2], box2[2])
        # box[3] = torch.max(box[3], box2[3])
        # return box
        assert box1.dtype == box2.dtype
        top_left = torch.min(box1[:2], box2[:2])
        bottom_right = torch.max(box1[2:], box2[2:])
        return torch.cat([top_left, bottom_right])
        return torch.tensor(
            [
                torch.min(box1[0], box2[0]),
                torch.min(box1[1], box2[1]),
                torch.max(box1[2], box2[2]),
                torch.max(box1[3], box2[3]),
            ],
            dtype=box1.dtype,
        )

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
        print(cls.box_str(box))

    @classmethod
    def move_box_center_to_point(cls, box, point):
        w = box[2] - box[0]
        h = box[3] - box[1]
        box[0] = point[0] - w / 2
        box[2] = point[0] + w / 2
        box[1] = point[1] - h / 2
        box[3] = point[1] + h / 2
        return box

    # def get_current_bounding_box(self, ids=None):
    #     bounding_intbox = self._video_frame.bounding_box()

    #     bounding_intbox[0], bounding_intbox[2] = bounding_intbox[2], bounding_intbox[1]
    #     bounding_intbox[1], bounding_intbox[3] = bounding_intbox[3], bounding_intbox[1]

    #     if ids is None:
    #         ids = self._online_ids

    #     for id in ids:
    #         tlwh = self.get_tlwh(id)
    #         x1 = tlwh[0]
    #         y1 = tlwh[1]
    #         w = tlwh[2]
    #         h = tlwh[3]
    #         intbox = np.array((x1, y1, x1 + w + 0.5, y1 + h + 0.5), dtype=np.int32)
    #         bounding_intbox[0] = min(bounding_intbox[0], intbox[0])
    #         bounding_intbox[1] = min(bounding_intbox[1], intbox[1])
    #         bounding_intbox[2] = max(bounding_intbox[2], intbox[2])
    #         bounding_intbox[3] = max(bounding_intbox[3], intbox[3])
    #     return bounding_intbox

    # def get_current_bounding_box(self, ids=None):
    #     bounding_intbox = self._video_frame.bounding_box().tolist()

    #     bounding_intbox[0], bounding_intbox[2] = bounding_intbox[2], bounding_intbox[1]
    #     bounding_intbox[1], bounding_intbox[3] = bounding_intbox[3], bounding_intbox[1]

    #     if ids is None:
    #         ids = self._online_ids

    #     for id in ids:
    #         tlwh = self.get_tlwh(id)
    #         x1 = tlwh[0]
    #         y1 = tlwh[1]
    #         w = tlwh[2]
    #         h = tlwh[3]
    #         intbox = np.array((x1, y1, x1 + w + 0.5, y1 + h + 0.5), dtype=np.int32)
    #         bounding_intbox[0] = min(bounding_intbox[0], intbox[0])
    #         bounding_intbox[1] = min(bounding_intbox[1], intbox[1])
    #         bounding_intbox[2] = max(bounding_intbox[2], intbox[2])
    #         bounding_intbox[3] = max(bounding_intbox[3], intbox[3])
    #     return torch.tensor(bounding_intbox, dtype=torch.float32)

    def get_current_bounding_box(self, ids=None):
        # bounding_intbox = self._video_frame.bounding_box()

        # bounding_intbox[0], bounding_intbox[2] = bounding_intbox[2], bounding_intbox[0]
        # bounding_intbox[1], bounding_intbox[3] = bounding_intbox[3], bounding_intbox[1]

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

    def ratioed_expand(self, box):
        ew = self._video_frame.width / 10
        eh = self._video_frame.height / 10
        diff = torch.tensor([-ew / 2, -eh / 2, ew / 2, eh / 2])

        # Expand less in the velocity-trailing x direction
        # (this gives the effect of "leading" the movement)
        if self._current_camera_box_speed_x < 0:
            diff[2] /= 4
        elif self._current_camera_box_speed_x > 0:
            diff[0] /= 4

        box = box + diff
        box = self.clamp(box)
        return box

    @classmethod
    def scale(cls, w, h, x, y, maximum=True):
        nw = y * w / h
        nh = x * h / w
        if int(maximum) ^ int(_as_scalar(nw >= x)):
            # if torch.pow(int(maximum), torch.Tensor(nw >= x)):
            return nw or 1, y
        return x, nh or 1

    # def make_box_proper_aspect_ratio(
    #     runner.maybe_run_converted():
    #                             _inner_update,
    #                             outputs,
    #                             frame_index,
    #                             self.img_size,
    #                             imgs[frame_index].cuda()
    #                         )

    def make_box_proper_aspect_ratio(
        self,
        frame_id: int,
        the_box: torch.Tensor,
        desired_aspect_ratio: float,
        max_in_aspec_ratio: bool,
        verbose: bool = False,
        extra_validation: bool = True,
    ):
        return self._make_box_proper_aspect_ratio(
            frame_id=frame_id,
            the_box=the_box,
            desired_aspect_ratio=desired_aspect_ratio,
            max_in_aspec_ratio=max_in_aspec_ratio,
            verbose=verbose,
            extra_validation=extra_validation,
        )

    def _make_box_proper_aspect_ratio(
        self,
        frame_id: int,
        the_box: torch.Tensor,
        desired_aspect_ratio: float,
        max_in_aspec_ratio: bool,
        verbose: bool = False,
        extra_validation: bool = True,
    ):
        """
        Make the given box the specified aspect ratio about the box's center.
        The final box must be within the initial box.
        """

        box_center = center(the_box)

        w = width(the_box)
        if w > self._video_frame.width:
            diff = w - self._video_frame.width
            the_box[0] -= diff / 2
            the_box[2] += diff / 2
            w = width(the_box)
        h = height(the_box)
        if h > self._video_frame.height:
            diff = w - self._video_frame.width
            the_box[1] -= diff / 2
            the_box[3] += diff / 2
            h = height(the_box)

        # Apparently we are enforcing it here to not be too small
        if w < self._video_frame.width / 3:
            w = self._video_frame.width / 3
        if h < self._video_frame.height / 2:
            h = self._video_frame.height / 2

        #assert w <= self._video_frame.width
        #assert h <= self._video_frame.height

        if True:
            if w / h > desired_aspect_ratio:
                # Constrain by height
                new_h = h
                new_w = new_h * desired_aspect_ratio
            else:
                # Constrain by width
                new_w = w
                new_h = new_w / desired_aspect_ratio
        else:
            new_w_1 = w
            new_h_1 = w / desired_aspect_ratio

            new_w_2 = h * desired_aspect_ratio
            new_h_2 = h
            if not max_in_aspec_ratio:
                if new_h_1 > self._video_frame.height:
                    new_h_1 = self._video_frame.height
                    new_w_1 = new_h_1 * desired_aspect_ratio
                if new_w_1 > self._video_frame.width:
                    new_w_1 = self._video_frame.width
                    new_h_1 = new_w_1 / desired_aspect_ratio
                new_w = new_w_1
                new_h = new_h_1
            else:
                new_w, new_h = self.scale(
                    new_w_1,
                    new_h_1,
                    (self._video_frame.height * desired_aspect_ratio),
                    self._video_frame.height,
                    maximum=True,
                )

        w_diff = new_w - self._video_frame.width
        if w_diff > 0 and w_diff < self._epsilon:
            new_w = float(self._video_frame.width)
        h_diff = new_h - self._video_frame.height
        if h_diff > 0 and h_diff < self._epsilon:
            new_h = float(self._video_frame.height)

        if extra_validation:
            assert new_w <= self._video_frame.width
            assert new_h <= self._video_frame.height

        # floating-point math gets funky sometimes and overflows
        # above the max width or height (due to python sucking,
        # most likely)
        box_center = box_center.trunc()
        #center[0] = float(int(center[0]))
        #center[1] = float(int(center[1]))

        new_box = make_box_at_center(box_center, new_w, new_h)

        if verbose:
            print(f"frame_id={frame_id}, ar={aspect_ratio(new_box)}")
        assert torch.isclose(aspect_ratio(new_box), desired_aspect_ratio)

        if extra_validation:
            # Damned numerics defy logic
            ww = width(new_box)
            hh = height(new_box)
            assert ww <= (self._video_frame.width + self._epsilon)
            assert hh <= (self._video_frame.height + self._epsilon)

        assert torch.isclose(aspect_ratio(new_box), desired_aspect_ratio)

        return new_box

    def apply_fixed_edge_scaling(
        self, box, edge_scaling_factor: float, verbose: bool = False
    ):
        current_center = center(box)
        w = width(box)
        # h = height(box)
        # ar = w / h
        dist_from_center_x = 0
        half_frame_width = float(self._video_frame.width) / 2
        if current_center[0] < half_frame_width:
            dist_from_center_x = half_frame_width - current_center[0]
        elif current_center[0] > half_frame_width:
            dist_from_center_x = current_center[0] - half_frame_width
        dist_center_ratio = dist_from_center_x / float(self._video_frame.width)
        if verbose:
            print(f"dist_center_ratio={dist_center_ratio}")
        box_scaling = edge_scaling_factor * dist_center_ratio
        width_reduction = w * box_scaling
        if verbose:
            print(f"width_reduction={width_reduction}")
        if width_reduction == 0.0:
            return box

        ar = w / height(box)
        h = self._video_frame.height
        w = h * ar
        w -= width_reduction
        h = w / ar
        new_box = make_box_at_center(current_center, w, h)

        if current_center[0] < half_frame_width:
            # shift left the amount
            new_box[0] -= width_reduction / 2
            new_box[2] -= width_reduction / 2
        elif current_center[0] > half_frame_width:
            new_box[0] += width_reduction / 2
            new_box[2] += width_reduction / 2
        return self.shift_box_to_edge(new_box)

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
                print(
                    f"ERROR: Width {width(box)} is too wide! Larger than video frame width of {self._video_frame.width}"
                )
            if height(box) > self._video_frame.height:
                print(
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

        #return translate_box_to_edge(box=box, bounds=self._video_frame.bounding_box())

    def apply_box_velocity(self, box, scale_speed: float, verbose: bool = False):
        dx = self._current_camera_box_speed_x * scale_speed
        dy = self._current_camera_box_speed_y * scale_speed
        # if verbose:
        #     print(f"Moving box by {dx} x {dy}")
        return box + torch.tensor([dx, dy, dx, dy], dtype=torch.float32, device=box.device)

    def adjust_veclocity_based_upon_new_box(
        self,
        proposed_point: List[int],
        last_point: List[int],
    ):
        dx = proposed_point[0] - last_point[0]
        dy = proposed_point[1] - last_point[1]
        # print(f"want: dx={dx}, dy={dy}")

        acceleration_dx = dx - self._current_camera_box_speed_x
        acceleration_dy = dy - self._current_camera_box_speed_y

        allowed_acceleration_dx = clamp_value(
            -self._camera_box_max_accel_x, acceleration_dx, self._camera_box_max_accel_x
        )
        allowed_acceleration_dy = clamp_value(
            -self._camera_box_max_accel_y, acceleration_dy, self._camera_box_max_accel_y
        )

        # if verbose:
        #     print(
        #         f"attempted accel x : {acceleration_dx}, allowed: {allowed_acceleration_dx}"
        #     )
        #     print(
        #         f"attempted accel y : {acceleration_dy}, allowed: {allowed_acceleration_dy}"
        #     )

        #
        # now adjust velocity based on the allowed accelerations
        #
        old_speed_x = self._current_camera_box_speed_x
        # old_speed_y = self._current_camera_box_speed_y

        self._last_acceleration_dx = allowed_acceleration_dx
        self._last_acceleration_dy = allowed_acceleration_dy

        self._current_camera_box_speed_x += allowed_acceleration_dx
        self._current_camera_box_speed_y += allowed_acceleration_dy

        #
        # Check if the new velocity is a change of direction
        #
        if old_speed_x < 0 and self._current_camera_box_speed_x >= 0:
            self._current_camera_box_speed_reversed_x = True
        elif old_speed_x > 0 and self._current_camera_box_speed_x <= 0:
            self._current_camera_box_speed_reversed_x = True
        # if old_speed_y < 0 and self._current_camera_box_speed_y >= 0:
        #     self._current_camera_box_speed_reversed_y = True
        # elif old_speed_y > 0 and self._current_camera_box_speed_y <= 0:
        #     self._current_camera_box_speed_reversed_y = True

        self._current_camera_box_speed_x = clamp_value(
            -self._camera_box_max_speed_x,
            self._current_camera_box_speed_x,
            self._camera_box_max_speed_x,
        )
        self._current_camera_box_speed_y = clamp_value(
            -self._camera_box_max_speed_y,
            self._current_camera_box_speed_y,
            self._camera_box_max_speed_y,
        )
        # print(
        #     f"change in speed speed: {old_speed_x} -> {self._current_camera_box_speed_y}"
        # )

    def get_next_temporal_box(
        self,
        proposed_new_box: List[int],
        last_box: List[int],
        pre_clamp_to_video_frame: bool = True,
        scale_speed: float = 1.0,
        verbose: bool = False,
    ):
        if last_box is not None:
            self.adjust_veclocity_based_upon_new_box(
                center(proposed_new_box), center(last_box)
            )
        else:
            last_box = proposed_new_box
        moved_box = self.apply_box_velocity(
            last_box, scale_speed=scale_speed, verbose=verbose
        )
        self._curtail_speed_at_edges(moved_box)
        return moved_box

    def smooth_resize_box(self, proposed_new_box, last_box):
        if last_box is None:
            last_box = proposed_new_box
        #
        # Now we curtail in their allowed height and width changes per frame
        # This should bew simpler, just a max allowable change in size
        #
        proposed_width = proposed_new_box[2] - proposed_new_box[0]
        proposed_height = proposed_new_box[3] - proposed_new_box[1]

        last_width = last_box[2] - last_box[0]
        last_height = last_box[3] - last_box[1]

        proposed_width_change = proposed_width - last_width
        proposed_height_change = proposed_height - last_height

        allowed_width_change = clamp_value(
            -self._camera_box_max_size_change_velocity_x,
            proposed_width_change,
            self._camera_box_max_size_change_velocity_x,
        )
        allowed_height_change = clamp_value(
            -self._camera_box_max_size_change_velocity_y,
            proposed_height_change,
            self._camera_box_max_size_change_velocity_y,
        )
        return make_box_at_center(
            center(proposed_new_box),
            last_width + allowed_width_change,
            last_height + allowed_height_change,
        )

    def old_get_next_temporal_box(
        self,
        proposed_new_box: List[int],
        last_box: List[int],
        pre_clamp_to_video_frame: bool = True,
        scale_speed: float = 1.0,
        verbose: bool = False,
    ):
        # if verbose:
        #     print(f"\nproposed_new_box: {self.box_str(proposed_new_box)}")
        #     if last_box is not None:
        #         print(f"last_box: {self.box_str(last_box)}")

        next_box = proposed_new_box.copy()

        if pre_clamp_to_video_frame:
            video_frame = self._video_frame.bounding_box()
            self._clamp(next_box, video_frame)

        if last_box is None:
            return next_box

        # First, we apply the center's proposed change
        proposed_center = center(proposed_new_box)
        last_center = center(last_box)
        dx = proposed_center[0] - last_center[0]
        dy = proposed_center[1] - last_center[1]

        acceleration_dx = dx - self._current_camera_box_speed_x
        acceleration_dy = dy - self._current_camera_box_speed_y

        if acceleration_dx < 0:
            allowed_acceleration_dx = max(
                acceleration_dx, -self._camera_box_max_accel_x
            )
        else:
            allowed_acceleration_dx = min(acceleration_dx, self._camera_box_max_accel_x)

        if acceleration_dy < 0:
            allowed_acceleration_dy = max(
                acceleration_dy, -self._camera_box_max_accel_y
            )
        else:
            allowed_acceleration_dy = min(acceleration_dy, self._camera_box_max_accel_y)

        # if verbose:
        #     print(
        #         f"attempted accel x : {acceleration_dx}, allowed: {allowed_acceleration_dx}"
        #     )
        #     print(
        #         f"attempted accel y : {acceleration_dy}, allowed: {allowed_acceleration_dy}"
        #     )

        #
        # now adjust velocity based on the allowed accelerations
        #
        # self._current_camera_box_speed_reversed_x = False
        # self._current_camera_box_speed_reversed_y = False

        old_s_x = self._current_camera_box_speed_x
        old_s_y = self._current_camera_box_speed_y

        self._last_acceleration_dx = allowed_acceleration_dx
        self._last_acceleration_dy = allowed_acceleration_dy

        self._current_camera_box_speed_x += allowed_acceleration_dx
        self._current_camera_box_speed_y += allowed_acceleration_dy

        if old_s_x < 0 and self._current_camera_box_speed_x >= 0:
            self._current_camera_box_speed_reversed_x = True
        elif old_s_x > 0 and self._current_camera_box_speed_x <= 0:
            self._current_camera_box_speed_reversed_x = True

        if old_s_y < 0 and self._current_camera_box_speed_y >= 0:
            self._current_camera_box_speed_reversed_y = True
        elif old_s_y > 0 and self._current_camera_box_speed_y <= 0:
            self._current_camera_box_speed_reversed_y = True

        if self._current_camera_box_speed_x < 0:
            self._current_camera_box_speed_x = max(
                self._current_camera_box_speed_x, -self._camera_box_max_speed_x
            )
        else:
            self._current_camera_box_speed_x = min(
                self._current_camera_box_speed_x, self._camera_box_max_speed_x
            )

        if self._current_camera_box_speed_y < 0:
            self._current_camera_box_speed_y = max(
                self._current_camera_box_speed_y, -self._camera_box_max_speed_y
            )
        else:
            self._current_camera_box_speed_y = min(
                self._current_camera_box_speed_y, self._camera_box_max_speed_y
            )

        if True:
            #
            # Now we curtail in ther allowed height and width changes per frame
            # This should bew simpler, just a max allowable change in size
            #
            proposed_width = proposed_new_box[2] - proposed_new_box[0]
            proposed_height = proposed_new_box[3] - proposed_new_box[1]

            last_width = last_box[2] - last_box[0]
            last_height = last_box[3] - last_box[1]

            proposed_width_change = proposed_width - last_width
            proposed_height_change = proposed_height - last_height

            if proposed_width_change < 0:
                allowed_width_change = max(
                    proposed_width_change, -self._camera_box_max_size_change_velocity_x
                )
            else:
                allowed_width_change = min(
                    proposed_width_change, self._camera_box_max_size_change_velocity_y
                )

            if proposed_height_change < 0:
                allowed_height_change = max(
                    proposed_height_change, -self._camera_box_max_size_change_velocity_x
                )
            else:
                allowed_height_change = min(
                    proposed_height_change, self._camera_box_max_size_change_velocity_x
                )

            # Apply the height change against the center
            # new_last_box = np.array(
            #     (
            #         last_box[0] - allowed_width_change / 2,
            #         last_box[1] - allowed_height_change / 2,
            #         last_box[2] + allowed_width_change / 2,
            #         last_box[3] + allowed_height_change / 2,
            #     ),
            #     dtype=np.float32,
            # )

            d0_mul = 1.0
            d1_mul = 1.0
            d2_mul = 1.0
            d3_mul = 1.0
            # Allow more width change in the direction of movement
            # if self._current_camera_box_speed_x < 0:
            #     d0_mul = 2.0
            # elif self._current_camera_box_speed_x > 0:
            #     d2_mul = 2.0
            # if self._current_camera_box_speed_y < 0:
            #     d1_mul = 2.0
            # elif self._current_camera_box_speed_y > 0:
            #     d3_mul = 2.0

            new_last_box = [
                last_box[0] - d0_mul * allowed_width_change / 2.0,
                last_box[1] - d1_mul * allowed_height_change / 2.0,
                last_box[2] + d2_mul * allowed_width_change / 2.0,
                last_box[3] + d3_mul * allowed_height_change / 2.0,
            ]
        else:
            new_last_box = last_box.copy()

        moved_box = self.apply_box_velocity(
            new_last_box, scale_speed=scale_speed, verbose=verbose
        )

        self._curtail_speed_at_edges(moved_box)

        # moved_box = self.shift_box_to_edge(moved_box)
        # moved_box = self.clamp(moved_box)

        # if verbose:
        #     print(f"moved_box: {self.box_str(moved_box)}")
        return moved_box

    def did_direction_change(
        self, dx: bool = True, dy: bool = True, reset: bool = False
    ):
        if reset:
            if dx:
                self._current_camera_box_speed_reversed_x = False
            if dy:
                self._current_camera_box_speed_reversed_y = False
            return None
        return (dx and self._current_camera_box_speed_reversed_x) or (
            dy and self._current_camera_box_speed_reversed_y
        )

    def set_direction_changed(self, dx: bool = True, dy: bool = True):
        if dx:
            self._current_camera_box_speed_reversed_x = True
        if dy:
            self._current_camera_box_speed_reversed_y = True
        return dx or dy

    def get_velocity_and_acceleratrion_xy(self):
        return (
            self._current_camera_box_speed_x,
            self._current_camera_box_speed_y,
            self._last_acceleration_dx,
            self._last_acceleration_dy,
        )

    def curtail_velocity_if_outside_box(self, current_box, bounding_box):
        if current_box[0] <= bounding_box[0] and self._current_camera_box_speed_x < 0:
            self._current_camera_box_speed_x /= 2
        if current_box[2] >= bounding_box[2] and self._current_camera_box_speed_x > 0:
            self._current_camera_box_speed_x /= 2
        if current_box[1] <= bounding_box[1] and self._current_camera_box_speed_y < 0:
            self._current_camera_box_speed_y /= 2
        if current_box[3] >= bounding_box[3] and self._current_camera_box_speed_y > 0:
            self._current_camera_box_speed_y /= 2

    def is_box_at_left_right_edge(self, box):
        if box[0] <= 0 or box[2] >= self._clamp_box[2]:
            return True
        return False

    def is_box_at_right_edge(self, box):
        return box[2] >= self._clamp_box[2]

    def is_box_at_left_edge(self, box):
        return box[0] <= 0

    def _curtail_speed_at_edges(self, box):
        """
        If we are on (or past) and edge, cut the speed in half.
        Setting to zero tends to make it have too much of a "bounce" effect,
        where it appears to bounce off the wall too quickly with respect
        to its speed before it hits the wall.
        """
        self.curtail_velocity_if_outside_box(box, self._clamp_box)
        # if box[0] <= 0 or box[2] >= self._clamp_box[2]:
        #     self._current_camera_box_speed_x /= 2.
        # if box[1] <= 0 or box[3] >= self._clamp_box[3]:
        #     self._current_camera_box_speed_y /= 2.


if __name__ == "__main__":
    """
    Display an image projected to a 2D birds-eye view
    """
    file_path: str = "/home/colivier/src/hockeymom/h-demo/frame/00000.png"
    for var in range(1, 200, 1):
        print(f"focal_length={var}")
        # print(f"sensor_size={var}")
        # print(f'tilt_degrees={var}')
        # print(f'roll_degrees={var}')
        hockey_mom = HockeyMOM(
            image_width=4096, image_height=1080, scale_width=0, scale_height=0
        )
        get_image_top_view(
            image_file=file_path,
            camera=hockey_mom.camera,
            dimensions_2d_m=(-30.5, 35.5, 0, 13),
            # dimensions_2d_m=(-100, 100, 0, 13),
        )
