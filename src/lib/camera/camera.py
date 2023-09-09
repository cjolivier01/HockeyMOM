import cameratransform as ct
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from fast_pytorch_kmeans import KMeans

import torch

# pytorch=1.7.0=py3.8_cuda10.2.89_cudnn7.6.5_0    pytorch
# torchvision=0.8.0=py38_cu102    pytorch

# nosec B101


def create_camera(
    elevation_m: float,
    image_size_px=(3264, 2448),
    tilt_degrees: float = 45.0,
    roll_degrees: float = 0.0,
    focal_length: float = 6.2,
    sensor_size_mm: Tuple[float, float] = (6.17, 4.55),
) -> ct.Camera:
    """
    Create a camera object witht he given parameters/position.
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


def width(box):
    return box[2] - box[0] + 1.0


def height(box):
    return box[3] - box[1] + 1.0


def center(box):
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def center_distance(box1, box2) -> float:
    if box1 is None or box2 is None:
        return 0.0
    c1 = center(box1)
    c2 = center(box2)
    if c1 == c2:
        return 0.0
    w = c2[0] - c1[0]
    h = c2[1] - c1[1]
    return math.sqrt(w * w + h * h)


# def scale(box, scale_width, scale_height):
#     w = width(box) * scale_width
#     h = height(box) * scale_height
#     c = center(box)
#     return np.array((c[0] - w / 2.0, c[1] - h / 2.0, c[0] + w / 2.0, c[1] + h / 2.0))


def aspect_ratio(box):
    return width(box) / height(box)


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
    return np.isclose(f1, f2, rtol=0.01, atol=0.01)


class VideoFrame(object):
    def __init__(
        self, image_width: int, image_height: int, scale_width=0.1, scale_height=0.05
    ):
        self._image_width = image_width
        self._image_height = image_height
        self._vertical_center = image_height / 2
        self._horizontal_center = image_width / 2
        self._scale_width = scale_width
        self._scale_height = scale_height

    def box(self):
        return np.array(
            (0, 0, self._image_width - 1, self._image_height - 1), dtype=np.int32
        )

    @property
    def width(self):
        return self._image_width

    @property
    def height(self):
        return self._image_height


class TlwhHistory(object):
    def __init__(self, id: int, video_frame: VideoFrame, max_history_length: int = 25):
        self.id_ = id
        self._video_frame = video_frame
        self._max_history_length = max_history_length
        self._image_position_history = list()
        self._spatial_position_history = list()
        self._spatial_distance_sum = 0.0
        self._current_spatial_speed = 0.0
        self._image_distance_sum = 0.0
        self._current_image_speed = 0.0
        self._spatial_speed_multiplier = 100.0

    @property
    def id(self):
        return self.id_

    def append(self, image_position: np.array, spatial_position: np.array):
        current_historty_length = len(self._spatial_position_history)
        if current_historty_length >= self._max_history_length - 1:
            # trunk-ignore(bandit/B101)
            assert current_historty_length == self._max_history_length - 1
            # dist = np.linalg.norm()
            removing_spatial_point = self._spatial_position_history[0]
            removing_image_point = self._image_position_history[0]
            self._spatial_position_history = self._spatial_position_history[1:]
            self._image_position_history = self._image_position_history[1:]
            old_spatial_distance = np.linalg.norm(
                removing_spatial_point - self._spatial_position_history[0]
            )
            self._spatial_distance_sum -= old_spatial_distance
            old_image_distance = np.linalg.norm(
                removing_image_point - self._image_position_history[0]
            )
            self._image_distance_sum -= old_image_distance
        if len(self._image_position_history) > 0:
            new_spatial_distance = np.linalg.norm(
                spatial_position - self._spatial_position_history[-1]
            )
            self._spatial_distance_sum += new_spatial_distance
            new_image_distance = np.linalg.norm(
                image_position - self._image_position_history[-1]
            )
            self._image_distance_sum += new_image_distance
        self._spatial_position_history.append(spatial_position)
        self._image_position_history.append(image_position)
        if len(self._spatial_position_history) > 1:
            self._current_spatial_speed = (
                self._spatial_speed_multiplier
                * self._spatial_distance_sum
                / (len(self._spatial_position_history) - 1)
            )
            self._current_image_speed = self._image_distance_sum / (
                len(self._image_position_history) - 1
            )
        else:
            self._current_spatial_speed = 0.0
            self._current_image_speed = 0.0

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
        return np.array((x_center, y_center), dtype=np.float32)

    def __len__(self):
        length = len(self._image_position_history)
        # trunk-ignore(bandit/B101)
        assert length == len(self._spatial_position_history)
        return length

    @property
    def spatial_speed(self):
        return self._current_spatial_speed

    @property
    def image_speed(self):
        return self._current_image_speed


class Clustering:
    def __init__(self):
        pass


class HockeyMOM:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        scale_width: float = 0.1,
        scale_height: float = 0.05,
        max_history: int = 26,
        speed_history: int = 26,
    ):
        self._video_frame = VideoFrame(
            image_width=image_width,
            image_height=image_height,
            scale_width=scale_width,
            scale_height=scale_height,
        )
        self._clamp_box = self._video_frame.box()
        self._online_tlwhs_history = list()
        self._max_history = max_history
        self._speed_history = speed_history
        self._online_ids = set()
        self._id_to_speed_map = dict()
        self._id_to_tlwhs_history_map = dict()

        self._kmeans_objects = dict()
        self._cluster_counts = dict()
        self._cluster_label_ids = dict()
        self._largest_cluster_label = dict()

        self._epsilon = 0.00001

        #
        # The camera's box speed itself
        #
        self._current_camera_box_speed_x = 0
        self._current_camera_box_speed_y = 0

        self._current_camera_box_speed_reversed_x = False
        self._current_camera_box_speed_reversed_y = False

        # self._camera_box_max_speed_x = max(image_width / 100.0, 5)
        # self._camera_box_max_speed_y = max(image_height / 100.0, 5)
        self._camera_box_max_speed_x = max(image_width / 300.0, 12.0)
        self._camera_box_max_speed_y = max(image_height / 300.0, 12.0)
        print(
            f"Camera Max speeds: x={self._camera_box_max_speed_x}, y={self._camera_box_max_speed_y}"
        )

        # Max acceleration in pixels per frame
        # self._camera_box_max_accel_x = 6
        # self._camera_box_max_accel_y = 6
        self._camera_box_max_accel_x = 3
        self._camera_box_max_accel_y = 3
        self._camera_box_max_accel_x = max(self._camera_box_max_speed_x / 15.0, 5.0)
        self._camera_box_max_accel_y = max(self._camera_box_max_speed_y / 15.0, 5.0)
        print(
            f"Camera Max acceleration: dx={self._camera_box_max_accel_x}, dy={self._camera_box_max_accel_y}"
        )

        self._camera_box_max_width_change_per_frame = 3
        self._camera_box_max_height_change_per_frame = 3

        self._camera = create_camera(
            elevation_m=3,
            tilt_degrees=45,
            roll_degrees=5,  # <-- looks correct
            focal_length=12,
            sensor_size_mm=(73, 4.55),
            # image_size_px=(4096, 1024),
            image_size_px=(image_width, image_height),
        )

    def get_speed(self):
        return math.sqrt(
                self._current_camera_box_speed_x * self._current_camera_box_speed_x
                + self._current_camera_box_speed_y * self._current_camera_box_speed_y
            )

    def is_fast(self, speed: float = 7):
        # return abs(self._current_camera_box_speed_x) > speed or abs(self._current_camera_box_speed_y) > speed
        return self.get_speed() >= speed

    def control_speed(self, abs_max_x: float, abs_max_y: float):
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
        self._online_ids = np.array(online_ids)
        self._online_tlws = np.array(online_tlws)
        self._online_image_center_points = np.array(
            [TlwhHistory.center_point(twls) for twls in online_tlws]
        )
        # Result will be 3D with all Z axis as zeroes, so just trim that dim
        if len(self._online_image_center_points):
            self._online_spatial = self._camera.spaceFromImage(
                self._online_image_center_points
            )
        else:
            self._online_spatial = np.array([])

        # Add to history
        prev_dict = self._id_to_tlwhs_history_map
        self._id_to_tlwhs_history_map = dict()
        # self._id_to_speed_map = dict()
        # trunk-ignore(ruff/B905)
        for id, image_pos, spatial_pos in zip(
            self._online_ids, self._online_tlws, self._online_spatial
        ):
            hist = prev_dict.get(id, TlwhHistory(id=id, video_frame=self._video_frame))
            hist.append(image_position=image_pos, spatial_position=spatial_pos)
            self._id_to_tlwhs_history_map[id] = hist

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
                # self._kmeans_objects = dict()
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
            torch_tensors.append(torch.from_numpy(n).to(device))
        tt = torch.cat(torch_tensors, dim=0)
        tt = torch.reshape(tt, (len(torch_tensors), 2))
        labels = self._kmeans_objects[n_clusters].fit_predict(tt)
        self._cluster_counts[n_clusters] = [0 for i in range(n_clusters)]
        cluster_counts = self._cluster_counts[n_clusters]
        cluster_label_ids = self._cluster_label_ids[n_clusters]
        id_count = len(self._online_ids)
        # trunk-ignore(bandit/B101)
        assert id_count == labels.shape[0]
        for i in range(id_count):
            id = self._online_ids[i]
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
        largest_clister_set = self.get_largest_cluster_id_set(n_clusters)
        result_ids = []
        for id in ids:
            if id in largest_clister_set:
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

    @property
    def camera(self):
        return self._camera

    @property
    def clamp_box(self):
        return self._clamp_box

    def get_image_tracking(self, online_ids):
        results = []
        for id in online_ids:
            results.append(self._id_to_tlwhs_history_map[id].image_position_history)
        return results

    def get_fast_ids(self, min_fast_items: int = 1, max_fast_items: int = 6):
        return self._prune_slow_ids(
            id_to_pos_history=self._id_to_tlwhs_history_map,
            min_fast_items=min_fast_items,
            max_fast_items=min_fast_items,
        )

    def get_spatial_speed(self, id: int):
        return self._id_to_tlwhs_history_map[id].spatial_speed

    def get_image_speed(self, id: int):
        return self._id_to_tlwhs_history_map[id].image_speed

    def get_tlwh(self, id: int):
        position_history = self._id_to_tlwhs_history_map[id]
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
        speeds = [hist.spatial_speed for hist in id_to_pos_history.values()]
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

    @classmethod
    def _box_center(cls, box):
        return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

    @classmethod
    def _clamp(cls, box, clamp_box):
        box[0] = max(box[0], clamp_box[0])
        box[1] = max(box[1], clamp_box[1])
        box[2] = min(box[2], clamp_box[2])
        box[3] = min(box[3], clamp_box[3])
        return box

    def clamp(self, box):
        return self._clamp(box, self._video_frame.box())

    @staticmethod
    def _tlwh_to_tlbr(box):
        return np.array(box[0], box[1], box[0] + box[2], box[1] + box[3])

    @staticmethod
    def _tlbr_to_tlwh(box):
        return np.array(box[0], box[1], box[2] - box[0], box[3] - box[1])

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
        box = box1.copy()
        box[0] = min(box[0], box2[0])
        box[1] = min(box[1], box2[1])
        box[2] = max(box[2], box2[2])
        box[3] = max(box[3], box2[3])
        return box

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
        center = cls._box_center(box)
        return (
            f"[x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}] "
            + f"w={box[2] - box[0]}, h={box[3] - box[1]}, "
            + f"center=({center[0]}, {center[1]})"
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

    def get_current_bounding_box(self, ids=None):
        bounding_intbox = self._video_frame.box()

        bounding_intbox[0], bounding_intbox[2] = bounding_intbox[2], bounding_intbox[1]
        bounding_intbox[1], bounding_intbox[3] = bounding_intbox[3], bounding_intbox[1]

        if ids is None:
            ids = self._online_ids

        for id in ids:
            tlwh = self.get_tlwh(id)
            x1 = tlwh[0]
            y1 = tlwh[1]
            w = tlwh[2]
            h = tlwh[3]
            intbox = np.array((x1, y1, x1 + w + 0.5, y1 + h + 0.5), dtype=np.int32)
            bounding_intbox[0] = min(bounding_intbox[0], intbox[0])
            bounding_intbox[1] = min(bounding_intbox[1], intbox[1])
            bounding_intbox[2] = max(bounding_intbox[2], intbox[2])
            bounding_intbox[3] = max(bounding_intbox[3], intbox[3])
        return bounding_intbox

    def ratioed_expand(self, box):
        ew = self._video_frame.width / 10
        eh = self._video_frame.height / 10
        diff = np.array([-ew / 2, -eh / 2, ew / 2, eh / 2])

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
        if maximum ^ (nw >= x):
            return nw or 1, y
        return x, nh or 1

    def make_box_proper_aspect_ratio(
        self,
        frame_id: int,
        the_box: torch.Tensor,
        desired_aspect_ratio: float,
        max_in_aspec_ratio: bool,
        no_max_in_aspec_ratio_at_edges: bool,
        verbose: bool = False,
        extra_validation: bool = True,
    ):
        """
        Make the given box the specified aspect ratio about the box's center.
        The final box must be within the initial box.
        """

        center = self._box_center(the_box)
        w = width(the_box)
        h = height(the_box)

        if w < self._video_frame.width / 3:
            w = self._video_frame.width / 3
        if h < self._video_frame.height / 2:
            h = self._video_frame.height / 2

        assert w <= self._video_frame.width
        assert h <= self._video_frame.height

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
            # if new_h_1 > self._video_frame.height:
            #     new_w = new_w_2
            #     new_h = new_h_2
            # elif new_w_2 > self._video_frame.width:
            #     new_w = new_w_2
            #     new_h = new_h_2
            # # Box with the largest area wins
            # elif new_w_1 * new_h_1 > new_w_2 * new_h_2:
            #     new_w = new_w_1
            #     new_h = new_h_1
            # else:
            #     new_w = new_w_2
            #     new_h = new_h_2
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

        # Within some reasonable epsilon, clamp to allowed width and height
        # so as not to set off assertions for trivial mathematical
        # variance

        # def _epsilon_clamp_down(value, max_value):
        #     nonlocal self
        #     diff = value - max_value
        #     if diff > 0 and diff < self._epsilon:
        #         value = max_value

        # new_w = _epsilon_clamp_down(new_w, self._video_frame.width)
        # new_h = _epsilon_clamp_down(new_h, self._video_frame.height)

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
        center[0] = float(int(center[0]))
        center[1] = float(int(center[1]))

        new_box = np.array(
            (
                center[0] - (new_w / 2.0) + 0.5,
                center[1] - (new_h / 2.0) + 0.5,
                center[0] + (new_w / 2.0) - 0.5,
                center[1] + (new_h / 2.0) - 0.5,
            ),
            dtype=np.float32,
        )

        if verbose:
            print(f"frame_id={frame_id}, ar={aspect_ratio(new_box)}")
        assert np.isclose(aspect_ratio(new_box), desired_aspect_ratio)

        if extra_validation:
            # Damned numerics defy logic
            ww = width(new_box)
            hh = height(new_box)
            assert ww <= (self._video_frame.width + self._epsilon)
            assert hh <= (self._video_frame.height + self._epsilon)

        # assert np.isclose(aspect_ratio(new_box), desired_aspect_ratio)

        return new_box

    def shift_box_to_edge(self, box):
        """
        If a box is off the edge of the image, translate
        the box to be flush with the edge instead.
        """
        assert width(box) <= self._video_frame.width
        assert height(box) <= self._video_frame.height
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

    def _apply_box_velocity(self, box, scale_speed: float):
        box[0] += self._current_camera_box_speed_x * scale_speed
        box[2] += self._current_camera_box_speed_x * scale_speed
        box[1] += self._current_camera_box_speed_y * scale_speed
        box[3] += self._current_camera_box_speed_y * scale_speed
        return box

    def get_next_temporal_box(
        self,
        proposed_new_box: List[int],
        last_box: List[int],
        pre_clamp_to_video_frame: bool = True,
        scale_speed: float = 1.0,
        verbose: bool = False,
    ):
        if verbose:
            print(f"\nproposed_new_box: {self.box_str(proposed_new_box)}")
            if last_box is not None:
                print(f"last_box: {self.box_str(last_box)}")

        next_box = proposed_new_box.copy()

        if pre_clamp_to_video_frame:
            video_frame = self._video_frame.box()
            self._clamp(next_box, video_frame)

        if last_box is None:
            return next_box

        # First, we apply the center's proposed change
        proposed_center = self._box_center(proposed_new_box)
        last_center = self._box_center(last_box)
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

        if verbose:
            print(
                f"attempted accel x : {acceleration_dx}, allowed: {allowed_acceleration_dx}"
            )
            print(
                f"attempted accel y : {acceleration_dy}, allowed: {allowed_acceleration_dy}"
            )

        #
        # now adjust velocity based on the allowed accelerations
        #
        # self._current_camera_box_speed_reversed_x = False
        # self._current_camera_box_speed_reversed_y = False

        old_s_x = self._current_camera_box_speed_x
        old_s_y = self._current_camera_box_speed_y

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
                proposed_width_change, -self._camera_box_max_width_change_per_frame
            )
        else:
            allowed_width_change = min(
                proposed_width_change, self._camera_box_max_height_change_per_frame
            )

        if proposed_height_change < 0:
            allowed_height_change = max(
                proposed_height_change, -self._camera_box_max_width_change_per_frame
            )
        else:
            allowed_height_change = min(
                proposed_height_change, self._camera_box_max_width_change_per_frame
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

        moved_box = self._apply_box_velocity(new_last_box, scale_speed=scale_speed)

        self._curtail_speeed_at_edges(moved_box)

        moved_box = self.shift_box_to_edge(moved_box)

        if verbose:
            print(f"moved_box: {self.box_str(moved_box)}")
        return moved_box

    def changed_direction(self, reset:bool = False):
        return self._current_camera_box_speed_reversed_x or self._current_camera_box_speed_reversed_x

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

    def _curtail_speeed_at_edges(self, box):
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
