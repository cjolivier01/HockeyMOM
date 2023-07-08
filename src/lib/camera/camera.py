import cameratransform as ct
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans


def create_camera(
    elevation_m: float,
    image_size_px=(3264, 2448),
    tilt_degrees: float = 45.0,
    roll_degrees: float = 0.0,
    focal_length: float = 6.2,
    sensor_size_mm: Tuple[float, float] = (6.17, 4.55),
) -> ct.Camera:
    """
    Create a camera object witht he given parameters/position
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
        return np.array((0, 0, self._image_width, self._image_height), dtype=np.int32)

    @property
    def width(self):
        return self._image_width

    @property
    def height(self):
        return self._image_height

    # def point_ratio_vertical_away(self, point):
    #     # TODO: Get rid of this
    #     """
    #     Vertical is farther away towards the top
    #     """
    #     # Is the first one X?
    #     y = point[0]
    #     dy = self._vertical_center - y
    #     return dy / self._vertical_center * self._scale_height

    # def point_ratio_horizontal_away(self, point):
    #     # TODO: Get rid of this
    #     """
    #     Horizontal is farther away towards the left anf right sides
    #     """
    #     # Is the first one X?
    #     x = point[1]
    #     dx = abs(x - self._horizontal_center)
    #     # TODO: Probably can work this out with trig, the actual linear distance,
    #     # # esp since we know the rink's size
    #     # Just do a meatball calculation for now...
    #     return dx / self._horizontal_center * self._scale_width


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
        assert length == len(self._spatial_position_history)
        return length

    @property
    def spatial_speed(self):
        return self._current_spatial_speed

    @property
    def image_speed(self):
        return self._current_image_speed


class HockeyMOM:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        scale_width: float = 0.1,
        scale_height: float = 0.05,
        max_history: int = 26,
        speed_history: int = 26,
        spatial_speed_multiplier: float = 1.0,
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
        # The camera's box speed itself
        self._camera_box_speed_x = 0
        self._camera_box_speed_y = 0
        self._camera_aspect_ratio = 16.0 / 9
        self._box_width_extra = image_width / 8
        self._box_height_extra = image_height / 8

        self._camera = create_camera(
            elevation_m=3,
            tilt_degrees=45,
            # tilt_degrees=65,
            # tilt_degrees=var,
            roll_degrees=5,  # <-- looks correct
            focal_length=12,
            # focal_length=var,
            sensor_size_mm=(73, 4.55),
            # sensor_size_mm=(73, var),
            # sensor_size_mm=(var, 4.55),
            image_size_px=(4096, 1024),
            # image_size_px=(image_width, image_height),
        )

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
        for id, image_pos, spatial_pos in zip(
            self._online_ids, self._online_tlws, self._online_spatial
        ):
            hist = prev_dict.get(id, TlwhHistory(id=id, video_frame=self._video_frame))
            hist.append(image_position=image_pos, spatial_position=spatial_pos)
            self._id_to_tlwhs_history_map[id] = hist

    def calculate_clusters(self, n_clusters):
        self._n_clusters = n_clusters
        self._cluster_label_ids = dict()
        self._largest_cluster_label = None
        if len(self._online_image_center_points) < n_clusters:
            # Minimum clusters that we can have is the minimum number of objects
            n_clusters = len(self._online_image_center_points)
        if not n_clusters:
            self._kmeans = None
            return
        else:
            self._kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            self._kmeans.fit(self._online_image_center_points)
        self._cluster_counts = [0 for i in range(n_clusters)]
        assert len(self._online_ids) == len(self._kmeans.labels_)
        for id, cluster_label in zip(self._online_ids, self._kmeans.labels_):
            self._cluster_counts[cluster_label] += 1
            if cluster_label not in self._cluster_label_ids:
                self._cluster_label_ids[cluster_label] = [id]
            else:
                self._cluster_label_ids[cluster_label].append(id)
        for cluster_label, cluster_id_list in self._cluster_label_ids.items():
            if self._largest_cluster_label is None:
                self._largest_cluster_label = cluster_label
            elif len(cluster_id_list) > len(
                self._cluster_label_ids[self._largest_cluster_label]
            ):
                self._largest_cluster_label = cluster_label
        # if self._largest_cluster_label is not None:
        #     print(
        #         f"Largest cluster: {self._cluster_label_ids[self._largest_cluster_label]}"
        #     )

    def get_largest_cluster_id_set(self):
        if self._largest_cluster_label is None:
            return set()
        return set(self._cluster_label_ids[self._largest_cluster_label])

    def prune_not_in_largest_cluster(self, ids):
        largest_clister_set = self.get_largest_cluster_id_set()
        result_ids = []
        for id in ids:
            if id in largest_clister_set:
                result_ids.append(id)
        return result_ids

    @property
    def largest_cluster_label(self):
        return self.self._largest_cluster_label

    @property
    def cluster_size(self, cluster_label):
        return len(self._cluster_label_ids[cluster_label])

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

    def get_fast_ids(self, min_fast_items: int = 2, max_fast_items: int = 6):
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
        fast_threshhold: float = 2,
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
        for sorted_id, sorted_speed in zip(sorted_ids, sorted_speeds):
            if sorted_speed >= fast_threshhold:
                if len(id_to_pos_history[sorted_id]) >= min_history_length:
                    fast_ids.append(sorted_id)
                    if max_fast_items > 0 and len(fast_ids) >= max_fast_items:
                        break

        if min_fast_items > 0:
            if len(fast_ids) < 4:
                fast_ids = []

        # Now prune to
        return fast_ids

    @staticmethod
    def _box_center(intbox):
        return [(intbox[0] + intbox[2]) / 2, (intbox[1] + intbox[3]) / 2]

    @staticmethod
    def _clamp(box, clamp_box):
        box[0] = max(box[0], clamp_box[0])
        box[1] = max(box[1], clamp_box[1])
        box[2] = min(box[2], clamp_box[2])
        box[3] = min(box[3], clamp_box[3])
        return box

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

    def make_normalized_bounding_box(self, bounding_box):
        bounding_box = copy.deepcopy(bounding_box)
        # im_w = self._video_frame.width
        # im_h = self._video_frame.height

        bounding_box[0] -= self._box_height_extra / 2
        bounding_box[1] -= self._box_width_extra / 2
        bounding_box[2] += self._box_width_extra / 2
        bounding_box[3] += self._box_width_extra / 2

        # Box may be overflowing the image, so adjust
        # if bounding_box[0] < 0:
        #     bounding_box[2] += -bounding_box[0]
        #     bounding_box[0] = 0
        #     if bounding_box[2] >= im_w:
        #         bounding_box[2] = im_w - 1

        # if bounding_box[2] > im_w - 1:
        #     bounding_box[0] -= bounding_box[2] - im_w - 1
        #     bounding_box[2] = im_w - 1

        # if bounding_box[1] < 0:
        #     bounding_box[3] += -bounding_box[1]
        #     bounding_box[1] = 0
        #     if bounding_box[3] >= im_h:
        #         bounding_box[3] = im_h - 1

        # if bounding_box[3] > im_h - 1:
        #     bounding_box[1] -= bounding_box[3] - im_h - 1
        #     bounding_box[3] = im_h - 1

        # Expand entire box a little
        # bounding_box[0] -= im_w / 10
        # bounding_box[1] -= im_h / 10
        # bounding_box[2] += im_w / 10
        # bounding_box[3] += im_h / 10

        # # Finally, clamp to image dimensions
        # bounding_box[0] = int(max(bounding_box[0], 0))
        # bounding_box[1] = int(max(bounding_box[1], 0))
        # bounding_box[2] = int(min(bounding_box[2], im_w - 1))
        # bounding_box[3] = int(min(bounding_box[3], im_h - 1))

        clamped_box = HockeyMOM._clamp(
            np.asarray(bounding_box, dtype=np.int32), self._video_frame.box()
        )
        return clamped_box

    @staticmethod
    def translate_box(intbox, new_box, clamp_box):

        max_x=clamp_box[2]/100
        max_y=clamp_box[3]/100

        old_center = HockeyMOM._box_center(intbox)
        new_center = HockeyMOM._box_center(new_box)
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        if dx > 0:
            dx = min(dx, max_x)
        else:
            dx = max(dx, -max_x)
        if dy > 0:
            dy = min(dy, max_y)
        else:
            dy = max(dy, -max_y)
        print(f"dx={dx}, dy={dy}")
        box_center = old_center
        box_center[0] += dx
        box_center[1] += dy
        old_w = intbox[2] - intbox[0]
        old_h = intbox[3] - intbox[1]
        new_w = new_box[2] - new_box[0]
        new_h = new_box[3] - new_box[1]
        dw = new_w - old_w
        dh = new_h - old_h
        if dw > 0:
            dw = min(dw, max_x)
        else:
            dw = max(dw, -max_x)
        if dh > 0:
            dh = min(dh, max_y)
        else:
            dh = max(dh, -max_y)

        # Translate
        intbox[0] += dx
        intbox[1] += dy
        intbox[2] += dx
        intbox[3] += dy

        # Resize
        intbox[0] -= dw / 2
        intbox[1] -= dh / 2
        intbox[2] += dw / 2
        intbox[3] += dh / 2
        clamped_box = HockeyMOM._clamp(np.asarray(intbox, dtype="int"), clamp_box)
        return clamped_box


if __name__ == "__main__":
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
