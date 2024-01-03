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
import multiprocessing
import queue

from pathlib import Path

import torch
import torchvision as tv

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.utils.utils import create_queue
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.camera.moving_box import MovingBox
from hmlib.video_out import ImageProcData, VideoOutput
from hmlib.camera.clusters import ClusterMan
from hmlib.utils.image import ImageHorizontalGaussianDistribution
from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.config import get_rink_config, get_game_config, get_camera_config

from hmlib.utils.box_functions import (
    width,
    height,
    center,
    center_batch,
    center_x_distance,
    center_distance,
    aspect_ratio,
    make_box_at_center,
    remove_largest_bbox,
)

from hmlib.utils.box_functions import tlwh_centers
from hockeymom import core

core.hello_world()


##
#  _____         __             _ _                                                     _
# |  __ \       / _|           | | |      /\                                           | |
# | |  | | ___ | |_  __ _ _   _| | |_    /  \    _ __  __ _ _   _ _ __ ___   ___  _ __ | |_  ___
# | |  | |/ _ \|  _|/ _` | | | | | __|  / /\ \  | '__|/ _` | | | | '_ ` _ \ / _ \| '_ \| __|/ __|
# | |__| |  __/| | | (_| | |_| | | |_  / ____ \ | |  | (_| | |_| | | | | | |  __/| | | | |_ \__ \
# |_____/ \___||_|  \__,_|\__,_|_|\__|/_/    \_\|_|   \__, |\__,_|_| |_| |_|\___||_| |_|\__||___/
#                                                      __/ |
#                                                     |___/
#
# Some experimental and debugging parameters that aid in development
#

# _PERSON_CLASS_ID = 1
# _PLAYER_CLASS_ID = 2
# # _REFEREE_CLASS_ID = 3

# category_list = [
#     {_PERSON_CLASS_ID: "person"},
#     {_PLAYER_CLASS_ID: "player"},
#     # {_REFEREE_CLASS_ID: "referee"},
# ]

#
# TODO: Moving to yaml files
#
RINK_CONFIG = {
    "vallco": {
        "fixed_edge_scaling_factor": 0.8,
        "fixed_edge_rotation_angle": 40.0,
        "boundaries": {
            "stockton2": {
                "upper": [],
                "lower": [],
            },
            "sharksbb1": {
                "upper": [
                    [77, 899, 982, 550],
                ],
                "lower": [
                    [30, 721, 358, 1420],
                    [3287, 1468, 3645, 720],
                ],
            },
        },
    },
    "yerba_buena": {
        "fixed_edge_scaling_factor": 1.5,
        "fixed_edge_rotation_angle": 25.0,
        "boundaries": {
            "upper": [],
            "lower": [],
        },
    },
    "dublin": {
        # "fixed_edge_scaling_factor": 1.5,
        # "fixed_edge_rotation_angle": 30.0,
        "boundaries": {
            "lbd3": {
                "upper": [
                    [254, 840, 1071, 598],
                    [1101, 568, 2688, 588],
                ],
                "lower": [
                    [7, 854, 375, 1082],
                    [264, 1007, 1188, 1539],
                    [1229, 1524, 2398, 1572],
                    [2362, 1517, 3245, 1181],
                    [3240, 1199, 3536, 1000],
                    [3370, 1095, 3808, 819],
                ],
            },
            "tvbb2": {
                "upper": [
                    [214, 753, 1265, 492],
                    [1179, 495, 3275, 653],
                ],
                "lower": [
                    [6, 737, 413, 894],
                    [2340, 1552, 3200, 1394],
                    [3142, 1405, 3631, 1151],
                    [423, 891, 1273, 1387],
                    [1279, 1394, 2235, 1564],
                    [3441, 1260, 3803, 1050],
                    [3803, 1050, 4134, 712],
                ],
            },
        },
    },
    "sharks_orange": {
        # "fixed_edge_scaling_factor": 0.8,
        # "fixed_edge_scaling_factor": 1.25,
        # "fixed_edge_rotation_angle": 25.0,
        "boundaries": {
            "sharksbb2-1": {
                "upper": [
                    [2001, 123, 2629, 208],
                    [1283, 117, 1979, 162],
                ],
                "lower": [
                    [21, 498, 1664, 878],
                    [1662, 856, 3034, 799],
                    [3044, 800, 3762, 673],
                ],
            },
            "sharksbb1-2": {
                "upper": [
                    [1300, 528, 1848, 467],
                    [1850, 469, 2478, 455],
                    [2535, 450, 3050, 500],
                    [2951, 480, 3735, 725],
                    [3671, 700, 3757, 809],
                    [324, 889, 919, 598],
                    # false positives on upper boards
                    [1897, 497, 1924, 495],
                    [1467, 535, 1518, 534],
                    [1543, 530, 1581, 528],
                    [2185, 486, 2214, 486],
                    [1598, 525, 1632, 525],
                ],
                "lower": [
                    [2862, 949, 3831, 733],
                    [1444, 1102, 2219, 1047],
                    [631, 998, 1441, 1126],
                    [12, 577, 342, 900],
                    [327, 909, 744, 1022],
                ],
            },
            "sharksbb1-2.1": {
                "upper": [
                    [2047, 758, 2312, 719],
                    [2312, 718, 2653, 686],
                    [2653, 686, 2971, 666],
                    [2971, 666, 3363, 657],
                    [3365, 649, 4001, 674],
                    [4071, 689, 4419, 750],
                    [4423, 756, 5374, 1063],
                    [5292, 1049, 5342, 1201],
                    [396, 1294, 1259, 893],
                    # false positives on upper boards
                    [2202, 755, 2247, 752],
                    [2291, 756, 2333, 756],
                    [2006, 785, 2042, 781],
                    [3381, 690, 3418, 691],
                    [3462, 685, 3502, 689],
                    [3120, 697, 3177, 697],
                ],
                "lower": [
                    [5319, 975, 5649, 959],
                    [15, 1188, 470, 1278],
                    [467, 1299, 1040, 1474],
                    [1043, 1466, 2546, 1628],
                    [2546, 1628, 4837, 1243],
                    [4837, 1243, 5307, 1096],
                ],
            },
        },
    },
    # "sharks_black": {
    #     "fixed_edge_scaling_factor": 1.25,
    #     "fixed_edge_rotation_angle": 25.0,
    #     "boundaries": {
    #         "onehockey-sharksbb2": {
    #             "upper": [
    #                 [3260, 1027, 4129, 1045],
    #                 [4107, 1035, 5233, 1159],
    #                 [7541, 1679, 8408, 1788],
    #             ],
    #             "lower": [
    #                 [1232, 1605, 1855, 1761],
    #                 [1805, 1770, 4063, 2127],
    #                 [4006, 2139, 6228, 2093],
    #                 [6168, 2115, 7543, 1818],
    #                 [7537, 1723, 8388, 1796],
    #             ],
    #         },
    #     },
    # },
}

BASIC_DEBUGGING = False


class DefaultArguments(core.HMPostprocessConfig):
    def __init__(
        self,
        game_id: str,
        rink: str,
        basic_debugging: bool = BASIC_DEBUGGING,
        show_image: bool = False,
        cam_ignore_largest: bool = False,
        # args: argparse.Namespace = None,
        camera: str = "gopro",
    ):
        # basic_debugging = False

        super().__init__()

        self.camera_config = get_camera_config(camera=camera, root_dir=os.getcwd())
        self.rink_config = get_rink_config(rink=rink, root_dir=os.getcwd())
        self.game_config = get_game_config(game_id=game_id, root_dir=os.getcwd())

        # Display the image every frame (slow)
        self.show_image = show_image or basic_debugging
        # self.show_image = True

        # Draw individual player boxes, tracking ids, speed and history trails
        self.plot_individual_player_tracking = False or basic_debugging
        # self.plot_individual_player_tracking = True

        # Draw all detection boxes (even if not tracking the detection)
        self.plot_all_detections = False
        # self.plot_all_detections = True

        # Draw intermediate boxes which are used to compute the final camera box
        self.plot_cluster_tracking = False or basic_debugging
        # self.plot_cluster_tracking = True

        # Use a differenmt algorithm when fitting to the proper aspect ratio,
        # such that the box calculated is much larger and often takes
        # the entire height.  The drawback is there's not much zooming.
        self.max_in_aspec_ratio = True
        # self.max_in_aspec_ratio = False

        # Zooming is fixed based upon the horizonal position's distance from center
        # self.apply_fixed_edge_scaling = False
        self.apply_fixed_edge_scaling = True

        # self.fixed_edge_scaling_factor = RINK_CONFIG[rink]["fixed_edge_scaling_factor"]
        self.fixed_edge_scaling_factor = self.rink_config["rink"]["camera"][
            "fixed_edge_scaling_factor"
        ]

        self.plot_camera_tracking = False or basic_debugging
        # self.plot_camera_tracking = True

        self.plot_moving_boxes = False or basic_debugging
        # self.plot_moving_boxes = True

        # Print each frame number in the upper left corner
        self.plot_frame_number = False or basic_debugging
        # self.plot_frame_number = True

        self.plot_boundaries = (
            False or basic_debugging or self.plot_individual_player_tracking
        )
        # self.plot_boundaries = True

        # Plot frame ID and speed/velocity in upper-left corner
        self.plot_speed = False

        # self.fixed_edge_rotation = False
        self.fixed_edge_rotation = True

        self.fixed_edge_rotation_angle = self.rink_config["rink"]["camera"][
            "fixed_edge_rotation_angle"
        ]

        # Plot the component shapes directly related to camera stickiness
        self.plot_sticky_camera = False or basic_debugging
        # self.plot_sticky_camera = True

        # Skip some number of frames before post-processing. Useful for debugging a
        # particular section of video and being able to reach
        # that portiuon of the video more quickly
        self.skip_frame_count = 0

        # Moving right-to-left
        # self.skip_frame_count = 450

        # Stop at the given frame and (presumably) output the final video.
        # Useful for debugging a
        # particular section of video and being able to reach
        # that portiuon of the video more quickly
        self.stop_at_frame = 0
        # self.stop_at_frame = 30*30

        self.cam_ignore_largest = cam_ignore_largest

        # Make the image the same relative dimensions as the initial image,
        # such that the highest possible resolution is available when the camera
        # box is either the same height or width as the original video image
        # (Slower, but better final quality)
        self.scale_to_original_image = True
        # self.scale_to_original_image = False

        # Crop the final image to the camera window (possibly zoomed)
        self.crop_output_image = True and not basic_debugging
        # self.crop_output_image = False

        # Use cuda for final image resizing (if possible)
        self.use_cuda = False

        # Draw watermark on the image
        self.use_watermark = True
        # self.use_watermark = False

        self.detection_inclusion_box = None

        #
        # SHARKS ORANGE RINK
        #
        # self.top_border_lines = RINK_CONFIG[rink]["boundaries"][game_id]["upper"]
        # self.bottom_border_lines = RINK_CONFIG[rink]["boundaries"][game_id]["lower"]
        self.top_border_lines = self.game_config["game"]["boundaries"]["upper"]
        self.bottom_border_lines = self.game_config["game"]["boundaries"]["lower"]


def scale_box(box, from_img, to_img):
    from_sz = (from_img.shape[1], from_img.shape[0])
    to_sz = (to_img.shape[1], to_img.shape[0])
    w_scale = to_sz[1] / from_sz[1]
    h_scale = to_sz[0] / from_sz[0]
    new_box = [box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale]
    print(f"from={box} -> to={new_box}")
    return new_box


def prune_by_inclusion_box(online_tlwhs, online_ids, inclusion_box, boundaries):
    if len(online_tlwhs) == 0:
        # online_ids should also be empty
        assert len(online_ids) == 0
        # nothing
        return online_tlwhs, online_ids
    if inclusion_box is None and boundaries is None:
        return online_tlwhs, online_ids
    filtered_online_tlwh = []
    filtered_online_ids = []
    online_tlwhs_centers = tlwh_centers(tlwhs=online_tlwhs)
    for i in range(len(online_tlwhs_centers)):
        center = online_tlwhs_centers[i]
        if inclusion_box is not None:
            if inclusion_box[0] and center[0] < inclusion_box[0]:
                continue
            elif inclusion_box[2] and center[0] > inclusion_box[2]:
                continue
            elif inclusion_box[1] and center[1] < inclusion_box[1]:
                continue
            elif inclusion_box[3] and center[1] > inclusion_box[3]:
                continue
        if boundaries is not None:
            # TODO: boundaries could be done with the box edges
            if boundaries.is_point_outside(center):
                # print(f"ignoring: {center}")
                continue
        filtered_online_tlwh.append(online_tlwhs[i])
        filtered_online_ids.append(online_ids[i])
    if len(filtered_online_tlwh) == 0:
        assert len(filtered_online_ids) == 0
        return [], []
    return torch.stack(filtered_online_tlwh), torch.stack(filtered_online_ids)


class Detection:
    def __init__(self, track_id, tlwh, history):
        self.track_id = track_id
        self.tlwh = tlwh
        self.history = history


class CamTrackPostProcessor:
    def __init__(
        self,
        hockey_mom,
        start_frame_id,
        data_type,
        fps: float,
        save_dir,
        device,
        opt,
        original_clip_box,
        args: argparse.Namespace,
        save_frame_dir: str = None,
        async_post_processing: bool = False,
        use_fork: bool = False,
        video_out_device: str = None,
    ):
        self._args = args
        self._start_frame_id = start_frame_id
        self._hockey_mom = hockey_mom
        self._queue = create_queue(mp=use_fork)
        self._data_type = data_type
        self._fps = fps
        self._opt = opt
        self._thread = None
        self._use_fork = use_fork
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float32)
        self._output_video = None
        self._final_image_processing_started = False
        self._async_post_processing = async_post_processing
        self._device = device
        self._horizontal_image_gaussian_distribution = None
        self._boundaries = None
        self._timer = Timer()
        self._cluster_man = None
        self._video_out_device = video_out_device

        if self._video_out_device is None:
            self._video_out_device = self._device

        self._save_dir = save_dir
        self._save_frame_dir = save_frame_dir

        self._outside_box_expansion_for_speed_curtailing = torch.tensor(
            [-100.0, -100.0, 100.0, 100.0],
            dtype=torch.float32,
            device=self._device,
        )

        if self._args.top_border_lines or self._args.bottom_border_lines:
            self._boundaries = BoundaryLines(
                self._args.top_border_lines,
                self._args.bottom_border_lines,
                original_clip_box,
            )

        # Persistent state across frames
        self._previous_cluster_union_box = None
        self._last_temporal_box = None
        self._last_sticky_temporal_box = None
        self._last_dx_shrink_size = 0
        self._center_dx_shift = 0
        self._frame_counter = 0

        self._current_roi = None
        self._current_roi_aspect = None
        self._video_output_campp = None
        self._video_output_boxtrack = None

    def get_first_frame_id(self):
        return self._args.skip_frame_count

    def start(self):
        if self._use_fork:
            self._child_pid = os.fork()
            if not self._child_pid:
                self._start()
        else:
            self._thread = Thread(target=self._start, name="CamPostProc")
            self._thread.start()

    def _start(self):
        return self.postprocess_frame_worker()

    def stop(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None

        elif self.use_fork:
            self._queue.put(None)
            if self._child_pid:
                os.waitpid(self._child_pid)
        self._video_output_campp.stop()

    def send(
        self, online_tlwhs, online_ids, detections, info_imgs, image, original_img
    ):
        while self._queue.qsize() > 10:
            #print("Cam post-process queue too large")
            time.sleep(0.001)
        try:
            dets = []
            # dets = [
            #     Detection(track_id=d.track_id, tlwh=d.tlwh, history=d.history)
            #     for d in detections
            # ]
            if self._async_post_processing:
                self._queue.put(
                    (
                        online_tlwhs,
                        online_ids,
                        dets,
                        info_imgs,
                        image,
                        original_img,
                    )
                )
            else:
                online_targets_and_img = (
                    online_tlwhs,
                    online_ids,
                    dets,
                    info_imgs,
                    image,
                    original_img,
                )
                self.cam_postprocess(online_targets_and_img=online_targets_and_img)
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise

    def postprocess_frame_worker(self):
        try:
            self._postprocess_frame_worker()
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise

    def prepare_online_image(self, online_im) -> np.array:
        if isinstance(online_im, torch.Tensor):
            online_im = online_im.numpy()
        if not online_im.flags["C_CONTIGUOUS"]:
            online_im = np.ascontiguousarray(online_im)
        return online_im

    def _get_gaussian(self, image_width: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = (
                ImageHorizontalGaussianDistribution(image_width)
            )
        return self._horizontal_image_gaussian_distribution

    def _postprocess_frame_worker(self):
        self._center_dx_shift = 0
        timer = Timer()

        if self._args.crop_output_image:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = (
                self._hockey_mom.video.height * self._final_aspect_ratio
            )
        else:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = self._hockey_mom.video.width

        assert self._video_output_campp is None
        self._video_output_campp = VideoOutput(
            args=self._args,
            output_video_path=os.path.join(self._save_dir, "tracking_output.avi")
            if self._save_dir is not None
            else None,
            fps=self._fps,
            use_fork=False,
            start=False,
            output_frame_width=self.final_frame_width,
            output_frame_height=self.final_frame_height,
            save_frame_dir=self._save_frame_dir,
            watermark_image_path=os.path.realpath(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "..",
                    "..",
                    "images",
                    "sports_ai_watermark.png",
                )
            )
            if self._args.use_watermark
            else None,
            device=self._video_out_device,
        )
        self._video_output_campp.start()

        if self._args.crop_output_image and self._save_dir is not None:
            self._video_output_boxtrack = VideoOutput(
                args=self._args,
                output_video_path=os.path.join(self._save_dir, "boxtrack_output.avi"),
                fps=self._fps,
                use_fork=False,
                start=False,
                output_frame_width=self.final_frame_width,
                output_frame_height=self.final_frame_height,
                watermark_image_path=os.path.realpath(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "..",
                        "..",
                        "..",
                        "images",
                        "sports_ai_watermark.png",
                    )
                )
                if self._args.use_watermark
                else None,
            )
            self._video_output_boxtrack.start()

        while True:
            online_targets_and_img = self._queue.get()
            if online_targets_and_img is None:
                break
            self.cam_postprocess(online_targets_and_img=online_targets_and_img)

    _INFO_IMGS_FRAME_ID_INDEX = 2

    def get_arena_box(self):
        return self._hockey_mom._video_frame.bounding_box()

    def _kmeans_cuda_device(self):
        if self._use_fork:
            return "cpu"
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        # return self._device

    def cam_postprocess(self, online_targets_and_img):
        self._timer.tic()
        max_dx_shrink_size = 100  # ???

        # if frame_id % 20 == 0:
        #     logger.info(
        #         "Post-Processing frame {} ({:.2f} fps)".format(
        #             frame_id, 1.0 / max(1e-5, timer.average_time)
        #         )
        #     )
        #     timer = Timer()
        # timer.tic()
        online_tlwhs = online_targets_and_img[0]
        online_ids = online_targets_and_img[1]
        detections = online_targets_and_img[2]
        info_imgs = online_targets_and_img[3]

        frame_ids = info_imgs[self._INFO_IMGS_FRAME_ID_INDEX]
        frame_id = frame_ids[self._frame_counter % len(frame_ids)]
        self._frame_counter += 1

        # print(online_ids)
        # print(online_tlwhs)

        if self._args.cam_ignore_largest and len(online_tlwhs):
            # Don't remove unless we have at least 4 online items being tracked
            online_tlwhs, mask = remove_largest_bbox(online_tlwhs, min_boxes=4)
            online_ids = online_ids[mask]

        # Exclude detections outside of an optional bounding box
        online_tlwhs, online_ids = prune_by_inclusion_box(
            online_tlwhs,
            online_ids,
            self._args.detection_inclusion_box,
            boundaries=self._boundaries,
        )

        # info_imgs = online_targets_and_img[3]
        img0 = online_targets_and_img[4]
        original_img = online_targets_and_img[5]

        self._hockey_mom.append_online_objects(online_ids, online_tlwhs)

        if self._cluster_man is None:
            self._cluster_man = ClusterMan(
                sizes=[3, 2], device=self._kmeans_cuda_device()
            )

        self._cluster_man.calculate_all_clusters(
            center_points=center_batch(online_tlwhs), ids=online_ids
        )

        if self._args.show_image or self._save_dir is not None:
            assert self._args.scale_to_original_image
            # del original_img
            # if self._args.scale_to_original_image:
            #     if isinstance(original_img, torch.Tensor):
            #         original_img = original_img.numpy()
            #     online_im = original_img
            #     del original_img
            # else:
            #     online_im = img0
            # online_im = self.prepare_online_image(online_im)
            online_im = original_img

            if self._args.plot_boundaries and self._boundaries is not None:
                self._boundaries.draw(online_im)

            if self._args.plot_all_detections:
                online_id_set = set(online_ids)
                offline_ids = []
                offline_tlwhs = []
                skipped_count = 0
                for det in detections:
                    if det.track_id not in online_id_set:
                        tlwh = det.tlwh
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > self._opt.min_box_area and not vertical:
                            offline_ids.append(det.track_id)
                            offline_tlwhs.append(det.tlwh)
                        else:
                            skipped_count += 1
                if skipped_count:
                    print(f"Skipped {skipped_count} detections")

                if offline_ids:
                    online_im = vis.plot_tracking(
                        online_im,
                        offline_tlwhs,
                        offline_ids,
                        frame_id=frame_id,
                        speeds=[],
                        line_thickness=1,
                        box_color=(255, 128, 255),
                        ignore_frame_id=True,
                        print_track_id=False,
                    )

            if self._args.plot_individual_player_tracking:
                online_im = vis.plot_tracking(
                    online_im,
                    online_tlwhs,
                    online_ids,
                    frame_id=frame_id,
                    speeds=[],
                    line_thickness=2,
                )

            # Examine as 2 clusters
            largest_cluster_ids_2 = self._cluster_man.prune_not_in_largest_cluster(
                num_clusters=2, ids=online_ids
            )
            if len(largest_cluster_ids_2):
                largest_cluster_ids_box2 = self._hockey_mom.get_current_bounding_box(
                    largest_cluster_ids_2
                )
                if self._args.plot_cluster_tracking:
                    vis.plot_rectangle(
                        online_im,
                        largest_cluster_ids_box2,
                        color=(128, 0, 0),  # dark red
                        thickness=6,
                        label="cluster_box2",
                    )
            else:
                largest_cluster_ids_box2 = None

            # Examine as 3 clusters
            largest_cluster_ids_3 = self._cluster_man.prune_not_in_largest_cluster(
                num_clusters=3, ids=online_ids
            )
            if len(largest_cluster_ids_3):
                largest_cluster_ids_box3 = self._hockey_mom.get_current_bounding_box(
                    largest_cluster_ids_3
                )
                if self._args.plot_cluster_tracking:
                    vis.plot_rectangle(
                        online_im,
                        largest_cluster_ids_box3,
                        color=(0, 0, 128),  # dark blue
                        thickness=6,
                        label="cluster_box3",
                    )
            else:
                largest_cluster_ids_box3 = None

            if (
                largest_cluster_ids_box2 is not None
                and largest_cluster_ids_box3 is not None
            ):
                current_box = self._hockey_mom.union_box(
                    largest_cluster_ids_box2, largest_cluster_ids_box3
                )
            elif largest_cluster_ids_box2 is not None:
                current_box = largest_cluster_ids_box2
            elif largest_cluster_ids_box3 is not None:
                current_box = largest_cluster_ids_box3
            elif self._previous_cluster_union_box is not None:
                current_box = self._previous_cluster_union_box.clone()
            else:
                current_box = self._hockey_mom._video_frame.bounding_box()

            if current_box is None:
                current_box = self._hockey_mom._video_frame.bounding_box()

            self._previous_cluster_union_box = current_box.clone()

            # Some players may be off-screen, so their box may go over an edge
            current_box = self._hockey_mom.clamp(current_box)

            if self._args.plot_cluster_tracking:
                # The union of the two cluster boxes
                online_im = vis.plot_alpha_rectangle(
                    online_im,
                    current_box,
                    color=(64, 64, 64),  # dark gray
                    label="union_clusters",
                    opacity_percent=25,
                )

            #
            # Current ROI box
            #
            if True:
                if self._current_roi is None:
                    start_box = current_box
                    self._current_roi = MovingBox(
                        label="Current ROI",
                        bbox=start_box,
                        arena_box=self.get_arena_box(),
                        max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1.5,
                        max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1.5,
                        max_accel_x=self._hockey_mom._camera_box_max_accel_x * 1.1,
                        max_accel_y=self._hockey_mom._camera_box_max_accel_y * 1.1,
                        max_width=self._hockey_mom._video_frame.width,
                        max_height=self._hockey_mom._video_frame.height,
                        color=(255, 128, 64),
                        thickness=5,
                        device=self._device,
                    )

                    size_unstick_size = self._hockey_mom._camera_box_max_speed_x * 5
                    size_stick_size = size_unstick_size / 3

                    self._current_roi_aspect = MovingBox(
                        label="AspectRatio",
                        bbox=self._current_roi,
                        arena_box=self.get_arena_box(),
                        max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1,
                        max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1,
                        max_accel_x=self._hockey_mom._camera_box_max_accel_x.clone(),
                        max_accel_y=self._hockey_mom._camera_box_max_accel_y.clone(),
                        max_width=self._hockey_mom._video_frame.width,
                        max_height=self._hockey_mom._video_frame.height,
                        width_change_threshold=_scalar_like(
                            size_unstick_size * 2, device=current_box.device
                        ),
                        width_change_threshold_low=_scalar_like(
                            size_stick_size * 2, device=current_box.device
                        ),
                        height_change_threshold=_scalar_like(
                            size_unstick_size * 2, device=current_box.device
                        ),
                        height_change_threshold_low=_scalar_like(
                            size_stick_size * 2, device=current_box.device
                        ),
                        sticky_translation=True,
                        sticky_sizing=True,
                        scale_width=1.1,
                        scale_height=1.1,
                        fixed_aspect_ratio=self._final_aspect_ratio,
                        color=(255, 0, 255),
                        thickness=5,
                        device=self._device,
                    )
                    self._current_roi = iter(self._current_roi)
                    self._current_roi_aspect = iter(self._current_roi_aspect)
                else:
                    self._current_roi.set_destination(
                        current_box, stop_on_dir_change=False
                    )

                self._current_roi = next(self._current_roi)
                self._current_roi_aspect = next(self._current_roi_aspect)
                if self._args.plot_moving_boxes:
                    self._current_roi_aspect.draw(img=online_im, draw_threasholds=True)
                    self._current_roi.draw(img=online_im)
                    vis.plot_line(
                        online_im,
                        center(self._current_roi.bbox),
                        center(current_box),
                        color=(255, 255, 255),
                        thickness=2,
                    )

                if (
                    self._video_output_boxtrack is not None
                    and self._current_roi is not None
                    and (self._video_output_campp is not None and not self._args.show_image)
                ):
                    imgproc_data = ImageProcData(
                        frame_id=frame_id.item(),
                        img=online_im,
                        current_box=self._current_roi_aspect.bounding_box(),
                    )
                    self._video_output_boxtrack.append(imgproc_data)

            # assert width(current_box) <= hockey_mom.video.width
            # assert height(current_box) <= hockey_mom.video.height

            outside_expanded_box = (
                current_box + self._outside_box_expansion_for_speed_curtailing
            )

            if self._args.plot_camera_tracking:
                vis.plot_rectangle(
                    online_im,
                    current_box,
                    color=(128, 0, 128),
                    thickness=2,
                    label="U:2&3",
                )

            if self._args.plot_speed:
                vis.plot_frame_id_and_speeds(
                    online_im,
                    -frame_id,
                    *self._hockey_mom.get_velocity_and_acceleratrion_xy(),
                )

            def _apply_temporal(
                current_box,
                last_box,
                scale_speed: float,
                grays_level: int = 128,
                verbose: bool = False,
            ):
                # assert width(current_box) <= hockey_mom.video.width
                # assert height(current_box) <= hockey_mom.video.height
                #
                # Temporal: Apply velocity and acceleration
                #
                # nonlocal current_box, self
                nonlocal self
                current_box = self._hockey_mom.get_next_temporal_box(
                    current_box,
                    last_box,
                    scale_speed=scale_speed,
                    verbose=verbose,
                )
                last_box = current_box.clone()
                if self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(grays_level, grays_level, grays_level),
                        thickness=2,
                        label="next_temporal_box",
                    )
                    # cv2.circle(
                    #     online_im,
                    #     [int(i) for i in center(current_box)],
                    #     radius=25,
                    #     color=(0, 0, 0),
                    #     thickness=25,
                    # )
                # assert width(current_box) <= hockey_mom.video.width
                # assert height(current_box) <= hockey_mom.video.height
                # assert width(last_box) <= hockey_mom.video.width
                # assert height(last_box) <= hockey_mom.video.height
                return current_box, last_box

            group_x_velocity, edge_center = self._hockey_mom.get_group_x_velocity(
                min_considered_velocity=3.0,
                group_threshhold=0.5,
                # min_considered_velocity=0.01,
                # group_threshhold=0.6,
            )
            if group_x_velocity:
                # print(f"frame {frame_id} group x velocity: {group_x_velocity}")
                # cv2.circle(
                #     online_im,
                #     [int(i) for i in edge_center],
                #     radius=30,
                #     color=(255, 0, 255),
                #     thickness=20,
                # )
                edge_center = torch.tensor(
                    edge_center, dtype=torch.float32, device=current_box.device
                )
                current_box = make_box_at_center(
                    edge_center, width(current_box), height(current_box)
                )
                # assert width(current_box) <= hockey_mom.video.width
                # assert height(current_box) <= hockey_mom.video.height
                self._hockey_mom._current_camera_box_speed_x += group_x_velocity / 2

                if self._current_roi is not None:
                    roi_center = center(self._current_roi.bounding_box())
                    # vis.plot_line(online_im, edge_center, roi_center, color=(128, 255, 128), thickness=4)
                    should_adjust_speed = torch.logical_or(
                        torch.logical_and(
                            group_x_velocity > 0, roi_center[0] < edge_center[0]
                        ),
                        torch.logical_and(
                            group_x_velocity < 0, roi_center[0] > edge_center[0]
                        ),
                    )
                    if should_adjust_speed.item():
                        self._current_roi.adjust_speed(
                            accel_x=group_x_velocity / 2,
                            accel_y=None,
                            use_constraints=False,
                            nonstop_delay=torch.tensor(
                                1, dtype=torch.int64, device=self._device
                            ),
                        )
                    else:
                        print("Skipping modifying group x velocity")
                        pass

            # current_box = hockey_mom.smooth_resize_box(current_box, self._last_temporal_box)
            current_box, self._last_temporal_box = _apply_temporal(
                current_box, self._last_temporal_box, scale_speed=1.0
            )

            if not group_x_velocity:
                self._hockey_mom.curtail_velocity_if_outside_box(
                    current_box, outside_expanded_box
                )

            #
            # HIJACK CURRENT ROI BOX POSITION
            #
            # current_box = self._hockey_mom.clamp(
            #     self._current_roi.bounding_box().clone()
            # )

            #
            # Aspect Ratio
            #
            # current_box = hockey_mom.clamp(current_box)
            # if self._args.plot_camera_tracking:
            #     vis.plot_rectangle(
            #         online_im,
            #         current_box,
            #         color=(0, 0, 0),
            #         thickness=10,
            #         label="fine_tracking_box",
            #     )

            fine_tracking_box = current_box.clone()

            # assert width(current_box) <= hockey_mom.video.width
            # assert height(current_box) <= hockey_mom.video.height

            current_box = self._hockey_mom.make_box_proper_aspect_ratio(
                frame_id=frame_id,
                the_box=current_box,
                desired_aspect_ratio=self._final_aspect_ratio,
                max_in_aspec_ratio=self._args.max_in_aspec_ratio,
            )
            assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

            # assert width(current_box) <= hockey_mom.video.width
            # assert height(current_box) <= hockey_mom.video.height

            current_box = self._hockey_mom.shift_box_to_edge(current_box)

            # assert width(current_box) <= hockey_mom.video.width
            # assert height(current_box) <= hockey_mom.video.height

            # print(f"shift_box_to_edge ar={aspect_ratio(current_box)}")
            if self._args.plot_camera_tracking:
                vis.plot_rectangle(
                    online_im,
                    current_box,
                    color=(255, 255, 255),  # White
                    thickness=1,
                    label="after-aspect",
                )
            if self._args.max_in_aspec_ratio:
                ZOOM_SHRINK_SIZE_INCREMENT = 1
                box_is_at_right_edge = self._hockey_mom.is_box_at_right_edge(
                    current_box
                )
                box_is_at_left_edge = self._hockey_mom.is_box_at_left_edge(current_box)
                cb_center = center(current_box)
                if box_is_at_right_edge:
                    lt_center = center(self._last_temporal_box)
                    # frame_center = center(hockey_mom._video_frame.bounding_box())
                    if cb_center[0] < lt_center[0]:
                        self._last_dx_shrink_size += ZOOM_SHRINK_SIZE_INCREMENT
                    elif cb_center[0] > lt_center[0]:
                        self._last_dx_shrink_size -= ZOOM_SHRINK_SIZE_INCREMENT
                elif box_is_at_left_edge:
                    lt_center = center(self._last_temporal_box)
                    # frame_center = center(hockey_mom._video_frame.bounding_box())
                    if cb_center[0] > lt_center[0]:
                        self._last_dx_shrink_size += ZOOM_SHRINK_SIZE_INCREMENT
                    elif cb_center[0] < lt_center[0]:
                        self._last_dx_shrink_size -= ZOOM_SHRINK_SIZE_INCREMENT
                else:
                    # When not on edge, decay away the shrink sizes
                    # TODO: do with min/max
                    if self._last_dx_shrink_size > 0:
                        self._last_dx_shrink_size -= ZOOM_SHRINK_SIZE_INCREMENT * 1.5
                        if self._last_dx_shrink_size < 0:
                            self._last_dx_shrink_size = 0
                    elif self._last_dx_shrink_size < 0:
                        self._last_dx_shrink_size += ZOOM_SHRINK_SIZE_INCREMENT * 1.5
                        if self._last_dx_shrink_size > 0:
                            self._last_dx_shrink_size = 0

                if self._last_dx_shrink_size > max_dx_shrink_size:
                    self._last_dx_shrink_size = max_dx_shrink_size
                elif self._last_dx_shrink_size < -max_dx_shrink_size:
                    self._last_dx_shrink_size = -max_dx_shrink_size
                if True:  # self._last_dx_shrink_size:
                    # print(f"Shrink width: {self._last_dx_shrink_size}")
                    w = width(current_box)
                    w -= self._last_dx_shrink_size
                    w = min(w, self._hockey_mom.video.width)
                    if box_is_at_right_edge:
                        self._center_dx_shift += 2
                        if self._center_dx_shift > self._last_dx_shrink_size:
                            self._center_dx_shift = self._last_dx_shrink_size
                        cb_center[0] += self._center_dx_shift
                    elif box_is_at_left_edge:
                        self._center_dx_shift -= 2
                        if self._center_dx_shift < -self._last_dx_shrink_size:
                            self._center_dx_shift = -self._last_dx_shrink_size
                    else:
                        if self._center_dx_shift < 0:
                            self._center_dx_shift += 2
                            if self._center_dx_shift > 0:
                                self._center_dx_shift = 0
                        elif self._center_dx_shift > 0:
                            self._center_dx_shift -= 2
                            if self._center_dx_shift < 0:
                                self._center_dx_shift = 0
                    cb_center[0] += self._center_dx_shift
                    h = w / self._final_aspect_ratio
                    h -= 1
                    w -= 1
                    current_box = torch.tensor(
                        (
                            cb_center[0] - (w / 2.0),
                            cb_center[1] - (h / 2.0),
                            cb_center[0] + (w / 2.0),
                            cb_center[1] + (h / 2.0),
                        ),
                        dtype=torch.float32,
                        device=cb_center.device,
                    )
                    # assert width(current_box) <= hockey_mom.video.width
                    # assert height(current_box) <= hockey_mom.video.height
                    current_box = self._hockey_mom.shift_box_to_edge(current_box)
                if self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(60, 60, 60),  # Gray
                        thickness=6,
                        label="after-aspect",
                    )

            # assert width(current_box) <= hockey_mom.video.width
            # assert height(current_box) <= hockey_mom.video.height

            assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

            def _fix_aspect_ratio(box):
                # assert width(box) <= hockey_mom.video.width
                # assert height(box) <= hockey_mom.video.height
                box = self._hockey_mom.make_box_proper_aspect_ratio(
                    frame_id=frame_id,
                    the_box=box,
                    desired_aspect_ratio=self._final_aspect_ratio,
                    max_in_aspec_ratio=False,
                )
                # assert width(box) <= hockey_mom.video.width
                # assert height(box) <= hockey_mom.video.height
                return self._hockey_mom.shift_box_to_edge(box)

            stuck = self._hockey_mom.did_direction_change(
                dx=True, dy=False, reset=False
            )

            if self._args.max_in_aspec_ratio:
                if self._last_sticky_temporal_box is not None:
                    gaussian_factor = self._get_gaussian(
                        online_im.shape[1]
                    ).get_gaussian_y_from_image_x_position(
                        center(self._last_sticky_temporal_box)[0]
                    )
                else:
                    gaussian_factor = 1
                # gaussian_mult = 10
                gaussian_mult = 6
                gaussian_add = gaussian_factor * gaussian_mult
                # print(f"gaussian_factor={gaussian_factor}, gaussian_add={gaussian_add}")
                sticky_size = self._hockey_mom._camera_box_max_speed_x * (
                    6 + gaussian_add
                )
                unsticky_size = sticky_size * 3 / 4
            else:
                sticky_size = self._hockey_mom._camera_box_max_speed_x * 5
                unsticky_size = sticky_size / 2

            if self._last_sticky_temporal_box is not None:
                # assert width(self._last_sticky_temporal_box) <= hockey_mom.video.width
                # assert height(self._last_sticky_temporal_box) <= hockey_mom.video.height

                if self._args.plot_sticky_camera:
                    vis.plot_rectangle(
                        online_im,
                        self._last_sticky_temporal_box,
                        color=(255, 255, 255),
                        thickness=6,
                    )
                    # sticky circle
                    cc = center(fine_tracking_box)
                    cl = center(self._last_sticky_temporal_box)

                    def _to_int(vals):
                        return [int(i) for i in vals]

                    cv2.circle(
                        online_im,
                        _to_int(cl),
                        radius=int(sticky_size),
                        color=(255, 0, 0) if stuck else (255, 255, 255),
                        thickness=3,
                    )
                    cv2.circle(
                        online_im,
                        _to_int(cl),
                        radius=int(unsticky_size),
                        color=(0, 255, 255) if stuck else (128, 128, 255),
                        thickness=2,
                    )
                    vis.plot_point(online_im, cl, color=(0, 0, 255), thickness=10)
                    vis.plot_point(online_im, cc, color=(0, 255, 0), thickness=6)
                    vis.plot_line(online_im, cl, cc, color=(255, 255, 255), thickness=2)

                    # current velocity vector
                    vis.plot_line(
                        online_im,
                        cl,
                        [
                            cl[0] + self._hockey_mom._current_camera_box_speed_x,
                            cl[1] + self._hockey_mom._current_camera_box_speed_y,
                        ],
                        color=(0, 0, 0),
                        thickness=2,
                    )
            else:
                self._last_sticky_temporal_box = current_box.clone()

            if self._args.apply_fixed_edge_scaling:
                cdist = center_x_distance(current_box, self._last_sticky_temporal_box)
            else:
                cdist = center_distance(current_box, self._last_sticky_temporal_box)

            # if stuck and (center_distance(current_box, self._last_sticky_temporal_box) > 30 or hockey_mom.is_fast(speed=10)):
            if stuck and (
                # center_distance(current_box, self._last_sticky_temporal_box) > 30
                # Past some distance of number of frames at max speed
                cdist
                > sticky_size
            ):
                self._hockey_mom.control_speed(
                    self._hockey_mom._camera_box_max_speed_x / 6,
                    self._hockey_mom._camera_box_max_speed_y / 6,
                    # set_speed_x=True,
                    set_speed_x=False,
                )
                self._hockey_mom.did_direction_change(dx=True, dy=True, reset=True)
                stuck = False
            elif cdist < unsticky_size:
                stuck = self._hockey_mom.set_direction_changed(dx=True, dy=True)

            if not stuck:
                # xx0 = center(current_box)[0]
                current_box, self._last_sticky_temporal_box = _apply_temporal(
                    current_box,
                    self._last_sticky_temporal_box,
                    scale_speed=1.0,
                    verbose=True,
                )

                # xx1 = center(current_box)[0]
                # print(f'A final temporal x change: {xx1 - xx0}')
                current_box = _fix_aspect_ratio(current_box)
                assert torch.isclose(
                    aspect_ratio(current_box), self._final_aspect_ratio
                )
                self._hockey_mom.did_direction_change(dx=True, dy=True, reset=True)
            elif self._last_sticky_temporal_box is None:
                self._last_sticky_temporal_box = current_box.clone()
                # assert width(self._last_sticky_temporal_box) <= hockey_mom.video.width
                # assert height(self._last_sticky_temporal_box) <= hockey_mom.video.height
                assert torch.isclose(
                    aspect_ratio(current_box), self._final_aspect_ratio
                )
            else:
                # assert width(self._last_sticky_temporal_box) <= hockey_mom.video.width
                # assert height(self._last_sticky_temporal_box) <= hockey_mom.video.height

                current_box = self._last_sticky_temporal_box.clone()
                current_box = _fix_aspect_ratio(current_box)
                assert torch.isclose(
                    aspect_ratio(current_box), self._final_aspect_ratio
                )

            if self._args.apply_fixed_edge_scaling:
                current_box = self._hockey_mom.apply_fixed_edge_scaling(
                    current_box,
                    edge_scaling_factor=self._args.fixed_edge_scaling_factor,
                )
                if self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(255, 0, 255),
                        thickness=5,
                        label="edge-scaled",
                    )

            if stuck and self._args.plot_camera_tracking:
                vis.plot_rectangle(
                    online_im,
                    current_box,
                    color=(0, 160, 255),
                    thickness=10,
                    label="stuck",
                )
            elif self._args.plot_camera_tracking:
                vis.plot_rectangle(
                    online_im,
                    current_box,
                    color=(160, 160, 255),
                    thickness=6,
                    label="post-sticky",
                )

            # Plot the trajectories
            if self._args.plot_individual_player_tracking:
                online_im = vis.plot_trajectory(
                    online_im,
                    self._hockey_mom.get_image_tracking(online_ids),
                    online_ids,
                )

            # kmeans = KMeans(n_clusters=3)
            # kmeans.fit(hockey_mom.online_image_center_points)
            # plt.scatter(x, y, c=kmeans.labels_)
            # plt.show()

            self._timer.toc()
            if frame_id.item() % 20 == 0:
                logger.info(
                    "Camera Processing frame {} ({:.2f} fps)".format(
                        frame_id.item(), 1.0 / max(1e-5, self._timer.average_time)
                    )
                )
                self._timer = Timer()

            assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
            if self._video_output_campp is not None:
                imgproc_data = ImageProcData(
                    frame_id=frame_id.item(),
                    img=online_im,
                    current_box=current_box,
                )
                self._video_output_campp.append(imgproc_data)
            # if (
            #     self._video_output_boxtrack is not None
            #     and self._current_roi is not None
            #     and (self._video_output_campp is not None and not self._args.show_image)
            # ):
            #     imgproc_data = ImageProcData(
            #         frame_id=frame_id.item(),
            #         img=online_im,
            #         current_box=self._current_roi_aspect.bounding_box(),
            #     )
            #     self._video_output_boxtrack.append(imgproc_data)


def _scalar_like(v, device):
    return torch.tensor(v, dtype=torch.float32, device=device)
