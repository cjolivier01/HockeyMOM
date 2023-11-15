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
from hmlib.camera.video_out import ImageProcData, VideoOutput
from hmlib.utils.image import ImageHorizontalGaussianDistribution

from hmlib.utils.box_functions import (
    width,
    height,
    center,
    center_x_distance,
    center_distance,
    aspect_ratio,
    make_box_at_center,
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

RINK_CONFIG = {
    "vallco": {
        "fixed_edge_scaling_factor": 0.8,
    },
    "dublin": {
        "fixed_edge_scaling_factor": 0.8,
    },
    "stockton": {
        "fixed_edge_scaling_factor": 0.6,
    },
    "roseville_2": {
        "fixed_edge_scaling_factor": 1.6,
    },
    "yerba_buena": {
        "fixed_edge_scaling_factor": 1.5,
    },
    "sharks_orange": {
        "fixed_edge_scaling_factor": 0.8,
    },
}

BASIC_DEBUGGING = False


class DefaultArguments(core.HMPostprocessConfig):
    def __init__(self, rink: str = "sharks_orange", args: argparse.Namespace = None):
        super().__init__()
        # Display the image every frame (slow)
        self.show_image = False or BASIC_DEBUGGING
        # self.show_image = True

        # Draw individual player boxes, tracking ids, speed and history trails
        self.plot_individual_player_tracking = True and BASIC_DEBUGGING
        # self.plot_individual_player_tracking = True

        # Draw all detection boxes (even if not tracking the detection)
        self.plot_all_detections = False
        # self.plot_all_detections = True

        # Draw intermediate boxes which are used to compute the final camera box
        self.plot_cluster_tracking = False or BASIC_DEBUGGING
        # self.plot_cluster_tracking = True

        # Use a differenmt algorithm when fitting to the proper aspect ratio,
        # such that the box calculated is much larger and often takes
        # the entire height.  The drawback is there's not much zooming.
        self.max_in_aspec_ratio = True
        # self.max_in_aspec_ratio = False

        # Zooming is fixed based upon the horizonal position's distance from center
        # self.apply_fixed_edge_scaling = False
        self.apply_fixed_edge_scaling = True

        self.fixed_edge_scaling_factor = RINK_CONFIG[rink]["fixed_edge_scaling_factor"]

        self.plot_camera_tracking = False or BASIC_DEBUGGING
        # self.plot_camera_tracking = False

        self.plot_moving_boxes = False or (
            BASIC_DEBUGGING
            and not (self.max_in_aspec_ratio or self.apply_fixed_edge_scaling)
        )
        # self.plot_moving_boxes = True

        # Print each frame number in the upper left corner
        self.plot_frame_number = False or BASIC_DEBUGGING
        # self.plot_frame_number = True

        # Plot frame ID and speed/velocity in upper-left corner
        self.plot_speed = False

        # self.fixed_edge_rotation = False
        self.fixed_edge_rotation = True

        self.fixed_edge_rotation_angle = 25.0
        # self.fixed_edge_rotation_angle = 35.0
        # self.fixed_edge_rotation_angle = 45.0

        # Use "sticky" panning, where panning occurs in less frequent,
        # but possibly faster, pans rather than a constant
        # pan (which may appear tpo "wiggle")
        # self.sticky_pan = True
        self.sticky_pan = False

        # Plot the component shapes directly related to camera stickiness
        self.plot_sticky_camera = False or BASIC_DEBUGGING
        # self.plot_sticky_camera = False

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

        # Make the image the same relative dimensions as the initial image,
        # such that the highest possible resolution is available when the camera
        # box is either the same height or width as the original video image
        # (Slower, but better final quality)
        self.scale_to_original_image = True
        # self.scale_to_original_image = False

        # Crop the final image to the camera window (possibly zoomed)
        self.crop_output_image = True and not BASIC_DEBUGGING
        # self.crop_output_image = False

        # Don't crop image, but performa of the calculations
        # except for the actual image manipulations
        self.fake_crop_output_image = False

        # Use cuda for final image resizing (if possible)
        self.use_cuda = False

        # Draw watermark on the image
        self.use_watermark = True
        # self.use_watermark = False

        self.detection_inclusion_box = None
        # self.detection_inclusion_box = [None, None, None, None]
        # self.detection_inclusion_box = [None, 140, None, None]

        # Roseville #2
        # self.detection_inclusion_box = torch.tensor(
        # [363, 600, 5388, 1714], dtype=torch.float32
        # )
        # print(f"Using Roseville2 inclusion box: {self.detection_inclusion_box}")

        # Above any of these lines, ignmore

        #
        # SHARKS ORANGE RINK
        #
        self.top_border_lines = [
            [2001, 123, 2629, 208],
            [1283, 117, 1979, 162],
        ]
        # Below any of these lines, ignore
        self.bottom_border_lines = [
            # tlbr
            # [6, 478, 225, 569],
            # [245, 571, 1650, 879],
            [21, 498, 1664, 878],
            [1662, 856, 3034, 799],
            [3044, 800, 3762, 673],
        ]


class BoundaryLines:
    def __init__(self, upper_border_lines, lower_border_lines):
        if upper_border_lines:
            self._upper_borders = torch.tensor(upper_border_lines, dtype=torch.float32)
            self._upper_line_vectors = self.tlbr_to_line_vectors(self._upper_borders)
        else:
            self._upper_borders = None
            self._upper_line_vectors = None
        if lower_border_lines:
            self._lower_borders = torch.tensor(lower_border_lines, dtype=torch.float32)
            self._lower_line_vectors = self.tlbr_to_line_vectors(self._lower_borders)
        else:
            self._lower_borders = None
            self._lower_line_vectors = None

    def tlbr_to_line_vectors(self, tlbr_batch):
        # Assuming tlbr_batch shape is (N, 4) with each box as [top, left, bottom, right]

        top_vectors = torch.stack(
            (tlbr_batch[:, 1], tlbr_batch[:, 0]), dim=1
        ) - torch.stack((tlbr_batch[:, 3], tlbr_batch[:, 0]), dim=1)
        left_vectors = torch.stack(
            (tlbr_batch[:, 1], tlbr_batch[:, 0]), dim=1
        ) - torch.stack((tlbr_batch[:, 1], tlbr_batch[:, 2]), dim=1)
        bottom_vectors = torch.stack(
            (tlbr_batch[:, 3], tlbr_batch[:, 2]), dim=1
        ) - torch.stack((tlbr_batch[:, 1], tlbr_batch[:, 2]), dim=1)
        right_vectors = torch.stack(
            (tlbr_batch[:, 3], tlbr_batch[:, 2]), dim=1
        ) - torch.stack((tlbr_batch[:, 3], tlbr_batch[:, 0]), dim=1)

        # Concatenating vectors for all sides
        line_vectors = torch.stack(
            (top_vectors, left_vectors, bottom_vectors, right_vectors), dim=1
        )

        return line_vectors

    def is_point_too_high(self, point):
        return False

    def is_point_too_low(self, point):
        too_low = self.is_point_below_line(point)
        return False

    def is_point_outside(self, point):
        return self.is_point_above_line(point) or self.is_point_below_line(point)

    def _is_point_above_line(self, point, line_start, line_end):
        # Create vectors
        line_vec = line_end - line_start
        point_vec = point - line_start

        # Compute the cross product
        cross_product_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]

        # Check if the point is above the line
        return cross_product_z < 0

    def is_point_above_line(self, point):
        if self._upper_borders is None:
            return False
        point = point.cpu()
        x = point[0]
        # y = point[1]
        for i, high_border in enumerate(self._upper_borders):
            if x >= high_border[0] and x <= high_border[2]:
                if self._is_point_above_line(point, high_border[0:2], high_border[2:4]):
                    return True
        return False

    def point_batch_check_point_above_segents(self, points):
        lines_start = self._upper_borders[:, 0:2]
        point_vecs = points[0] - lines_start
        cross_z = (
            self._upper_line_vectors[:, 0] * point_vecs[:, 1]
            - self._upper_line_vectors[:, 1] * point_vecs[:, 0]
        )
        return cross_z < 0

    def is_point_below_line(self, point):
        if self._lower_borders is None:
            return False
        point = point.cpu()
        x = point[0]
        # y = point[1]
        for i, low_border in enumerate(self._lower_borders):
            if x >= low_border[0] and x <= low_border[2]:
                if self._is_point_below_line(point, low_border[0:2], low_border[2:4]):
                    return True
        return False

    def _is_point_below_line(self, point, line_start, line_end):
        # Create vectors
        line_vec = line_end - line_start
        point_vec = point - line_start

        # Compute the cross product
        cross_product_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]

        # Check if the point is below the line
        return cross_product_z > 0


def scale_box(box, from_img, to_img):
    from_sz = (from_img.shape[1], from_img.shape[0])
    to_sz = (to_img.shape[1], to_img.shape[0])
    w_scale = to_sz[1] / from_sz[1]
    h_scale = to_sz[0] / from_sz[0]
    new_box = [box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale]
    print(f"from={box} -> to={new_box}")
    return new_box


def make_scale_array(from_img, to_img):
    from_sz = torch.tensor([from_img.shape[1], from_img.shape[0]], dtype=torch.float32)
    to_sz = torch.tensor([to_img.shape[1], to_img.shape[0]], dtype=torch.float32)
    return from_sz / to_sz


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
    # outside_boundaries = None
    # if boundaries is not None:
    #     outside_boundaries = boundaries.point_batch_check_point_above_segents(
    #         online_tlwhs_centers
    #     )
    for i in range(len(online_tlwhs_centers)):
        # if outside_boundaries is not None and outside_boundaries[i]:
        #     continue
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


class FramePostProcessor:
    def __init__(
        self,
        hockey_mom,
        start_frame_id,
        data_type,
        fps: float,
        save_dir,
        device,
        opt,
        args: argparse.Namespace,
        async_post_processing: bool = False,
        use_fork: bool = False,
    ):
        self._args = args
        self._start_frame_id = start_frame_id
        self._hockey_mom = hockey_mom
        self._queue = create_queue(mp=use_fork)
        self._data_type = data_type
        self._fps = fps
        self._opt = opt
        self._thread = None
        # self._imgproc_queue = create_queue(mp=use_fork)
        # self._imgproc_thread = None
        self._use_fork = use_fork
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float32)
        self._output_video = None
        self._final_image_processing_started = False
        self._async_post_processing = async_post_processing
        self._device = device
        self._horizontal_image_gaussian_distribution = None
        self._boundaries = None

        self._save_dir = save_dir
        # results_dir = Path(save_dir)
        # results_dir.mkdir(parents=True, exist_ok=True)

        # self.watermark = cv2.imread(
        #     os.path.realpath(
        #         os.path.join(
        #             os.path.dirname(os.path.realpath(__file__)),
        #             "..",
        #             "..",
        #             "..",
        #             "images",
        #             "sports_ai_watermark.png",
        #         )
        #     ),
        #     cv2.IMREAD_UNCHANGED,
        # )
        # self.watermark_height = self.watermark.shape[0]
        # self.watermark_width = self.watermark.shape[1]
        # self.watermark_rgb_channels = self.watermark[:, :, :3]
        # self.watermark_alpha_channel = self.watermark[:, :, 3]
        # self.watermark_mask = cv2.merge(
        #     [
        #         self.watermark_alpha_channel,
        #         self.watermark_alpha_channel,
        #         self.watermark_alpha_channel,
        #     ]
        # )

        self._outside_box_expansion_for_speed_curtailing = torch.tensor(
            [-100.0, -100.0, 100.0, 100.0],
            dtype=torch.float32,
            device=self._device,
        )

        if self._args.top_border_lines or self._args.bottom_border_lines:
            self._boundaries = BoundaryLines(
                self._args.top_border_lines, self._args.bottom_border_lines
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
        self._video_output = None

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
            # self._imgproc_thread = Thread(
            #     target=self._start_final_image_processing, name="FinalImgProc"
            # )
            # self._imgproc_thread.start()

    def _start(self):
        # if self._args.fake_crop_output_image:
        #     self.crop_output_image = True
        return self.postprocess_frame_worker()

    # def _start_final_image_processing(self):
    #     return self.final_image_processing()

    def stop(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
            # self._imgproc_queue.put(None)
            # self._imgproc_thread.join()
            # self._imgproc_thread = None

        elif self.use_fork:
            self._queue.put(None)
            if self._child_pid:
                os.waitpid(self._child_pid)
        self._video_output.stop()

    def send(
        self, online_tlwhs, online_ids, detections, info_imgs, image, original_img
    ):
        while self._queue.qsize() > 10:
            time.sleep(0.001)
        try:
            dets = [
                Detection(track_id=d.track_id, tlwh=d.tlwh, history=d.history)
                for d in detections
            ]
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
        # self._last_temporal_box = None
        # self._last_sticky_temporal_box = None
        # self._last_dx_shrink_size = 0
        # max_dx_shrink_size = 100
        self._center_dx_shift = 0
        timer = Timer()

        if self._args.crop_output_image and not self._args.fake_crop_output_image:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = (
                self._hockey_mom.video.height * self._final_aspect_ratio
            )
        else:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = self._hockey_mom.video.width

        assert self._video_output is None
        self._video_output = VideoOutput(
            args=self._args,
            output_video_path=os.path.join(self._save_dir, "tracking_output.avi"),
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
        self._video_output.start()

        # self._imgproc_queue.put("ready")
        # frame_id = self._start_frame_id - 1
        while True:
            # frame_id += 1
            online_targets_and_img = self._queue.get()
            if online_targets_and_img is None:
                break
            self.cam_postprocess(online_targets_and_img=online_targets_and_img)

    _INFO_IMGS_FRAME_ID_INDEX = 2

    def get_arena_box(self):
        # if self._args.detection_inclusion_box is not None:
        #     return self._args.detection_inclusion_box
        return self._hockey_mom._video_frame.bounding_box()

    def cam_postprocess(self, online_targets_and_img):
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

        self._hockey_mom.reset_clusters()

        def _kmeans_cuda_device():
            # return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            return self._device

        kmeans_device = _kmeans_cuda_device()
        self._hockey_mom.calculate_clusters(n_clusters=2, device=kmeans_device)
        self._hockey_mom.calculate_clusters(n_clusters=3, device=kmeans_device)

        if self._args.show_image or self._save_dir is not None:
            if self._args.scale_to_original_image:
                if isinstance(original_img, torch.Tensor):
                    original_img = original_img.numpy()
                online_im = original_img
                del original_img
            else:
                online_im = img0
            online_im = self.prepare_online_image(online_im)

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
                        # fps=1.0 / timer.average_time if timer.average_time else 1000.0,
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
                    # fps=1.0 / timer.average_time if timer.average_time else 1000.0,
                    speeds=[],
                    line_thickness=2,
                )

            # Examine as 2 clusters
            largest_cluster_ids_2 = self._hockey_mom.prune_not_in_largest_cluster(
                n_clusters=2, ids=online_ids
            )
            if largest_cluster_ids_2:
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
            largest_cluster_ids_3 = self._hockey_mom.prune_not_in_largest_cluster(
                n_clusters=3, ids=online_ids
            )
            if largest_cluster_ids_3:
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
            if self._current_roi is None:
                # start_box = self._hockey_mom._video_frame.bounding_box()
                start_box = current_box
                self._current_roi = MovingBox(
                    label="Current ROI",
                    bbox=start_box,
                    # arena_box=self._hockey_mom._video_frame.bounding_box(),
                    arena_box=self.get_arena_box(),
                    max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1.5,
                    max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1.5,
                    max_accel_x=self._hockey_mom._camera_box_max_accel_x * 1.1,
                    max_accel_y=self._hockey_mom._camera_box_max_accel_y * 1.1,
                    max_width=self._hockey_mom._video_frame.width,
                    max_height=self._hockey_mom._video_frame.height,
                    color=(255, 128, 64),
                    thickness=5,
                )

                unstick_size = self._hockey_mom._camera_box_max_speed_x * 5
                stick_size = unstick_size / 2

                size_unstick_size = self._hockey_mom._camera_box_max_speed_x * 5
                size_stick_size = size_unstick_size / 3

                self._current_roi_aspect = MovingBox(
                    label="AspectRatio",
                    bbox=self._current_roi,
                    # arena_box=self._hockey_mom._video_frame.bounding_box(),
                    arena_box=self.get_arena_box(),
                    max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1,
                    max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1,
                    max_accel_x=self._hockey_mom._camera_box_max_accel_x,
                    max_accel_y=self._hockey_mom._camera_box_max_accel_y,
                    max_width=self._hockey_mom._video_frame.width,
                    max_height=self._hockey_mom._video_frame.height,
                    width_change_threshold=_scalar_like(
                        size_unstick_size, device=current_box.device
                    ),
                    width_change_threshold_low=_scalar_like(
                        size_stick_size, device=current_box.device
                    ),
                    height_change_threshold=_scalar_like(
                        size_unstick_size, device=current_box.device
                    ),
                    height_change_threshold_low=_scalar_like(
                        size_stick_size, device=current_box.device
                    ),
                    sticky_translation=True,
                    # translation_threshold=_scalar_like(
                    #     unstick_size, device=current_box.device
                    # ),
                    # translation_threshold_low=_scalar_like(
                    #     stick_size, device=current_box.device
                    # ),
                    scale_width=1.1,
                    scale_height=1.1,
                    fixed_aspect_ratio=self._final_aspect_ratio,
                    color=(255, 0, 255),
                    thickness=5,
                )
                self._current_roi = iter(self._current_roi)
                self._current_roi_aspect = iter(self._current_roi_aspect)
            else:
                self._current_roi.set_destination(current_box, stop_on_dir_change=False)

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

            # assert width(current_box) <= hockey_mom.video.width
            # assert height(current_box) <= hockey_mom.video.height

            # TODO: make this expand box a class member so not recreated every time
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
                min_considered_velocity=3.5,
                group_threshhold=0.5,
                # min_considered_velocity=0.01,
                # group_threshhold=0.6,
            )
            if group_x_velocity:
                # print(f"frame {frame_id} group x velocity: {group_x_velocity}")
                # cv2.circle(
                #     online_im,
                #     _to_int(edge_center),
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
                self._current_roi.adjust_speed(
                    accel_x=group_x_velocity / 2,
                    accel_y=None,
                    use_constraints=False,
                    nonstop_delay=torch.tensor(
                        1, dtype=torch.int64, device=self._device
                    ),
                )

            # current_box = hockey_mom.smooth_resize_box(current_box, self._last_temporal_box)
            current_box, self._last_temporal_box = _apply_temporal(
                current_box, self._last_temporal_box, scale_speed=1.0
            )

            # vis.plot_rectangle(
            #     online_im,
            #     outside_expanded_box,
            #     color=(0, 0, 0),
            #     thickness=5,
            #     label="",
            # )
            if not group_x_velocity:
                self._hockey_mom.curtail_velocity_if_outside_box(
                    current_box, outside_expanded_box
                )

            #
            # Aspect Ratio
            #
            # current_box = hockey_mom.clamp(current_box)
            if self._args.plot_camera_tracking:
                vis.plot_rectangle(
                    online_im,
                    current_box,
                    color=(0, 0, 0),
                    thickness=10,
                    label="fine_tracking_box",
                )
                # vis.plot_rectangle(
                #     online_im,
                #     current_box,
                #     color=(0, 0, 0),
                #     thickness=10,
                #     label="clamped_pre_aspect",
                # )

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
                # movement_speed_divisor = 1.0
            else:
                sticky_size = self._hockey_mom._camera_box_max_speed_x * 5
                unsticky_size = sticky_size / 2
                # movement_speed_divisor = 3.0

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
                    # cc = center(current_box)
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

            assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
            imgproc_data = ImageProcData(
                frame_id=frame_id.item(),
                img=online_im,
                current_box=current_box
                if (
                    self._args.max_in_aspec_ratio or self._args.apply_fixed_edge_scaling
                )
                else self._current_roi_aspect.bounding_box(),
            )
            self._video_output.append(imgproc_data)


def _scalar_like(v, device):
    return torch.tensor(v, dtype=torch.float32, device=device)
