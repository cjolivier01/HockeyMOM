from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import cv2
import argparse
import numpy as np
import traceback

import yaml
from typing import List, Dict

import torch
import torchvision as tv

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.utils.utils import create_queue
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.camera.moving_box import MovingBox
from hmlib.video_out import ImageProcData, VideoOutput
from hmlib.camera.clusters import ClusterMan
from hmlib.utils.image import ImageHorizontalGaussianDistribution, make_visible_image
from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.config import get_nested_value

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
    get_enclosing_box,
    tlwh_to_tlbr_single,
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
class DefaultArguments(core.HMPostprocessConfig):
    def __init__(
        self,
        game_config: Dict,
        basic_debugging: int = 0,
        output_video_path: str = None,
        opts: argparse.Namespace = None,
    ):
        # basic_debugging = False
        self.debug = int(basic_debugging)

        super().__init__()

        self.game_config = game_config

        self._output_video_path = output_video_path

        # Display the image every frame (slow)
        self.show_image = self.show_image or basic_debugging

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

        self.fixed_edge_scaling_factor = self.game_config["rink"]["camera"][
            "fixed_edge_scaling_factor"
        ]

        self.plot_camera_tracking = False or basic_debugging
        # self.plot_camera_tracking = True

        self.plot_moving_boxes = False or basic_debugging
        # self.plot_moving_boxes = True

        # self.old_tracking_use_new_moving_box = True
        self.old_tracking_use_new_moving_box = False

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

        self.fixed_edge_rotation_angle = self.game_config["rink"]["camera"][
            "fixed_edge_rotation_angle"
        ]

        # Plot the component shapes directly related to camera stickiness
        self.plot_sticky_camera = False or basic_debugging
        # self.plot_sticky_camera = True

        # self.cam_ignore_largest = self.game_config["rink"]["tracking"]["cam_ignore_largest"]
        self.cam_ignore_largest = get_nested_value(
            self.game_config, "rink.tracking.cam_ignore_largest", default_value=False
        )

        # Crop the final image to the camera window (possibly zoomed)
        self.crop_output_image = True and not basic_debugging
        # self.crop_output_image = False

        # Use cuda for final image resizing (if possible)
        self.use_cuda = False

        # Draw watermark on the image
        self.use_watermark = True
        # self.use_watermark = False

        # Deprecated
        self.detection_inclusion_box = None

        self.skip_final_video_save = False

        #
        # Detection boundaries
        # TODO: Somehow move into boundaries class like witht he clip stuff
        # TODO: Get rid of this, probably no longer needed
        #
        self.top_border_lines = get_nested_value(
            self.game_config, "game.boundaries.upper", []
        )
        self.bottom_border_lines = get_nested_value(
            self.game_config, "game.boundaries.lower", []
        )
        upper_tune_position = get_nested_value(
            self.game_config, "game.boundaries.upper_tune_position", []
        )
        lower_tune_position = get_nested_value(
            self.game_config, "game.boundaries.lower_tune_position", []
        )
        boundary_scale_width = get_nested_value(
            self.game_config, "game.boundaries.scale_width", 1.0
        )
        boundary_scale_height = get_nested_value(
            self.game_config, "game.boundaries.scale_width", 1.0
        )
        if self.top_border_lines and upper_tune_position:
            for i in range(len(self.top_border_lines)):
                if boundary_scale_width:
                    self.top_border_lines[i][0] *= boundary_scale_width
                    self.top_border_lines[i][1] *= boundary_scale_width
                if boundary_scale_height:
                    self.top_border_lines[i][2] *= boundary_scale_height
                    self.top_border_lines[i][3] *= boundary_scale_height
                self.top_border_lines[i][0] += upper_tune_position[0]
                self.top_border_lines[i][2] += upper_tune_position[0]
                self.top_border_lines[i][1] += upper_tune_position[1]
                self.top_border_lines[i][3] += upper_tune_position[1]

        if self.bottom_border_lines and lower_tune_position:
            for i in range(len(self.top_border_lines)):
                if boundary_scale_width:
                    self.bottom_border_lines[i][0] *= boundary_scale_width
                    self.bottom_border_lines[i][1] *= boundary_scale_width
                if boundary_scale_height:
                    self.bottom_border_lines[i][3] *= boundary_scale_height
                    self.bottom_border_lines[i][2] *= boundary_scale_height
                self.bottom_border_lines[i][0] += lower_tune_position[0]
                self.bottom_border_lines[i][2] += lower_tune_position[0]
                self.bottom_border_lines[i][1] += lower_tune_position[1]
                self.bottom_border_lines[i][3] += lower_tune_position[1]

        if opts is not None:
            self.copy_args_if_not_exist(opts, self)

    @staticmethod
    def copy_args_if_not_exist(source, target):
        """
        Copy all attributes from source to target if they don't already exist in target.

        Parameters:
        - source: An object (e.g., argparse.Namespace) from which to copy attributes.
        - target: The target object to which attributes should be copied.
        """
        for attribute in vars(source):
            if not attribute.startswith("_"):
                if not hasattr(target, attribute):
                    setattr(target, attribute, getattr(source, attribute))


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


class BreakawayDetection:
    def __init__(self, config: dict):
        breakaway_detection = get_nested_value(
            config, "rink.camera.breakaway_detection", None
        )
        self.min_considered_group_velocity = breakaway_detection[
            "min_considered_group_velocity"
        ]
        self.group_ratio_threshold = breakaway_detection["group_ratio_threshold"]
        self.group_velocity_speed_ratio = breakaway_detection[
            "group_velocity_speed_ratio"
        ]
        self.scale_speed_constraints = breakaway_detection["scale_speed_constraints"]
        self.nonstop_delay_count = breakaway_detection["nonstop_delay_count"]
        self.overshoot_scale_speed_ratio = breakaway_detection[
            "overshoot_scale_speed_ratio"
        ]


class CamTrackPostProcessor(torch.nn.Module):
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
        self._original_clip_box = original_clip_box
        self._breakaway_detection = BreakawayDetection(args.game_config)

        if self._video_out_device is None:
            self._video_out_device = self._device

        self._save_dir = save_dir
        self._save_frame_dir = save_frame_dir

        # self._outside_box_expansion_for_speed_curtailing = torch.tensor(
        #     [-100.0, -100.0, 100.0, 100.0],
        #     dtype=torch.float32,
        #     device=self._device,
        # )

        if self._args.top_border_lines or self._args.bottom_border_lines:
            self._boundaries = BoundaryLines(
                self._args.top_border_lines,
                self._args.bottom_border_lines,
                self._original_clip_box,
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
        self._queue_timer = Timer()
        self._send_to_timer_post_process = Timer()

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
        try:
            with TimeTracker(
                "Send to cam post process queue",
                self._send_to_timer_post_process,
                print_interval=50,
            ):
                wait_count = 0
                while self._queue.qsize() > 1:
                    if not self._args.debug and not self._args.show_image:
                        wait_count += 1
                        if wait_count % 10 == 0:
                            print("Cam post-process queue too large")
                    time.sleep(0.001)
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
            # TODO: Does self._hockey_mom.video.height take into account clipping of the stitched frame?

            # self.final_frame_width = min(self._hockey_mom.video.width, 4096)
            # self.final_frame_height = self.final_frame_width / self._final_aspect_ratio
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = (
                self._hockey_mom.video.height * self._final_aspect_ratio
            )
            if self.final_frame_width > 4096:
                self.final_frame_width = 4096
                self.final_frame_height = (
                    self.final_frame_width / self._final_aspect_ratio
                )

        else:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = self._hockey_mom.video.width

        assert self._video_output_campp is None
        output_video_path = self._args._output_video_path
        if output_video_path is None:
            output_video_path = (
                os.path.join(self._save_dir, "tracking_output.mkv")
                if self._save_dir is not None
                else None
            )
        self._video_output_campp = VideoOutput(
            name="TRACKING",
            args=self._args,
            output_video_path=output_video_path,
            fps=self._fps,
            use_fork=False,
            start=False,
            output_frame_width=self.final_frame_width,
            output_frame_height=self.final_frame_height,
            save_frame_dir=self._save_frame_dir,
            original_clip_box=self._original_clip_box,
            watermark_image_path=(
                os.path.realpath(
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
                else None
            ),
            device=self._video_out_device,
            skip_final_save=self._args.skip_final_video_save,
            image_channel_adjustment=self._args.game_config["rink"]["camera"][
                "image_channel_adjustment"
            ],
        )
        self._video_output_campp.start()

        while True:
            online_targets_and_img = self._queue.get()
            if online_targets_and_img is None:
                break

            frame_id, online_im, current_box = self.forward(
                online_targets_and_img=online_targets_and_img
            )

            assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
            if self._video_output_campp is not None:
                imgproc_data = ImageProcData(
                    frame_id=frame_id.item(),
                    img=online_im,
                    current_box=current_box,
                )
                self._video_output_campp.append(imgproc_data)

    _INFO_IMGS_FRAME_ID_INDEX = 2

    def get_arena_box(self):
        return self._hockey_mom._video_frame.bounding_box()

    def _kmeans_cuda_device(self):
        if self._use_fork:
            return "cpu"
        # return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        return "cpu"
        # return self._device

    def get_cluster_boxes(
        self,
        online_tlwhs: torch.Tensor,
        online_ids: torch.Tensor,
        cluster_counts: List[int],
    ):
        if self._cluster_man is None:
            self._cluster_man = ClusterMan(
                sizes=cluster_counts, device=self._kmeans_cuda_device()
            )

        self._cluster_man.calculate_all_clusters(
            center_points=center_batch(online_tlwhs), ids=online_ids
        )
        boxes_map = dict()
        boxes_list = []
        for cluster_count in cluster_counts:
            largest_cluster_ids = self._cluster_man.prune_not_in_largest_cluster(
                num_clusters=cluster_count, ids=online_ids
            )
            if len(largest_cluster_ids):
                largest_cluster_ids_box = self._hockey_mom.get_current_bounding_box(
                    largest_cluster_ids
                )
                boxes_map[cluster_count] = largest_cluster_ids_box
                boxes_list.append(largest_cluster_ids_box)
            else:
                largest_cluster_ids_box = None
        if not boxes_map:
            return {}, None
        return boxes_map, torch.stack(boxes_list)

    def forward(self, online_targets_and_img):
        self._timer.tic()
        # max_dx_shrink_size = 100  # ???

        online_tlwhs = online_targets_and_img[0]
        online_ids = online_targets_and_img[1]
        # detections = online_targets_and_img[2]
        info_imgs = online_targets_and_img[3]

        frame_ids = info_imgs[self._INFO_IMGS_FRAME_ID_INDEX]
        frame_id = frame_ids[self._frame_counter % len(frame_ids)]
        self._frame_counter += 1

        largest_bbox = None

        # Exclude detections outside of an optional bounding box
        online_tlwhs, online_ids = prune_by_inclusion_box(
            online_tlwhs,
            online_ids,
            self._args.detection_inclusion_box,
            boundaries=self._boundaries,
        )

        if self._args.cam_ignore_largest and len(online_tlwhs):
            # Don't remove unless we have at least 4 online items being tracked
            online_tlwhs, mask, largest_bbox = remove_largest_bbox(
                online_tlwhs, min_boxes=4
            )
            online_ids = online_ids[mask]

        # info_imgs = online_targets_and_img[3]
        original_img = online_targets_and_img[5]

        online_im = original_img
        # online_im = original_img.clone()
        if online_im.ndim == 4:
            assert online_im.shape[0] == 1
            online_im = online_im.squeeze(0)

        self._hockey_mom.append_online_objects(online_ids, online_tlwhs)

        #
        # Clusters
        #
        cluster_counts = [3, 2]
        cluster_boxes_map, cluster_boxes = self.get_cluster_boxes(
            online_tlwhs, online_ids, cluster_counts=cluster_counts
        )

        if cluster_boxes_map:
            cluster_enclosing_box = get_enclosing_box(cluster_boxes)
        elif self._previous_cluster_union_box is not None:
            cluster_enclosing_box = self._previous_cluster_union_box.clone()
        else:
            cluster_enclosing_box = self._hockey_mom._video_frame.bounding_box()

        current_box = cluster_enclosing_box

        if self._args.plot_boundaries and self._boundaries is not None:
            online_im = self._boundaries.draw(online_im)

        if self._args.plot_individual_player_tracking:
            online_im = vis.plot_tracking(
                online_im,
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                speeds=[],
                line_thickness=2,
            )
            # print(f"Tracking {len(online_ids)} players...")
            if largest_bbox is not None:
                online_im = vis.plot_rectangle(
                    online_im,
                    [int(i) for i in tlwh_to_tlbr_single(largest_bbox)],
                    color=(0, 0, 0),
                    thickness=1,
                    label=f"IGNORED",
                )

        if self._args.plot_cluster_tracking:
            cluster_box_colors = {
                cluster_counts[0]: (128, 0, 0),  # dark red
                cluster_counts[1]: (0, 0, 128),  # dark blue
            }
            assert len(cluster_counts) == len(cluster_box_colors)
            for cc in cluster_counts:
                if cc in cluster_boxes_map:
                    online_im = vis.plot_rectangle(
                        online_im,
                        cluster_boxes_map[cc],
                        color=cluster_box_colors[cc],
                        thickness=1,
                        label=f"cluster_box_{cc}",
                    )

            if cluster_boxes_map:
                # The union of the two cluster boxes
                online_im = vis.plot_alpha_rectangle(
                    online_im,
                    cluster_enclosing_box,
                    color=(64, 64, 64),  # dark gray
                    label="union_clusters",
                    opacity_percent=25,
                )

        if current_box is None:
            assert False  # how does this happen?
            current_box = self._hockey_mom._video_frame.bounding_box()

        self._previous_cluster_union_box = current_box.clone()

        # Some players may be off-screen, so their box may go over an edge
        current_box = self._hockey_mom.clamp(current_box)

        #
        # Current ROI box
        #
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
            # self._current_roi.eval()

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
                scale_width=self._args.game_config["rink"]["camera"][
                    "follower_box_scale_width"
                ],
                scale_height=self._args.game_config["rink"]["camera"][
                    "follower_box_scale_height"
                ],
                fixed_aspect_ratio=self._final_aspect_ratio,
                color=(255, 0, 255),
                thickness=5,
                device=self._device,
            )
            self._current_roi = iter(self._current_roi)
            self._current_roi_aspect = iter(self._current_roi_aspect)
            # self._current_roi_aspect.eval()
        else:
            # self._current_roi.set_destination(current_box, stop_on_dir_change=False)
            self._current_roi.forward(current_box, stop_on_dir_change=False)

        self._current_roi = next(self._current_roi)
        self._current_roi_aspect = next(self._current_roi_aspect)
        if self._args.plot_moving_boxes:
            online_im = self._current_roi_aspect.draw(
                img=online_im, draw_threasholds=True
            )
            online_im = self._current_roi.draw(img=online_im)
            online_im = vis.plot_line(
                online_im,
                center(self._current_roi.bbox),
                center(current_box),
                color=(255, 255, 255),
                thickness=2,
            )

        if self._args.plot_camera_tracking:
            online_im = vis.plot_rectangle(
                online_im,
                current_box,
                color=(128, 0, 128),
                thickness=2,
                label="U:2&3",
            )

        if self._args.plot_speed:
            vis.plot_frame_id_and_speeds(
                online_im,
                frame_id,
                *self._hockey_mom.get_velocity_and_acceleratrion_xy(),
            )

        group_x_velocity, edge_center = self._hockey_mom.get_group_x_velocity(
            min_considered_velocity=self._breakaway_detection.min_considered_group_velocity,
            group_threshhold=self._breakaway_detection.group_ratio_threshold,
        )
        #
        # Breakway detection
        #
        if group_x_velocity:
            # print(f"frame {frame_id} group x velocity: {group_x_velocity}")
            if self._args.plot_individual_player_tracking:
                cv2.circle(
                    online_im,
                    [int(i) for i in edge_center],
                    radius=30,
                    color=(255, 0, 255),
                    thickness=20,
                )
            edge_center = torch.tensor(
                edge_center, dtype=torch.float32, device=current_box.device
            )
            current_box = make_box_at_center(
                edge_center, width(current_box), height(current_box)
            )

            # If group x velocity is in different direction than current speed, behave a little differently
            if self._current_roi is not None:
                roi_center = center(self._current_roi_aspect.bounding_box())
                if self._args.plot_individual_player_tracking:
                    vis.plot_line(
                        online_im,
                        edge_center,
                        roi_center,
                        color=(128, 255, 128),
                        thickness=4,
                    )
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
                        accel_x=group_x_velocity
                        * self._breakaway_detection.group_velocity_speed_ratio,
                        accel_y=None,
                        scale_constraints=self._breakaway_detection.scale_speed_constraints,
                        nonstop_delay=torch.tensor(
                            self._breakaway_detection.nonstop_delay_count,
                            dtype=torch.int64,
                            device=self._device,
                        ),
                    )
                else:
                    # Cut the speed quickly due to overshoot
                    # self._current_roi.scale_speed(ratio_x=0.6)
                    self._current_roi.scale_speed(
                        ratio_x=self._breakaway_detection.overshoot_scale_speed_ratio
                    )
                    # print("Reducing group x velocity due to overshoot")

        return frame_id, online_im, self._current_roi_aspect.bounding_box()


def _scalar_like(v, device):
    if isinstance(v, torch.Tensor):
        return v.clone()
    return torch.tensor(v, dtype=torch.float32, device=device)
