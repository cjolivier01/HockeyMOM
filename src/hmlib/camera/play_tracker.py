from __future__ import absolute_import, division, print_function

import argparse
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.bbox.box_functions import (
    center,
    center_batch,
    clamp_box,
    get_enclosing_box,
    height,
    make_box_at_center,
    remove_largest_bbox,
    scale_box,
    tlwh_centers,
    tlwh_to_tlbr_single,
    width,
)
from hmlib.builder import HM
from hmlib.camera.camera import HockeyMOM
from hmlib.camera.clusters import ClusterMan
from hmlib.camera.moving_box import MovingBox
from hmlib.config import get_nested_value
from hmlib.jersey.jersey_tracker import JerseyTracker
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis

# from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.utils.gpu import StreamCheckpoint, StreamTensor
from hmlib.utils.image import make_channels_last
from hmlib.utils.progress_bar import ProgressBar
from hockeymom.core import AllLivingBoxConfig, PlayTrackerConfig, BBox
from hockeymom.core import PlayTracker as CppPlayTracker

from .living_box import PyLivingBox, from_bbox, to_bbox

_CPP_BOXES: bool = True
# _CPP_BOXES: bool = False

_CPP_PLAYTRACKER: bool = True and _CPP_BOXES
# _CPP_PLAYTRACKER: bool = False and _CPP_BOXES


def batch_tlbrs_to_tlwhs(tlbrs: torch.Tensor) -> torch.Tensor:
    tlwhs = tlbrs.clone()
    # make boxes tlwh
    tlwhs[:, 2] = tlwhs[:, 2] - tlwhs[:, 0]  # width = x2 - x1
    tlwhs[:, 3] = tlwhs[:, 3] - tlwhs[:, 1]  # height = y2 - y1
    return tlwhs


class BreakawayDetection:
    def __init__(self, config: dict):
        breakaway_detection = get_nested_value(config, "rink.camera.breakaway_detection", None)
        self.min_considered_group_velocity = breakaway_detection["min_considered_group_velocity"]
        self.group_ratio_threshold = breakaway_detection["group_ratio_threshold"]
        self.group_velocity_speed_ratio = breakaway_detection["group_velocity_speed_ratio"]
        self.scale_speed_constraints = breakaway_detection["scale_speed_constraints"]
        self.nonstop_delay_count = breakaway_detection["nonstop_delay_count"]
        self.overshoot_scale_speed_ratio = breakaway_detection["overshoot_scale_speed_ratio"]


@HM.register_module()
class PlayTracker(torch.nn.Module):

    def __init__(
        self,
        hockey_mom: HockeyMOM,
        play_box: torch.Tensor,
        device: torch.device,
        original_clip_box: Optional[torch.Tensor],
        progress_bar: Optional[ProgressBar],
        args: argparse.Namespace,
        cpp_boxes: bool = _CPP_BOXES,
        cpp_playtracker: bool = _CPP_PLAYTRACKER,
    ):
        """
        Track the play

        :param hockey_mom: The old HockeyMom object
        :param play_box: The box allowed for play (assumed the visual play does not exist outside of this box)
        :param device: Device to use for computations`
        :param original_clip_box: Clip box that has been applied to the original image (if any)
        :param progress_bar: Progress bar
        :param args: _description_
        """
        super(PlayTracker, self).__init__()
        self._args = args
        self._cpp_boxes = cpp_boxes
        self._cpp_playtracker = cpp_playtracker
        self._playtracker: Union[PlayTracker, None] = None
        self._hockey_mom: HockeyMOM = hockey_mom
        # Amount to scale speed-related calculations based upon non-standard fps
        self._play_box = clamp_box(play_box, hockey_mom._video_frame.bounding_box())
        self._thread = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._output_video = None
        self._final_image_processing_started = False
        self._device = device
        self._horizontal_image_gaussian_distribution = None
        self._boundaries = None
        self._cluster_man: Optional[ClusterMan] = None
        self._original_clip_box = original_clip_box
        self._progress_bar = progress_bar

        self._jersey_tracker = JerseyTracker(show=args.plot_jersey_numbers)

        # Tracking specific ids
        self._track_ids: Set[int] = set()
        if args.track_ids:
            self._track_ids = set([int(i) for i in args.track_ids.split(",")])

        # if (
        #     self._args.plot_boundaries
        #     and self._args.top_border_lines
        #     or self._args.bottom_border_lines
        # ):
        #     # Only used for plotting the lines
        #     self._boundaries = BoundaryLines(
        #         self._args.top_border_lines,
        #         self._args.bottom_border_lines,
        #         self._original_clip_box,
        #     )

        # Persistent state across frames
        self._previous_cluster_union_box = None
        self._last_temporal_box = None
        self._last_sticky_temporal_box = None
        self._frame_counter: int = 0

        play_width = width(self._play_box)
        play_height = height(self._play_box)

        assert play_width <= self._hockey_mom._video_frame.width
        assert play_height <= self._hockey_mom._video_frame.height

        # speed_scale = 1.0
        speed_scale = self._hockey_mom.fps_speed_scale

        start_box = self._play_box.clone()

        if self._cpp_boxes:

            # Create and configure `AllLivingBoxConfig` for `_current_roi`
            current_roi_config = AllLivingBoxConfig()

            # Translation
            config.max_speed_x = self._hockey_mom._camera_box_max_speed_x * 1.5 / speed_scale
            config.max_speed_y = self._hockey_mom._camera_box_max_speed_y * 1.5 / speed_scale
            config.max_accel_x = self._hockey_mom._camera_box_max_accel_x * 1.1 / speed_scale
            config.max_accel_y = self._hockey_mom._camera_box_max_accel_y * 1.1 / speed_scale

            # Resizing
            config.max_speed_w = self._hockey_mom._camera_box_max_speed_x * 1.5 / speed_scale / 1.8
            config.max_speed_h = self._hockey_mom._camera_box_max_speed_y * 1.5 / speed_scale / 1.8
            config.max_accel_w = self._hockey_mom._camera_box_max_accel_x * 1.1 / speed_scale
            config.max_accel_h = self._hockey_mom._camera_box_max_accel_y * 1.1 / speed_scale

            current_roi_config.max_width = play_width
            current_roi_config.max_height = play_height
            current_roi_config.min_height = 10

            current_roi_config.stop_resizing_on_dir_change = False
            current_roi_config.stop_translation_on_dir_change = False
            current_roi_config.arena_box = to_bbox(self.get_arena_box(), self._cpp_boxes)

            #
            # Create and configure `AllLivingBoxConfig` for `_current_roi_aspect`
            #
            current_roi_aspect_config = AllLivingBoxConfig()

            current_roi_aspect_config.max_speed_x = (
                self._hockey_mom._camera_box_max_speed_x * 1 / speed_scale
            )
            current_roi_aspect_config.max_speed_y = (
                self._hockey_mom._camera_box_max_speed_y * 1 / speed_scale
            )
            current_roi_aspect_config.max_accel_x = (
                self._hockey_mom._camera_box_max_accel_x / speed_scale
            )
            current_roi_aspect_config.max_accel_y = (
                self._hockey_mom._camera_box_max_accel_y / speed_scale
            )

            current_roi_aspect_config.max_speed_w = (
                self._hockey_mom._camera_box_max_speed_x * 1 / speed_scale / 1.8
            )
            current_roi_aspect_config.max_speed_h = (
                self._hockey_mom._camera_box_max_speed_y * 1 / speed_scale / 1.8
            )
            current_roi_aspect_config.max_accel_w = (
                self._hockey_mom._camera_box_max_accel_x / speed_scale
            )
            current_roi_aspect_config.max_accel_h = (
                self._hockey_mom._camera_box_max_accel_y / speed_scale
            )

            current_roi_aspect_config.max_width = play_width
            current_roi_aspect_config.max_height = play_height
            current_roi_aspect_config.min_height = play_height / 5
            current_roi_config.stop_resizing_on_dir_change = True
            current_roi_config.stop_translation_on_dir_change = True
            current_roi_aspect_config.sticky_translation = True
            current_roi_aspect_config.arena_box = to_bbox(self.get_arena_box(), self._cpp_boxes)
            current_roi_aspect_config.sticky_size_ratio_to_frame_width = args.game_config["rink"][
                "camera"
            ]["sticky_size_ratio_to_frame_width"]
            current_roi_aspect_config.sticky_translation_gaussian_mult = args.game_config["rink"][
                "camera"
            ]["sticky_translation_gaussian_mult"]
            current_roi_aspect_config.unsticky_translation_size_ratio = args.game_config["rink"][
                "camera"
            ]["unsticky_translation_size_ratio"]
            current_roi_aspect_config.sticky_sizing = True
            current_roi_aspect_config.scale_dest_width = args.game_config["rink"]["camera"][
                "follower_box_scale_width"
            ]
            current_roi_aspect_config.scale_dest_height = args.game_config["rink"]["camera"][
                "follower_box_scale_height"
            ]
            current_roi_aspect_config.fixed_aspect_ratio = self._final_aspect_ratio

            if not self._cpp_playtracker:
                self._breakaway_detection = BreakawayDetection(args.game_config)
                #
                # Initialize `_current_roi` MovingBox with `current_roi_config`
                #
                self._current_roi: Union[MovingBox, PyLivingBox] = PyLivingBox(
                    "Current ROI",
                    to_bbox(start_box, self._cpp_boxes),
                    current_roi_config,
                    color=(255, 128, 64),
                    thickness=5,
                )

                # Initialize `_current_roi_aspect` MovingBox with `current_roi_aspect_config`
                self._current_roi_aspect: Union[MovingBox, PyLivingBox] = PyLivingBox(
                    "AspectRatio",
                    to_bbox(start_box, self._cpp_boxes),
                    current_roi_aspect_config,
                    color=(255, 0, 255),
                    thickness=5,
                )
            else:
                pt_config = PlayTrackerConfig()
                pt_config.living_boxes = [
                    current_roi_config,
                    current_roi_aspect_config,
                ]
                pt_config.ignore_largest_bbox = self._args.cam_ignore_largest
                pt_config.no_wide_start = self._args.no_wide_start

                breakaway_detection = get_nested_value(
                    args.game_config, "rink.camera.breakaway_detection", None
                )
                pt_config.play_detector.min_considered_group_velocity = breakaway_detection[
                    "min_considered_group_velocity"
                ]
                pt_config.play_detector.group_ratio_threshold = breakaway_detection[
                    "group_ratio_threshold"
                ]
                pt_config.play_detector.group_velocity_speed_ratio = breakaway_detection[
                    "group_velocity_speed_ratio"
                ]
                pt_config.play_detector.scale_speed_constraints = breakaway_detection[
                    "scale_speed_constraints"
                ]
                pt_config.play_detector.nonstop_delay_count = breakaway_detection[
                    "nonstop_delay_count"
                ]
                pt_config.play_detector.overshoot_scale_speed_ratio = breakaway_detection[
                    "overshoot_scale_speed_ratio"
                ]
                self._breakaway_detection = None

                self._playtracker = CppPlayTracker(BBox(0, 0, play_width, play_height), pt_config)
                self._current_roi = None
                self._current_roi_aspect = None
        else:
            assert not self._cpp_playtracker

            self._breakaway_detection = BreakawayDetection(args.game_config)

            self._current_roi: Union[MovingBox, PyLivingBox] = MovingBox(
                label="Current ROI",
                bbox=start_box.clone(),
                arena_box=self.get_arena_box(),
                max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1.5 / speed_scale,
                max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1.5 / speed_scale,
                max_accel_x=self._hockey_mom._camera_box_max_accel_x * 1.1 / speed_scale,
                max_accel_y=self._hockey_mom._camera_box_max_accel_y * 1.1 / speed_scale,
                max_width=play_width,
                max_height=play_height,
                stop_on_dir_change=False,
                color=(255, 128, 64),
                thickness=5,
                device=self._device,
            )

            self._current_roi_aspect: Union[MovingBox, PyLivingBox] = MovingBox(
                label="AspectRatio",
                bbox=start_box.clone(),
                arena_box=self.get_arena_box(),
                max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1 / speed_scale,
                max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1 / speed_scale,
                max_accel_x=self._hockey_mom._camera_box_max_accel_x.clone() / speed_scale,
                max_accel_y=self._hockey_mom._camera_box_max_accel_y.clone() / speed_scale,
                max_width=play_width,
                max_height=play_height,
                stop_on_dir_change=True,
                sticky_translation=True,
                sticky_size_ratio_to_frame_width=self._args.game_config["rink"]["camera"][
                    "sticky_size_ratio_to_frame_width"
                ],
                sticky_translation_gaussian_mult=self._args.game_config["rink"]["camera"][
                    "sticky_translation_gaussian_mult"
                ],
                unsticky_translation_size_ratio=self._args.game_config["rink"]["camera"][
                    "unsticky_translation_size_ratio"
                ],
                sticky_sizing=True,
                scale_width=self._args.game_config["rink"]["camera"]["follower_box_scale_width"],
                scale_height=self._args.game_config["rink"]["camera"]["follower_box_scale_height"],
                fixed_aspect_ratio=self._final_aspect_ratio,
                color=(255, 0, 255),
                thickness=5,
                device=self._device,
                min_height=play_height / 5,
            )

    _INFO_IMGS_FRAME_ID_INDEX = 2

    @property
    def play_box(self) -> torch.Tensor:
        return self._play_box.clone()

    def train(self, mode: bool = True):
        if isinstance(self._current_roi, torch.nn.Module):
            self._current_roi.train(mode)
            self._current_roi_aspect.train(mode)
        return super().train(mode)

    def get_arena_box(self):
        return self._play_box.clone()

    def _kmeans_cuda_device(self):
        return "cpu"

    def set_initial_tracking_box(self, box: torch.Tensor):
        """
        Set the initial tracking boxes
        """
        assert self._frame_counter <= 1, "Not currently meant for setting at runtime"
        frame_box = self.get_arena_box()
        fw, fh = width(frame_box), height(frame_box)
        # Should fit in the video frame
        # assert width(box) <= fw and height(box) <= fh
        scale_w, scale_h = self._current_roi.get_size_scale()
        box_roi = clamp_box(
            box=scale_box(box, scale_width=scale_w, scale_height=scale_h),
            clamp_box=frame_box,
        )

        # We set the roi box to be this exact size
        self._current_roi.set_bbox(to_bbox(box_roi, self._cpp_boxes))
        # Then we scale up as needed for the aspect roi
        scale_w, scale_h = self._current_roi_aspect.get_size_scale()
        box_roi = clamp_box(
            box=scale_box(box, scale_width=scale_w, scale_height=scale_h),
            clamp_box=frame_box,
        )
        self._current_roi_aspect.set_bbox(to_bbox(box_roi, self._cpp_boxes))

    def get_cluster_boxes(
        self,
        online_tlwhs: torch.Tensor,
        online_ids: torch.Tensor,
        cluster_counts: List[int],
        centroids: Optional[torch.Tensor] = None,
    ):
        if self._cluster_man is None:

            if False and centroids is None:
                # DEBUGGING ONLY
                centroids = torch.tensor(
                    [[1607.35034, 391.35437], [2095.76343, 359.94247], [2361.51660, 388.40149]],
                    dtype=torch.float,
                )

            self._cluster_man = ClusterMan(
                sizes=cluster_counts,
                device=self._kmeans_cuda_device(),
                centroids=centroids,
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

    def process_jerseys_info(self, frame_index: int, frame_id: int, data: Dict[str, Any]) -> None:
        jersey_results = data.get("jersey_results")
        if not jersey_results:
            return
        jersey_results = jersey_results[frame_index]
        if not jersey_results:
            return
        for current_info in jersey_results:
            self._jersey_tracker.observe_tracking_id_number_info(
                frame_id=frame_id, info=current_info
            )

    # @torch.jit.script
    def forward(self, results: Dict[str, Any]):
        track_data_sample = results["data_samples"]

        original_images = results.pop("original_images")

        if isinstance(original_images, StreamTensor):
            original_images._verbose = True
            original_images = original_images.get()

        # Figure out what device this image should be on
        image_device = self._device
        if image_device.type != "cuda" and original_images.device.type == "cuda":
            # prefer the image's cuda device
            image_device = original_images.device

        if original_images.device != image_device:
            original_images = original_images.to(device=image_device, non_blocking=True)

        frame_ids_list: List[torch.Tensor] = []
        current_box_list: List[torch.Tensor] = []
        current_fast_box_list: List[torch.Tensor] = []
        online_images: List[torch.Tensor] = []

        # Make the original images into a list so that we can release the batched one
        original_images_list: List[Optional[torch.Tensor]] = []
        for i in range(original_images.shape[0]):
            original_images_list.append(original_images[i])
        del original_images

        for frame_index, video_data_sample in enumerate(track_data_sample.video_data_samples):
            scalar_frame_id = video_data_sample.frame_id
            frame_id = torch.tensor([scalar_frame_id], dtype=torch.int64)
            online_tlwhs = batch_tlbrs_to_tlwhs(video_data_sample.pred_track_instances.bboxes)
            online_ids = video_data_sample.pred_track_instances.instances_id

            if True:
                # goes a few fps faster when async if this is on CPU
                frame_id = frame_id.cpu()
                online_tlwhs = online_tlwhs.cpu()
                online_ids = online_ids.cpu()

            self.process_jerseys_info(
                frame_index=frame_index, frame_id=scalar_frame_id, data=results
            )

            self._frame_counter += 1

            online_im = original_images_list[frame_index]
            # deref the original one to free the cuda memory
            original_images_list[frame_index] = None

            #
            # Play
            #
            cluster_counts = [3, 2]

            vis_ignored_tracking_ids: Union[Set[int], None] = None

            if self._playtracker is not None:
                online_bboxes = [BBox(*b) for b in video_data_sample.pred_track_instances.bboxes]
                playtracker_results = self._playtracker.forward(
                    online_ids.cpu().tolist(), online_bboxes
                )

                if playtracker_results.largest_tracking_bbox_id is not None:
                    largest_bbox = from_bbox(playtracker_results.largest_tracking_bbox)
                    largest_bbox = batch_tlbrs_to_tlwhs(largest_bbox.unsqueeze(0)).squeeze(0)
                    vis_ignored_tracking_ids = {playtracker_results.largest_tracking_bbox_id}
                else:
                    largest_bbox = None

                cluster_enclosing_box = from_bbox(playtracker_results.final_cluster_box)
                cluster_boxes_map = playtracker_results.cluster_boxes
                # cluster_boxes = [cluster_enclosing_box, cluster_enclosing_box]

                fast_roi_bounding_box = from_bbox(playtracker_results.tracking_boxes[0])
                current_fast_box_list.append(fast_roi_bounding_box)

                current_box = from_bbox(playtracker_results.tracking_boxes[1])
                current_box_list.append(current_box)

                if self._args.plot_moving_boxes:
                    # Play box
                    if (
                        torch.sum(self._play_box == self._hockey_mom._video_frame.bounding_box())
                        != 4
                    ):
                        online_im = vis.draw_dashed_rectangle(
                            img=online_im,
                            box=self._play_box,
                            color=(255, 0, 0),
                            thickness=2,
                        )

                    # Fast
                    online_im = PyLivingBox.draw_impl(
                        live_box=self._playtracker.get_live_box(0),
                        img=online_im,
                        color=(255, 128, 64),
                        thickness=5,
                    )
                    # Following
                    online_im = PyLivingBox.draw_impl(
                        live_box=self._playtracker.get_live_box(1),
                        img=online_im,
                        draw_thresholds=True,
                        following_box=self._playtracker.get_live_box(0),
                        color=(255, 0, 255),
                        thickness=5,
                    )
                    online_im = vis.plot_line(
                        online_im,
                        center(fast_roi_bounding_box),
                        center(current_box),
                        color=(255, 255, 255),
                        thickness=2,
                    )

                if (
                    self._args.plot_individual_player_tracking
                    and playtracker_results.play_detection is not None
                    and playtracker_results.play_detection.breakaway_edge_center is not None
                ):
                    """
                    When detecting a breakaway, draw a circle on the player
                    that represents the forward edge of the breakaway players
                    (person out in front the most, although this may be a defenseman
                    backing up, for instance)
                    """
                    edge_center = playtracker_results.play_detection.breakaway_edge_center
                    edge_center = [edge_center.x, edge_center.y]
                    online_im = vis.plot_circle(
                        online_im,
                        edge_center,
                        radius=30,
                        color=(255, 0, 255),
                        thickness=20,
                    )
                    online_im = vis.plot_line(
                        online_im,
                        edge_center,
                        center(fast_roi_bounding_box),
                        color=(128, 255, 128),
                        thickness=4,
                    )

            else:
                largest_bbox = None
                if self._args.cam_ignore_largest and len(online_tlwhs):
                    # Don't remove unless we have at least 4 online items being tracked
                    online_tlwhs, mask, largest_bbox = remove_largest_bbox(
                        online_tlwhs, min_boxes=4
                    )
                    online_ids = online_ids[mask]

                self._hockey_mom.append_online_objects(online_ids, online_tlwhs)

                #
                # BEGIN Clusters and Cluster Boxes
                #
                cluster_boxes_map, cluster_boxes = self.get_cluster_boxes(
                    online_tlwhs, online_ids, cluster_counts=cluster_counts
                )

                if cluster_boxes_map:
                    cluster_enclosing_box = get_enclosing_box(cluster_boxes)
                elif self._previous_cluster_union_box is not None:
                    cluster_enclosing_box = self._previous_cluster_union_box.clone()
                else:
                    cluster_enclosing_box = self.get_arena_box()

                current_box = cluster_enclosing_box

            if self._args.plot_boundaries and self._boundaries is not None:
                online_im = self._boundaries.draw(online_im)

            if self._args.plot_all_detections is not None:
                detections = video_data_sample.pred_instances.bboxes
                if not isinstance(detections, dict):
                    for detection, score in zip(
                        detections, video_data_sample.pred_instances.scores
                    ):
                        if score >= self._args.plot_all_detections:
                            online_im = vis.plot_rectangle(
                                img=online_im,
                                box=detection,
                                color=(64, 64, 64),
                                thickness=1,
                            )
                            if score < 0.7:
                                online_im = vis.plot_text(
                                    online_im,
                                    format(float(score), ".2f"),
                                    (
                                        int(detection[0] + width(detection) / 2),
                                        int(detection[1]),
                                    ),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1,
                                    (255, 255, 255),
                                    thickness=1,
                                )

                # Maybe draw trajectories...
                if self._args.plot_trajectories:
                    for tid in online_ids:
                        hist = self._hockey_mom.get_history(tid)
                        if hist is not None:
                            online_img = hist.draw(online_im)

            if self._args.plot_individual_player_tracking:
                online_im = vis.plot_tracking(
                    online_im,
                    online_tlwhs,
                    online_ids,
                    frame_id=frame_id,
                    speeds=[],
                    line_thickness=2,
                    ignore_tracking_ids=vis_ignored_tracking_ids,
                    ignored_color=(0, 0, 0),
                )
                # logger.info(f"Tracking {len(online_ids)} players...")
                if largest_bbox is not None and vis_ignored_tracking_ids is None:
                    online_im = vis.plot_rectangle(
                        online_im,
                        [int(i) for i in tlwh_to_tlbr_single(largest_bbox)],
                        color=(0, 0, 0),
                        thickness=1,
                        label="IGNORED",
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
                            from_bbox(cluster_boxes_map[cc]),
                            color=cluster_box_colors[cc],
                            thickness=1,
                            label=f"cluster_box_{cc}",
                        )

                if cluster_boxes_map:
                    if len(cluster_boxes_map) == 1 and 0 in cluster_boxes_map:
                        color = (255, 0, 0)
                    else:
                        color = (64, 64, 64)  # dark gray
                    # The union of the two cluster boxes
                    online_im = vis.plot_alpha_rectangle(
                        online_im,
                        from_bbox(cluster_enclosing_box),
                        color=color,
                        label="union_clusters",
                        opacity_percent=25,
                    )
            #
            # END Clusters and Cluster Boxes
            #

            online_im = self._jersey_tracker.draw(
                image=online_im, tracking_ids=online_ids, bboxes=online_tlwhs
            )

            if self._playtracker is None:
                current_box, online_im = self.calculate_breakaway(
                    current_box=current_box,
                    online_im=online_im,
                    speed_adjust_box=self._current_roi,
                    average_current_box=True,
                )

                #
                # Backup the last calculated box
                #
                self._previous_cluster_union_box = current_box.clone()

                # Some players may be off-screen, so their box may go over an edge
                current_box = clamp_box(current_box, self._play_box)

                # Maybe set initial box sizes if we aren't starting with a wide frame
                if self._frame_counter == 1 and self._args.no_wide_start:
                    self.set_initial_tracking_box(current_box)

                #
                # Apply the new calculated play
                #
                fast_roi_bounding_box = self._current_roi.forward(
                    to_bbox(current_box, self._cpp_boxes)
                )

                current_box = self._current_roi_aspect.forward(fast_roi_bounding_box)

                fast_roi_bounding_box = from_bbox(fast_roi_bounding_box)
                current_box = from_bbox(current_box)

                current_box_list.append(from_bbox(self._current_roi_aspect.bounding_box()))
                current_fast_box_list.append(from_bbox(self._current_roi.bounding_box()))

                if self._args.plot_moving_boxes:
                    online_im = self._current_roi_aspect.draw(
                        img=online_im,
                        draw_thresholds=True,
                        following_box=self._current_roi,
                    )
                    online_im = self._current_roi.draw(img=online_im)
                    online_im = vis.plot_line(
                        online_im,
                        center(fast_roi_bounding_box),
                        center(current_box),
                        color=(255, 255, 255),
                        thickness=2,
                    )

            if self._args.plot_speed:
                vis.plot_frame_id_and_speeds(
                    online_im,
                    frame_id,
                    *self._hockey_mom.get_velocity_and_acceleratrion_xy(),
                )

            frame_ids_list.append(frame_id)
            # current_box_list.append(self._current_roi_aspect.bounding_box().clone().cpu())
            # current_fast_box_list.append(self._current_roi.bounding_box().clone().cpu())
            if isinstance(online_im, np.ndarray):
                online_im = torch.from_numpy(online_im).to(device=image_device, non_blocking=True)
                if online_im.ndim == 4 and online_im.shape[0] == 1:
                    online_im = online_im.squeeze(0)
            assert online_im.device == image_device
            online_images.append(make_channels_last(online_im))

        results["frame_ids"] = torch.stack(frame_ids_list)
        results["current_box"] = torch.stack(current_box_list)
        results["current_fast_box_list"] = torch.stack(current_fast_box_list)

        # We want to track if it's slow
        img = torch.stack(online_images)
        img = StreamCheckpoint(img)
        img._verbose = True
        results["img"] = img

        return results

    def calculate_breakaway(
        self,
        current_box: torch.Tensor,
        online_im: Union[torch.Tensor, np.ndarray],
        speed_adjust_box: MovingBox,
        average_current_box: bool,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, np.ndarray]]:
        #
        # BEGIN Breakway detection
        #
        group_x_velocity, edge_center = self._hockey_mom.get_group_x_velocity(
            min_considered_velocity=self._breakaway_detection.min_considered_group_velocity,
            group_threshhold=self._breakaway_detection.group_ratio_threshold,
        )
        if group_x_velocity:

            # if True:
            #     return current_box, online_im

            if self._args.plot_individual_player_tracking:
                """
                When detecting a breakaway, draw a circle on the player
                that represents the forward edge of the breakaway players
                (person out in front the most, although this may be a defenseman
                backing up, for instance)
                """
                online_im = vis.plot_circle(
                    online_im,
                    edge_center,
                    radius=30,
                    color=(255, 0, 255),
                    thickness=20,
                )
            edge_center = torch.tensor(edge_center, dtype=torch.float, device=current_box.device)

            if average_current_box:
                average_center = (edge_center + center(current_box)) / 2.0
                current_box = make_box_at_center(
                    average_center, width(current_box), height(current_box)
                )
            else:
                current_box = make_box_at_center(
                    edge_center, width(current_box), height(current_box)
                )

            # If group x velocity is in different direction than current speed, behave a little differently
            if speed_adjust_box is not None:
                speed_adjust_bbox = from_bbox(speed_adjust_box.bounding_box())
                roi_center = center(speed_adjust_bbox)
                if self._args.plot_individual_player_tracking:
                    online_im = vis.plot_line(
                        online_im,
                        edge_center,
                        roi_center,
                        color=(128, 255, 128),
                        thickness=4,
                    )
                # Previous way
                should_adjust_speed = torch.logical_or(
                    torch.logical_and(group_x_velocity > 0, roi_center[0] < edge_center[0]),
                    torch.logical_and(group_x_velocity < 0, roi_center[0] > edge_center[0]),
                )
                if should_adjust_speed.item():
                    group_x_velocity = group_x_velocity.cpu()
                    if isinstance(speed_adjust_box, PyLivingBox):
                        group_x_velocity = group_x_velocity.item()
                    speed_adjust_box.adjust_speed(
                        accel_x=group_x_velocity
                        * self._breakaway_detection.group_velocity_speed_ratio,
                        accel_y=None,
                        scale_constraints=self._breakaway_detection.scale_speed_constraints,
                        nonstop_delay=self._breakaway_detection.nonstop_delay_count,
                    )
                else:
                    # Cut the speed quickly due to overshoot
                    # self._current_roi.scale_speed(ratio_x=0.6)
                    speed_adjust_box.scale_speed(
                        ratio_x=self._breakaway_detection.overshoot_scale_speed_ratio
                    )
        #
        # END Breakway detection
        #
        return current_box, online_im
