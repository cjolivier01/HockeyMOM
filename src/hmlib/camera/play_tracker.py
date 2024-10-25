from __future__ import absolute_import, division, print_function

import argparse
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.builder import HM, PIPELINES
from hmlib.camera.camera import HockeyMOM
from hmlib.camera.clusters import ClusterMan
from hmlib.camera.moving_box import MovingBox
from hmlib.config import get_nested_value
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.box_functions import (
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
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_channels_last
from hmlib.utils.progress_bar import ProgressBar


def batch_tlbrs_to_tlwhs(tlbrs: torch.Tensor) -> torch.Tensor:
    tlwhs = tlbrs.clone()
    # make boxes tlwh
    tlwhs[:, 2] = tlwhs[:, 2] - tlwhs[:, 0]  # width = x2 - x1
    tlwhs[:, 3] = tlwhs[:, 3] - tlwhs[:, 1]  # height = y2 - y1
    return tlwhs


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
    for i, this_center in enumerate(online_tlwhs_centers):
        if inclusion_box is not None:
            if inclusion_box[0] and this_center[0] < inclusion_box[0]:
                continue
            elif inclusion_box[2] and this_center[0] > inclusion_box[2]:
                continue
            elif inclusion_box[1] and this_center[1] < inclusion_box[1]:
                continue
            elif inclusion_box[3] and this_center[1] > inclusion_box[3]:
                continue
        if boundaries is not None:
            # TODO: boundaries could be done with the box edges
            if boundaries.is_point_outside(this_center):
                # logger.info(f"ignoring: {this_center}")
                continue
        filtered_online_tlwh.append(online_tlwhs[i])
        filtered_online_ids.append(online_ids[i])
    if len(filtered_online_tlwh) == 0:
        assert len(filtered_online_ids) == 0
        return [], []
    return torch.stack(filtered_online_tlwh), torch.stack(filtered_online_ids)


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
    ):
        """
        Track the play

        :param hockey_mom: The old HockeyMom object
        :param play_box: The box allowed for play (assumed the visual play does not exist outside of this box)
        :param device: Device to use for computations
        :param original_clip_box: Clip box that has been applied to the original image (if any)
        :param progress_bar: Progress bar
        :param args: _description_
        """
        super(PlayTracker, self).__init__()
        self._args = args
        self._hockey_mom: HockeyMOM = hockey_mom
        # Amount to scale speed-related calculations based upon non-standard fps
        self._play_box = play_box
        self._thread = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._output_video = None
        self._final_image_processing_started = False
        self._device = device
        self._horizontal_image_gaussian_distribution = None
        self._boundaries = None
        self._timer = Timer()
        self._cluster_man: Optional[ClusterMan] = None
        self._original_clip_box = original_clip_box
        self._breakaway_detection = BreakawayDetection(args.game_config)
        self._progress_bar = progress_bar

        self._tracking_id_jersey: Dict[int, Tuple[int, float]] = {}

        # Tracking specific ids
        self._track_ids: Set[int] = set()
        if args.track_ids:
            self._track_ids = set([int(i) for i in args.track_ids.split(",")])

        if (
            self._args.plot_boundaries
            and self._args.top_border_lines
            or self._args.bottom_border_lines
        ):
            # Only used for plotting the lines
            self._boundaries = BoundaryLines(
                self._args.top_border_lines,
                self._args.bottom_border_lines,
                self._original_clip_box,
            )

        # Persistent state across frames
        self._previous_cluster_union_box = None
        self._last_temporal_box = None
        self._last_sticky_temporal_box = None
        self._frame_counter: int = 0

        play_width = width(self._play_box)
        play_height = height(self._play_box)

        assert width(self._play_box) == self._hockey_mom._video_frame.width
        assert height(self._play_box) == self._hockey_mom._video_frame.height

        # speed_scale = 1.0
        speed_scale = self._hockey_mom.fps_speed_scale

        start_box = self._play_box.clone()
        self._current_roi = MovingBox(
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

        self._current_roi_aspect = MovingBox(
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

    def train(self, mode: bool = True):
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
        self._current_roi.set_bbox(box_roi)
        # Then we scale up as needed for the aspect roi
        box_roi = self._current_roi.bounding_box()
        scale_w, scale_h = self._current_roi_aspect.get_size_scale()
        box_roi = clamp_box(
            box=scale_box(box, scale_width=scale_w, scale_height=scale_h),
            clamp_box=frame_box,
        )
        self._current_roi_aspect.set_bbox(box_roi)

    def get_cluster_boxes(
        self,
        online_tlwhs: torch.Tensor,
        online_ids: torch.Tensor,
        cluster_counts: List[int],
    ):
        if self._cluster_man is None:
            # self._cluster_man = ClusterMan(sizes=cluster_counts, device=online_tlwhs.device)
            self._cluster_man = ClusterMan(sizes=cluster_counts, device=self._kmeans_cuda_device())

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

    def process_jerseys_info(self, data: Dict[str, Any]) -> None:
        return None

        jersey_results = data["tracking_results"].get("jersey_results")
        if not jersey_results:
            return
        for tracking_id, (number, score) in jersey_results.items():
            jersey_info = self._tracking_id_jersey.get(tracking_id)
            if jersey_info is None:
                self._tracking_id_jersey[tracking_id] = (number, score)
            else:
                prev_number, prev_score = jersey_info
                # if number != prev_number:
                if number != prev_number and score > prev_score:
                    print(
                        f"Tracking ID change! trackig id {tracking_id} is changing from number {prev_number} to {number}"
                    )
                    self._tracking_id_jersey[tracking_id] = (number, score)

    def forward(self, results: Dict[str, Any]):
        self._timer.tic()

        track_data_sample = results["data_samples"]

        original_images = results.pop("original_images")

        if isinstance(original_images, StreamTensor):
            original_images.verbose = True
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

        for frame_index, video_data_sample in enumerate(track_data_sample.video_data_samples):
            frame_ids = torch.tensor([video_data_sample.frame_id], dtype=torch.int64)
            online_tlwhs = batch_tlbrs_to_tlwhs(video_data_sample.pred_track_instances.bboxes)
            online_ids = video_data_sample.pred_track_instances.instances_id
            # detections = batch_tlbrs_to_tlwhs(video_data_sample.pred_instances.bboxes)
            # online_tlwhs = results["online_tlwhs"]
            # online_ids = results["online_ids"]
            # detections = results["detections"]
            # info_imgs = results["info_imgs"]

            # online_tlwhs = online_tlwhs.cpu()
            # online_ids = online_ids.cpu()

            self.process_jerseys_info(data=results)

            # frame_ids = info_imgs[self._INFO_IMGS_FRAME_ID_INDEX]
            frame_id = frame_ids[self._frame_counter % len(frame_ids)]
            self._frame_counter += 1

            largest_bbox = None

            if self._args.cam_ignore_largest and len(online_tlwhs):
                # Don't remove unless we have at least 4 online items being tracked
                online_tlwhs, mask, largest_bbox = remove_largest_bbox(online_tlwhs, min_boxes=4)
                online_ids = online_ids[mask]

            online_im = original_images[frame_index]
            # if online_im.ndim == 4:
            #     assert online_im.shape[0] == 1
            #     online_im = online_im.squeeze(0)

            self._hockey_mom.append_online_objects(online_ids, online_tlwhs)

            #
            # BEGIN Clusters and Cluster Boxes
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
                cluster_enclosing_box = self.get_arena_box()

            current_box = cluster_enclosing_box

            if self._args.plot_boundaries and self._boundaries is not None:
                online_im = self._boundaries.draw(online_im)

            if self._args.plot_all_detections is not None:
                detections = batch_tlbrs_to_tlwhs(video_data_sample.pred_instances.bboxes)
                if not isinstance(detections, dict):
                    for detection in detections:
                        if detection[4] >= self._args.plot_all_detections:
                            online_im = vis.plot_rectangle(
                                img=online_im,
                                box=detection[:4],
                                color=(64, 64, 64),
                                thickness=1,
                            )
                            if detection[4] < 0.7:
                                cv2.putText(
                                    online_im,
                                    format(float(detection[4]), ".2f"),
                                    (
                                        int(detection[0] + width(detection[:4] / 2)),
                                        int(detection[1]),
                                    ),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1,
                                    (255, 255, 255),
                                    thickness=1,
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
                # logger.info(f"Tracking {len(online_ids)} players...")
                if largest_bbox is not None:
                    online_im = vis.plot_rectangle(
                        online_im,
                        [int(i) for i in tlwh_to_tlbr_single(largest_bbox)],
                        color=(0, 0, 0),
                        thickness=1,
                        label="IGNORED",
                    )

            if self._args.plot_jersey_numbers:
                online_im = vis.plot_jersey_numbers(
                    online_im,
                    online_tlwhs,
                    online_ids,
                    player_number_map=self._tracking_id_jersey,
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
                    if len(cluster_boxes_map) == 1 and 0 in cluster_boxes_map:
                        color = (255, 0, 0)
                    else:
                        color = (64, 64, 64)  # dark gray
                    # The union of the two cluster boxes
                    online_im = vis.plot_alpha_rectangle(
                        online_im,
                        cluster_enclosing_box,
                        color=color,
                        label="union_clusters",
                        opacity_percent=25,
                    )
            #
            # END Clusters and Cluster Boxes
            #

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
            # current_box = self._hockey_mom.clamp(current_box)
            current_box = clamp_box(current_box, self._play_box)

            # Maybe set initial box sizes if we aren't starting with a wide frame
            if self._frame_counter == 1 and self._args.no_wide_start:
                self.set_initial_tracking_box(current_box)

            #
            # Apply the new calculated play
            #
            fast_roi_bounding_box = self._current_roi(current_box)
            current_box = self._current_roi_aspect(fast_roi_bounding_box)

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

            # results["frame_id"] = frame_id
            # results["img"] = online_im
            # video_data_sample.set_metainfo({"img": online_im})
            # video_data_sample.set_metainfo({"frame_id": frame_id})
            # video_data_sample.set_metainfo(
            #     {"current_fast_box": self._current_roi.bounding_box().clone().cpu()}
            # )
            # video_data_sample.set_metainfo(
            #     {"current_box": self._current_roi_aspect.bounding_box().clone().cpu()}
            # )
            # results["current_fast_box"] = self._current_roi.bounding_box().clone()
            # results["current_box"] = self._current_roi_aspect.bounding_box().clone()

            frame_ids_list.append(frame_id)
            current_box_list.append(self._current_roi_aspect.bounding_box().clone().cpu())
            current_fast_box_list.append(self._current_roi.bounding_box().clone().cpu())
            if isinstance(online_im, np.ndarray):
                online_im = torch.from_numpy(online_im).to(device=image_device, non_blocking=True)
            assert online_im.device == image_device
            online_images.append(make_channels_last(online_im))

        results["frame_ids"] = torch.stack(frame_ids_list)
        results["current_box"] = torch.stack(current_box_list)
        results["current_fast_box_list"] = torch.stack(current_fast_box_list)
        results["img"] = torch.stack(online_images)

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
                roi_center = center(speed_adjust_box.bounding_box())
                if self._args.plot_individual_player_tracking:
                    vis.plot_line(
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
                    speed_adjust_box.adjust_speed(
                        accel_x=group_x_velocity.cpu()
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
                    speed_adjust_box.scale_speed(
                        ratio_x=self._breakaway_detection.overshoot_scale_speed_ratio
                    )
        #
        # END Breakway detection
        #
        return current_box, online_im


def _scalar_like(v, device):
    if isinstance(v, torch.Tensor):
        return v.clone()
    return torch.tensor(v, dtype=torch.float, device=device)
