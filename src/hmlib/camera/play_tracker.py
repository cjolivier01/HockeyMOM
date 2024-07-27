from __future__ import absolute_import, division, print_function

import argparse
from typing import Dict, List, Optional, Set

import cv2
import torch

from hmlib.builder import HM, PIPELINES
from hmlib.camera.clusters import ClusterMan
from hmlib.camera.moving_box import MovingBox
from hmlib.config import get_nested_value
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.tracking_utils.log import logger
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


# def scale_box(box, from_img, to_img):
#     from_sz = (from_img.shape[1], from_img.shape[0])
#     to_sz = (to_img.shape[1], to_img.shape[0])
#     w_scale = to_sz[1] / from_sz[1]
#     h_scale = to_sz[0] / from_sz[0]
#     new_box = [box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale]
#     logger.info(f"from={box} -> to={new_box}")
#     return new_box


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
                # logger.info(f"ignoring: {center}")
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


@HM.register_module()
class PlayTracker(torch.nn.Module):
    def __init__(
        self,
        hockey_mom,
        device,
        original_clip_box,
        args: argparse.Namespace,
    ):
        super(PlayTracker, self).__init__()
        self._args = args
        self._hockey_mom = hockey_mom
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

        # Tracking specific ids
        self._track_ids: Set[int] = set()
        if args.track_ids:
            self._track_ids = set([int(i) for i in args.track_ids.split(",")])

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
        self._frame_counter: int = 0

        start_box = self._hockey_mom._video_frame.bounding_box()
        self._current_roi = MovingBox(
            label="Current ROI",
            bbox=start_box.clone(),
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
            bbox=start_box.clone(),
            arena_box=self.get_arena_box(),
            max_speed_x=self._hockey_mom._camera_box_max_speed_x * 1,
            max_speed_y=self._hockey_mom._camera_box_max_speed_y * 1,
            max_accel_x=self._hockey_mom._camera_box_max_accel_x.clone(),
            max_accel_y=self._hockey_mom._camera_box_max_accel_y.clone(),
            max_width=self._hockey_mom._video_frame.width,
            max_height=self._hockey_mom._video_frame.height,
            width_change_threshold=_scalar_like(
                size_unstick_size * 2, device=self._device
            ),
            width_change_threshold_low=_scalar_like(
                size_stick_size * 2, device=self._device
            ),
            height_change_threshold=_scalar_like(
                size_unstick_size * 2, device=self._device
            ),
            height_change_threshold_low=_scalar_like(
                size_stick_size * 2, device=self._device
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
            min_height=self._hockey_mom._video_frame.height / 5,
        )

    _INFO_IMGS_FRAME_ID_INDEX = 2

    def get_arena_box(self):
        return self._hockey_mom._video_frame.bounding_box()

    def _kmeans_cuda_device(self):
        return "cpu"

    def set_initial_tracking_box(self, box: torch.Tensor):
        """
        Set the initial tracking boxes
        """
        assert self._frame_counter <= 1, "Not currently meant for setting at runtime"
        frame_box = start_box = self._hockey_mom._video_frame.bounding_box()
        fw, fh = width(frame_box), height(frame_box)
        # Should fit in the video frame
        assert width(box) <= fw and height(box) <= fh
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
        tracking_ids: Set[int],  # Track specific id's
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

        have_tracking_ids_boxes_map: Dict[int, torch.Tensor] = dict()

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
                if len(tracking_ids):
                    tracking_ids_in_this_cluster = (
                        self._cluster_man.prune_not_in_largest_cluster(
                            num_clusters=cluster_count, ids=tracking_ids
                        )
                    )
                    if len(tracking_ids_in_this_cluster) != 0:
                        if len(tracking_ids_in_this_cluster) > 1:
                            logger.warn(
                                f"Warning: Found more than one tracking ID in a single frame: {tracking_ids_in_this_cluster}"
                            )
                        have_tracking_ids_boxes_map[cluster_count] = (
                            largest_cluster_ids_box
                        )
            else:
                largest_cluster_ids_box = None
        if not boxes_map:
            return {}, None
        if have_tracking_ids_boxes_map:
            values = [v for v in have_tracking_ids_boxes_map.values()]
            boxes_list: torch.Tensor = torch.stack(values)
            widths: torch.Tensor = boxes_list[:, 3] - boxes_list[:, 1]
            heights: torch.Tensor = boxes_list[:, 2] - boxes_list[:, 0]
            areas = widths * heights
            min_area_index = torch.argmin(areas)
            boxes_list = boxes_list[min_area_index].unsqueeze(0)
            boxes_map = {0: boxes_list[0]}
        else:
            boxes_list: torch.Tensor = torch.stack(boxes_list)

        return boxes_map, boxes_list

    def forward(self, online_targets_and_img):
        self._timer.tic()

        online_tlwhs = online_targets_and_img[0]
        online_ids = online_targets_and_img[1]
        detections = online_targets_and_img[2]
        info_imgs = online_targets_and_img[3]

        frame_ids = info_imgs[self._INFO_IMGS_FRAME_ID_INDEX]
        frame_id = frame_ids[self._frame_counter % len(frame_ids)]
        self._frame_counter += 1

        largest_bbox = None

        if self._args.cam_ignore_largest and len(online_tlwhs):
            # Don't remove unless we have at least 4 online items being tracked
            online_tlwhs, mask, largest_bbox = remove_largest_bbox(
                online_tlwhs, min_boxes=4
            )
            online_ids = online_ids[mask]

        original_img = online_targets_and_img[5]

        online_im = original_img
        if online_im.ndim == 4:
            assert online_im.shape[0] == 1
            online_im = online_im.squeeze(0)

        self._hockey_mom.append_online_objects(online_ids, online_tlwhs)

        #
        # Clusters
        #
        cluster_counts = [3, 2]
        cluster_boxes_map, cluster_boxes = self.get_cluster_boxes(
            online_tlwhs,
            online_ids,
            cluster_counts=cluster_counts,
            tracking_ids=(
                torch.tensor(list(self._track_ids), dtype=torch.int32)
                if self._track_ids
                else []
            ),
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

        if self._args.plot_all_detections is not None:
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

        if current_box is None:
            assert False  # how does this happen?
            current_box = self._hockey_mom._video_frame.bounding_box()

        self._previous_cluster_union_box = current_box.clone()

        # Some players may be off-screen, so their box may go over an edge
        current_box = self._hockey_mom.clamp(current_box)

        # Maybe set initial box sizes if we aren't starting with a wide frame
        if self._frame_counter == 1 and self._args.no_wide_start:
            self.set_initial_tracking_box(current_box)

        fast_roi_bounding_box = self._current_roi(current_box, stop_on_dir_change=False)
        current_box = self._current_roi_aspect(
            fast_roi_bounding_box, stop_on_dir_change=True
        )

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

        group_x_velocity, edge_center = self._hockey_mom.get_group_x_velocity(
            min_considered_velocity=self._breakaway_detection.min_considered_group_velocity,
            group_threshhold=self._breakaway_detection.group_ratio_threshold,
        )
        #
        # Breakway detection
        #
        if group_x_velocity:
            # logger.info(f"frame {frame_id} group x velocity: {group_x_velocity}")
            if self._args.plot_individual_player_tracking:
                cv2.circle(
                    online_im,
                    [int(i) for i in edge_center],
                    radius=30,
                    color=(255, 0, 255),
                    thickness=20,
                )
            edge_center = torch.tensor(
                edge_center, dtype=torch.float, device=current_box.device
            )
            current_box = make_box_at_center(
                edge_center, width(current_box), height(current_box)
            )

            # If group x velocity is in different direction than current speed, behave a little differently
            if frame_id == 7500:
                pass
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
                        group_x_velocity > 0,
                        roi_center[0] < edge_center[0],
                    ),
                    torch.logical_and(
                        group_x_velocity < 0,
                        roi_center[0] > edge_center[0],
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
                    should_cut_speed = torch.logical_or(
                        torch.logical_and(
                            self._current_roi._current_speed_x < 0,
                            roi_center[0] < edge_center[0],
                        ),
                        torch.logical_and(
                            self._current_roi._current_speed_x > 0,
                            roi_center[0] > edge_center[0],
                        ),
                    )
                    # Cut the speed quickly due to overshoot
                    # self._current_roi.scale_speed(ratio_x=0.6)
                    if should_cut_speed:
                        self._current_roi.scale_speed(
                            ratio_x=self._breakaway_detection.overshoot_scale_speed_ratio,
                            clamp_to_max=True,
                        )
                        logger.info(
                            f"Reducing group x velocity due to overshoot: group_x_velocity={group_x_velocity}, "
                            f"current_speed_x={self._current_roi._current_speed_x}"
                        )

        return frame_id, online_im, self._current_roi_aspect.bounding_box()


def _scalar_like(v, device):
    if isinstance(v, torch.Tensor):
        return v.clone()
    return torch.tensor(v, dtype=torch.float, device=device)
