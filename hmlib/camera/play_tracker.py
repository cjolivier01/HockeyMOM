from __future__ import absolute_import, division, print_function

import argparse
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.actions.action_tracker import ActionTracker, TrackingIdActionInfo
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
from hmlib.config import get_game_config_private, get_nested_value, save_private_config
from hmlib.constants import WIDTH_NORMALIZATION_SIZE
from hmlib.jersey.jersey_tracker import JerseyTracker
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis

# from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.utils.gpu import StreamCheckpoint, StreamTensor
from hmlib.utils.image import make_channels_last
from hmlib.utils.progress_bar import ProgressBar
from hockeymom.core import AllLivingBoxConfig, BBox
from hockeymom.core import PlayTracker as CppPlayTracker
from hockeymom.core import PlayTrackerConfig

from .camera_transformer import (
    CameraNorm,
    CameraPanZoomTransformer,
    build_frame_features,
    make_box_from_center_h,
    unpack_checkpoint,
)
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
        self._cluster_centroids: Optional[torch.Tensor] = None
        self._original_clip_box = original_clip_box
        self._progress_bar = progress_bar
        self._camera_ui_enabled = bool(getattr(args, "camera_ui", 0))
        self._ui_window_name = "Tracker Controls"
        self._ui_inited = False
        self._ui_color_window_name = "Tracker Controls (Color)"
        self._ui_color_inited = False
        self._ui_defaults = {}

        cluster_centroids = getattr(args, "cluster_centroids", None)
        if cluster_centroids is not None:
            centroids_tensor = torch.as_tensor(cluster_centroids, dtype=torch.float32).cpu()
            if centroids_tensor.ndim != 2 or centroids_tensor.shape[1] != 2:
                raise ValueError("cluster_centroids must be shaped (N, 2)")
            self._cluster_centroids = centroids_tensor

        self._jersey_tracker = JerseyTracker(show=args.plot_jersey_numbers)
        self._action_tracker = ActionTracker(show=getattr(args, 'plot_actions', False))
        # Cache for rink_profile (combined_mask, centroid, etc.) pulled from data samples meta
        self._rink_profile_cache = None

        # Tracking specific ids
        self._track_ids: Set[int] = set()
        if args.track_ids:
            self._track_ids = set([int(i) for i in args.track_ids.split(",")])

        # Persistent state across frames
        self._previous_cluster_union_box = None
        self._last_temporal_box = None
        self._last_sticky_temporal_box = None
        self._frame_counter: int = 0
        self._initial_box_applied: bool = False

        play_width = width(self._play_box)
        play_height = height(self._play_box)

        assert play_width <= self._hockey_mom._video_frame.width
        assert play_height <= self._hockey_mom._video_frame.height

        # speed_scale = 1.0
        speed_scale = self._hockey_mom.speed_scale

        start_box = self._play_box.clone()

        if self._cpp_boxes:

            # Create and configure `AllLivingBoxConfig` for `_current_roi`
            current_roi_config = AllLivingBoxConfig()

            # Translation
            current_roi_config.max_speed_x = self._hockey_mom._camera_box_max_speed_x * 1.5 / speed_scale
            current_roi_config.max_speed_y = self._hockey_mom._camera_box_max_speed_y * 1.5 / speed_scale
            current_roi_config.max_accel_x = self._hockey_mom._camera_box_max_accel_x * 1.1 / speed_scale
            current_roi_config.max_accel_y = self._hockey_mom._camera_box_max_accel_y * 1.1 / speed_scale

            # Resizing
            current_roi_config.max_speed_w = self._hockey_mom._camera_box_max_speed_x * 1.5 / speed_scale / 1.8
            current_roi_config.max_speed_h = self._hockey_mom._camera_box_max_speed_y * 1.5 / speed_scale / 1.8
            current_roi_config.max_accel_w = self._hockey_mom._camera_box_max_accel_x * 1.1 / speed_scale
            current_roi_config.max_accel_h = self._hockey_mom._camera_box_max_accel_y * 1.1 / speed_scale

            current_roi_config.max_width = play_width
            current_roi_config.max_height = play_height
            current_roi_config.min_height = play_height / 10

            current_roi_config.stop_resizing_on_dir_change = False
            current_roi_config.stop_translation_on_dir_change = False
            current_roi_config.arena_box = to_bbox(self.get_arena_box(), self._cpp_boxes)
            # Frames-to-destination speed limiting (scaled by fps)
            ttg_frames = int(args.game_config["rink"]["camera"].get("time_to_dest_speed_limit_frames", 10))
            current_roi_config.time_to_dest_speed_limit_frames = int(ttg_frames * self._hockey_mom.fps_speed_scale)

            #
            # Create and configure `AllLivingBoxConfig` for `_current_roi_aspect`
            #
            current_roi_aspect_config = AllLivingBoxConfig()

            kEXTRA_FOLLOWING_SCALE_DOWN = 1

            current_roi_aspect_config.max_speed_x = (
                self._hockey_mom._camera_box_max_speed_x * 1 / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )
            current_roi_aspect_config.max_speed_y = (
                self._hockey_mom._camera_box_max_speed_y * 1 / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )
            current_roi_aspect_config.max_accel_x = (
                self._hockey_mom._camera_box_max_accel_x / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )
            current_roi_aspect_config.max_accel_y = (
                self._hockey_mom._camera_box_max_accel_y / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )

            current_roi_aspect_config.max_speed_w = self._hockey_mom._camera_box_max_speed_x * 1 / speed_scale / 1.8
            current_roi_aspect_config.max_speed_h = self._hockey_mom._camera_box_max_speed_y * 1 / speed_scale / 1.8
            current_roi_aspect_config.max_accel_w = self._hockey_mom._camera_box_max_accel_x / speed_scale
            current_roi_aspect_config.max_accel_h = self._hockey_mom._camera_box_max_accel_y / speed_scale

            current_roi_aspect_config.max_width = play_width
            current_roi_aspect_config.max_height = play_height
            current_roi_aspect_config.min_height = play_height / 5

            current_roi_aspect_config.stop_resizing_on_dir_change = True
            current_roi_aspect_config.stop_translation_on_dir_change = True
            current_roi_aspect_config.sticky_translation = True
            # Prefer YAML; fall back to CLI hm_opts defaults/overrides
            camera_cfg = args.game_config["rink"]["camera"]
            stop_dir_delay = int(
                camera_cfg.get(
                    "stop_on_dir_change_delay",
                    getattr(args, "stop_on_dir_change_delay", 0),
                )
            )
            cancel_stop = bool(
                camera_cfg.get(
                    "cancel_stop_on_opposite_dir",
                    bool(getattr(args, "cancel_stop_on_opposite_dir", 0)),
                )
            )
            cancel_hyst = int(
                camera_cfg.get(
                    "stop_cancel_hysteresis_frames",
                    getattr(args, "stop_cancel_hysteresis_frames", 0),
                )
            )
            cooldown_frames = int(
                camera_cfg.get(
                    "stop_delay_cooldown_frames",
                    getattr(args, "stop_delay_cooldown_frames", 0),
                )
            )
            current_roi_aspect_config.stop_translation_on_dir_change_delay = stop_dir_delay
            current_roi_aspect_config.cancel_stop_on_opposite_dir = cancel_stop
            current_roi_aspect_config.cancel_stop_hysteresis_frames = cancel_hyst
            current_roi_aspect_config.stop_delay_cooldown_frames = cooldown_frames

            # FIXME: get this from config
            current_roi_aspect_config.dynamic_acceleration_scaling = 1.0
            current_roi_aspect_config.arena_angle_from_vertical = 30.0

            current_roi_aspect_config.arena_box = to_bbox(self.get_arena_box(), self._cpp_boxes)
            current_roi_aspect_config.sticky_size_ratio_to_frame_width = args.game_config["rink"]["camera"][
                "sticky_size_ratio_to_frame_width"
            ]
            current_roi_aspect_config.sticky_translation_gaussian_mult = args.game_config["rink"]["camera"][
                "sticky_translation_gaussian_mult"
            ]
            current_roi_aspect_config.unsticky_translation_size_ratio = args.game_config["rink"]["camera"][
                "unsticky_translation_size_ratio"
            ]
            current_roi_aspect_config.sticky_sizing = True
            current_roi_aspect_config.scale_dest_width = args.game_config["rink"]["camera"]["follower_box_scale_width"]
            current_roi_aspect_config.scale_dest_height = args.game_config["rink"]["camera"][
                "follower_box_scale_height"
            ]
            current_roi_aspect_config.fixed_aspect_ratio = self._final_aspect_ratio

            # If we are not using the C++ PlayTracker, or if the controller is the transformer,
            # initialize Python-side movers without creating the C++ PlayTracker.
            if (not self._cpp_playtracker) or (getattr(args, "camera_controller", None) == "transformer"):
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

                pt_config.ignore_outlier_players = True  # EXPERIMENTAL
                pt_config.ignore_left_and_right_extremes = False  # EXPERIMENTAL

                breakaway_detection = get_nested_value(args.game_config, "rink.camera.breakaway_detection", None)
                pt_config.play_detector.min_considered_group_velocity = breakaway_detection[
                    "min_considered_group_velocity"
                ]
                pt_config.play_detector.group_ratio_threshold = breakaway_detection["group_ratio_threshold"]
                pt_config.play_detector.group_velocity_speed_ratio = breakaway_detection["group_velocity_speed_ratio"]
                pt_config.play_detector.scale_speed_constraints = breakaway_detection["scale_speed_constraints"]
                pt_config.play_detector.nonstop_delay_count = breakaway_detection["nonstop_delay_count"]
                pt_config.play_detector.overshoot_scale_speed_ratio = breakaway_detection["overshoot_scale_speed_ratio"]
                # Optional: use stop-delay braking instead of multiplicative overshoot scale
                pt_config.play_detector.overshoot_stop_delay_count = int(
                    breakaway_detection.get("overshoot_stop_delay_count", 0)
                )
                # After nonstop ends, optionally brake to stop
                post_nonstop = int(breakaway_detection.get("post_nonstop_stop_delay_count", 0))
                current_roi_aspect_config.post_nonstop_stop_delay_count = post_nonstop
                self._breakaway_detection = None

                self._playtracker = CppPlayTracker(BBox(0, 0, play_width, play_height), pt_config)
                self._current_roi = None
                self._current_roi_aspect = None
        else:
            assert not self._cpp_playtracker

            self._breakaway_detection = BreakawayDetection(args.game_config)

            camera_cfg = args.game_config["rink"]["camera"]
            stop_dir_delay = int(
                camera_cfg.get(
                    "stop_on_dir_change_delay",
                    getattr(args, "stop_on_dir_change_delay", 0),
                )
            )
            cancel_stop = bool(
                camera_cfg.get(
                    "cancel_stop_on_opposite_dir",
                    bool(getattr(args, "cancel_stop_on_opposite_dir", 0)),
                )
            )
            cancel_hyst = int(
                camera_cfg.get(
                    "stop_cancel_hysteresis_frames",
                    getattr(args, "stop_cancel_hysteresis_frames", 0),
                )
            )
            cooldown_frames = int(
                camera_cfg.get(
                    "stop_delay_cooldown_frames",
                    getattr(args, "stop_delay_cooldown_frames", 0),
                )
            )

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
                stop_on_dir_change_delay=stop_dir_delay,
                cancel_stop_on_opposite_dir=cancel_stop,
                stop_delay_cooldown_frames=cooldown_frames,
                color=(255, 128, 64),
                thickness=5,
                device=self._device,
                min_height=play_height / 10,
                time_to_dest_speed_limit_frames=int(
                    args.game_config["rink"]["camera"].get("time_to_dest_speed_limit_frames", 10) * speed_scale
                ),
            )

            post_nonstop = int(camera_cfg.get("breakaway_detection", {}).get("post_nonstop_stop_delay_count", 0))
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
                stop_on_dir_change_delay=stop_dir_delay,
                cancel_stop_on_opposite_dir=cancel_stop,
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
                post_nonstop_stop_delay=post_nonstop,
                cancel_hysteresis_frames=cancel_hyst,
                stop_delay_cooldown_frames=cooldown_frames,
                time_to_dest_speed_limit_frames=int(
                    args.game_config["rink"]["camera"].get("time_to_dest_speed_limit_frames", 10) * speed_scale
                ),
            )

        if self._camera_ui_enabled:
            self._init_ui_controls()

        # Optional transformer-based camera controller
        self._camera_controller = getattr(args, "camera_controller", "rule")
        self._camera_model: Optional[CameraPanZoomTransformer] = None
        self._camera_norm: Optional[CameraNorm] = None
        self._camera_window: int = int(getattr(args, "camera_window", 8))
        self._camera_feat_buf: deque = deque(maxlen=self._camera_window)
        self._camera_prev_center: Optional[Tuple[float, float]] = None
        self._camera_prev_h: Optional[float] = None
        self._camera_aspect = float(self._final_aspect_ratio)
        cm_path = getattr(args, "camera_model", None)
        if self._camera_controller == "transformer" and cm_path:
            try:
                ckpt = torch.load(cm_path, map_location="cpu")
                sd, norm, window = unpack_checkpoint(ckpt)
                d_in = 11  # must match build_frame_features
                self._camera_model = CameraPanZoomTransformer(d_in=d_in)
                self._camera_model.load_state_dict(sd)
                self._camera_model.to(self._device)
                self._camera_model.eval()
                self._camera_norm = norm
                # Override window if checkpoint carries its own
                self._camera_window = int(getattr(args, "camera_window", window))
                self._camera_feat_buf = deque(maxlen=self._camera_window)
                logger.info(f"Loaded camera transformer from {cm_path} (window={self._camera_window})")
            except Exception as ex:
                logger.warning(f"Failed to load camera model at {cm_path}: {ex}. Falling back to rule controller.")
                self._camera_controller = "rule"

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
        if (self._frame_counter > 1) and (not getattr(self._args, "no_wide_start", False)):
            raise AssertionError("Not currently meant for setting at runtime")
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
        self._initial_box_applied = True

    def get_cluster_boxes(
        self,
        online_tlwhs: torch.Tensor,
        online_ids: torch.Tensor,
        cluster_counts: List[int],
        centroids: Optional[torch.Tensor] = None,
    ):
        if centroids is None:
            centroids = self._cluster_centroids

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

        self._cluster_man.calculate_all_clusters(center_points=center_batch(online_tlwhs), ids=online_ids)
        boxes_map = dict()
        boxes_list = []
        for cluster_count in cluster_counts:
            largest_cluster_ids = self._cluster_man.prune_not_in_largest_cluster(
                num_clusters=cluster_count, ids=online_ids
            )
            if len(largest_cluster_ids):
                largest_cluster_ids_box = self._hockey_mom.get_current_bounding_box(largest_cluster_ids)
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
            self._jersey_tracker.observe_tracking_id_number_info(frame_id=frame_id, info=current_info)

    def process_actions_info(self, frame_index: int, frame_id: int, data: Dict[str, Any]) -> None:
        action_results = data.get("action_results")
        if not action_results:
            return
        action_results = action_results[frame_index]
        if not action_results:
            return
        # action_results is a list of dicts per frame
        self._action_tracker.observe(frame_id=frame_id, infos=action_results)

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
        # Per-frame player footprint centers (bottom of bbox midpoints) and ids
        player_bottom_points_list: List[torch.Tensor] = []
        player_ids_list: List[torch.Tensor] = []

        # Make the original images into a list so that we can release the batched one
        original_images_list: List[Optional[torch.Tensor]] = []
        for i in range(original_images.shape[0]):
            original_images_list.append(original_images[i])
        del original_images

        debug = getattr(self._args, 'debug_play_tracker', False)
        for frame_index, video_data_sample in enumerate(track_data_sample.video_data_samples):
            scalar_frame_id = video_data_sample.frame_id
            frame_id = torch.tensor([scalar_frame_id], dtype=torch.int64)
            det_count = len(video_data_sample.pred_instances.bboxes) if hasattr(video_data_sample, 'pred_instances') and hasattr(video_data_sample.pred_instances, 'bboxes') else -1
            online_tlwhs = batch_tlbrs_to_tlwhs(video_data_sample.pred_track_instances.bboxes)
            online_ids = video_data_sample.pred_track_instances.instances_id

            if True:
                # goes a few fps faster when async if this is on CPU
                frame_id = frame_id.cpu()
                online_tlwhs = online_tlwhs.cpu()
                online_ids = online_ids.cpu()

            if debug:
                try:
                    n = int(len(online_ids))
                except Exception:
                    n = -1
                logger.info(f"PlayTracker frame {int(scalar_frame_id)}: det={det_count} tracks={n}")

            self.process_jerseys_info(frame_index=frame_index, frame_id=scalar_frame_id, data=results)

            # Cache rink_profile once if available in metainfo
            try:
                if self._rink_profile_cache is None and hasattr(video_data_sample, 'metainfo'):
                    rp = video_data_sample.metainfo.get("rink_profile", None)
                    if rp is not None:
                        self._rink_profile_cache = rp
            except Exception:
                pass

            self._frame_counter += 1

            online_im = original_images_list[frame_index]
            # deref the original one to free the cuda memory
            original_images_list[frame_index] = None

            #
            # Play
            #
            cluster_counts = [3, 2]

            vis_ignored_tracking_ids: Union[Set[int], None] = set()
            cluster_boxes_map: Dict[int, Union[BBox, torch.Tensor]] = {}
            cluster_enclosing_box: Optional[torch.Tensor] = None
            removed_cluster_outlier_box: Dict[int, Union[BBox, torch.Tensor]] = {}

            # Optionally use transformer-based controller
            use_transformer = self._camera_controller == "transformer" and self._camera_model is not None

            # Always sync camera UI controls so sliders affect tracking even without plotting.
            self._apply_ui_controls()

            if self._playtracker is not None:
                assert not use_transformer, "Cannot use transformer with C++ PlayTracker"
                online_bboxes = [BBox(*b) for b in video_data_sample.pred_track_instances.bboxes]
                playtracker_results = self._playtracker.forward(online_ids.cpu().tolist(), online_bboxes)

                if playtracker_results.largest_tracking_bbox is not None:
                    largest_bbox = from_bbox(playtracker_results.largest_tracking_bbox.bbox)
                    largest_bbox = batch_tlbrs_to_tlwhs(largest_bbox.unsqueeze(0)).squeeze(0)
                    vis_ignored_tracking_ids = {playtracker_results.largest_tracking_bbox.tracking_id}
                else:
                    largest_bbox = None

                # if playtracker_results.leftmost_tracking_bbox_id is not None:
                #     vis_ignored_tracking_ids.add(playtracker_results.leftmost_tracking_bbox_id)

                # if playtracker_results.rightmost_tracking_bbox_id is not None:
                #     vis_ignored_tracking_ids.add(playtracker_results.rightmost_tracking_bbox_id)

                cluster_enclosing_box = from_bbox(playtracker_results.final_cluster_box)
                cluster_boxes_map = playtracker_results.cluster_boxes
                removed_cluster_outlier_box = getattr(
                    playtracker_results, "removed_cluster_outlier_box", {}
                )
                # cluster_boxes = [cluster_enclosing_box, cluster_enclosing_box]

                fast_roi_bounding_box = from_bbox(playtracker_results.tracking_boxes[0])
                current_fast_box_list.append(fast_roi_bounding_box)

                current_box = from_bbox(playtracker_results.tracking_boxes[1])
                current_box_list.append(current_box)
                if debug:
                    logger.info(f"  boxes: fast={fast_roi_bounding_box.tolist()} current={current_box.tolist()}")

                if self._args.plot_moving_boxes:
                    # Play box
                    if torch.sum(self._play_box == self._hockey_mom._video_frame.bounding_box()) != 4:
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
                    online_im = self._draw_ui_overlay(online_im)

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
                    online_tlwhs, mask, largest_bbox = remove_largest_bbox(online_tlwhs, min_boxes=4)
                    online_ids = online_ids[mask]

                self._hockey_mom.append_online_objects(online_ids, online_tlwhs)

                # Prefer external camera boxes provided by an upstream trunk
                external_cam_boxes = results.get("camera_boxes", None)
                current_box = None
                if external_cam_boxes is not None and frame_index < len(external_cam_boxes):
                    cb = external_cam_boxes[frame_index]
                    if not isinstance(cb, torch.Tensor):
                        cb = torch.as_tensor(cb, dtype=torch.float32)
                    # Keep on same device as play_box to avoid clamp device mismatch
                    current_box = cb.to(device=self._play_box.device, dtype=torch.float32)

                if current_box is None and use_transformer:
                    tlwh_np = (
                        online_tlwhs.numpy() if isinstance(online_tlwhs, torch.Tensor) else online_tlwhs
                    )
                    # Build and append features
                    feat_np = build_frame_features(
                        tlwh=tlwh_np,
                        norm=self._camera_norm,
                        prev_cam_center=self._camera_prev_center,
                        prev_cam_h=self._camera_prev_h,
                    ) if self._camera_norm is not None else None
                    if feat_np is not None:
                        self._camera_feat_buf.append(torch.from_numpy(feat_np).to(self._device))
                    if len(self._camera_feat_buf) >= self._camera_window:
                        x = torch.stack(list(self._camera_feat_buf), dim=0).unsqueeze(0)
                        with torch.no_grad():
                            pred = self._camera_model(x).squeeze(0).detach().cpu().numpy()
                        cx, cy, hr = float(pred[0]), float(pred[1]), float(pred[2])
                        self._camera_prev_center = (cx, cy)
                        self._camera_prev_h = hr
                        # Map to current arena box
                        arena_box = self.get_arena_box()
                        W = float(width(arena_box))
                        H = float(height(arena_box))
                        w_px = max(1.0, hr * H * float(self._final_aspect_ratio))
                        h_px = max(1.0, hr * H)
                        cx_px = float(arena_box[0] + cx * W)
                        cy_px = float(arena_box[1] + cy * H)
                        left = cx_px - w_px / 2.0
                        top = cy_px - h_px / 2.0
                        right = left + w_px
                        bottom = top + h_px
                        current_box = torch.tensor([left, top, right, bottom], dtype=torch.float, device=image_device)
                        current_box = clamp_box(current_box, self._play_box)
                if current_box is None:
                    # BEGIN Clusters and Cluster Boxes (rule-based default)
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
                elif self._args.plot_cluster_tracking and not cluster_boxes_map:
                    # Populate cluster boxes for visualization even if camera boxes are external.
                    cluster_boxes_map, cluster_boxes = self.get_cluster_boxes(
                        online_tlwhs, online_ids, cluster_counts=cluster_counts
                    )
                    if cluster_boxes_map:
                        cluster_enclosing_box = get_enclosing_box(cluster_boxes)

            # Compute per-player representative points for minimap
            if isinstance(online_tlwhs, torch.Tensor) and online_tlwhs.numel() > 0:
                # centers and bottoms
                cx = online_tlwhs[:, 0] + online_tlwhs[:, 2] / 2.0
                cy = online_tlwhs[:, 1] + online_tlwhs[:, 3] / 2.0
                by = online_tlwhs[:, 1] + online_tlwhs[:, 3]
                # Default to bottom points
                rep_x = cx
                rep_y = by
                # If rink centroid known, choose center vs bottom based on half of rink
                try:
                    if self._rink_profile_cache is not None and "centroid" in self._rink_profile_cache:
                        ry = float(self._rink_profile_cache["centroid"][1])
                        # For near half (y > ry), prefer centers (skates off-ice / occlusion by boards)
                        mask = cy > ry
                        rep_y = torch.where(mask, cy, by)
                        rep_x = cx
                except Exception:
                    pass
                foot_points = torch.stack([rep_x, rep_y], dim=1).to(torch.float32)
            else:
                foot_points = torch.empty((0, 2), dtype=torch.float32)
            player_bottom_points_list.append(foot_points)
            player_ids_list.append(online_ids.clone() if isinstance(online_ids, torch.Tensor) else torch.tensor([]))

            if self._args.plot_boundaries and self._boundaries is not None:
                online_im = self._boundaries.draw(online_im)

            if self._args.plot_all_detections is not None:
                detections = video_data_sample.pred_instances.bboxes
                if not isinstance(detections, dict):
                    for detection, score in zip(detections, video_data_sample.pred_instances.scores):
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
                        online_im = vis.plot_alpha_rectangle(
                            online_im,
                            from_bbox(cluster_boxes_map[cc]),
                            color=cluster_box_colors[cc],
                            thickness=1,
                            label=f"cluster_box_{cc}",
                            opacity_percent=20,
                        )
                    if cc in removed_cluster_outlier_box:
                        online_im = vis.plot_alpha_rectangle(
                            online_im,
                            from_bbox(removed_cluster_outlier_box[cc]),
                            color=cluster_box_colors[cc],
                            label=f"OUTLIER({cc})",
                            opacity_percent=75,
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
                # for cluster_id, outlier_box in playtracker_results.removed_cluster_outlier_box.items():
                #     online_im = vis.plot_alpha_rectangle(
                #         online_im,
                #         from_bbox(outlier_box),
                #         color=(255, 0, 0),
                #         label=f"OUTLIER({cluster_id})",
                #         opacity_percent=75,
                #     )

            #
            # END Clusters and Cluster Boxes
            #

            # Update trackers with per-frame metrics
            self.process_actions_info(frame_index=frame_index, frame_id=frame_id.item() if torch.is_tensor(frame_id) else int(frame_id), data=results)
            online_im = self._jersey_tracker.draw(image=online_im, tracking_ids=online_ids, bboxes=online_tlwhs)
            online_im = self._action_tracker.draw(image=online_im, tracking_ids=online_ids, bboxes_tlwh=online_tlwhs)

            # Run ROI mover when using Python-only path; skip breakaway when C++ tracker or external cam boxes are used
            # use_external_cam = results.get("camera_boxes") is not None
            use_external_cam = None
            if (self._playtracker is None) or use_external_cam:
                # Only apply Python breakaway if no C++ tracker and no external camera controller and not transformer
                if (self._playtracker is None) and (not use_external_cam) and (self._camera_controller != "transformer"):
                    current_box, online_im = self.calculate_breakaway(
                        current_box=current_box,
                        online_im=online_im,
                        speed_adjust_box=self._current_roi,
                        average_current_box=True,
                    )

                # Backup the last calculated box
                self._previous_cluster_union_box = current_box.clone()

                # Clamp to arena
                current_box = clamp_box(current_box, self._play_box)

                # Maybe set initial box sizes when --no-wide-start is enabled
                if self._args.no_wide_start and (not self._initial_box_applied):
                    has_valid_cluster_box = bool(cluster_boxes_map)
                    box_is_full_frame = torch.allclose(
                        current_box.to(dtype=torch.float32, device=self._play_box.device),
                        self._play_box,
                    )
                    if has_valid_cluster_box or (not box_is_full_frame):
                        self.set_initial_tracking_box(current_box)

                # Apply through ROI moving boxes (fast follower + aspect follower)
                roi_input_box = (
                    to_bbox(current_box, self._cpp_boxes)
                    if isinstance(self._current_roi, PyLivingBox)
                    else current_box
                )
                fast_roi_bounding_box = self._current_roi.forward(roi_input_box)
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
        # Attach per-frame player bottom points and ids for downstream overlays
        results["player_bottom_points"] = player_bottom_points_list
        results["player_ids"] = player_ids_list
        if self._rink_profile_cache is not None:
            results["rink_profile"] = self._rink_profile_cache
        # print(f"FAST: {current_fast_box_list}")
        # print(f"CURRENT: {current_box_list}")

        # We want to track if it's slow
        img = torch.stack(online_images)
        img = StreamCheckpoint(img)
        img._verbose = True
        results["img"] = img

        return results

    # ---------------------- UI Controls ----------------------
    def _init_ui_controls(self):
        try:
            cv2.namedWindow(self._ui_window_name, cv2.WINDOW_NORMAL)
            camera_cfg = self._args.game_config["rink"]["camera"]
            bkd = camera_cfg.get("breakaway_detection", {})
            # Trackbar ranges
            def tb(name, maxv, init):
                cv2.createTrackbar(name, self._ui_window_name, int(init), int(maxv), lambda v: None)

            stop_dir_delay = int(camera_cfg.get("stop_on_dir_change_delay", getattr(self._args, "stop_on_dir_change_delay", 10)))
            cancel_stop = 1 if bool(camera_cfg.get("cancel_stop_on_opposite_dir", bool(getattr(self._args, "cancel_stop_on_opposite_dir", 1)))) else 0
            hyst = int(camera_cfg.get("stop_cancel_hysteresis_frames", getattr(self._args, "stop_cancel_hysteresis_frames", 2)))
            cooldown = int(camera_cfg.get("stop_delay_cooldown_frames", getattr(self._args, "stop_delay_cooldown_frames", 2)))
            ov_delay = int(bkd.get("overshoot_stop_delay_count", getattr(self._args, "overshoot_stop_delay_count", 6)))
            postns = int(bkd.get("post_nonstop_stop_delay_count", getattr(self._args, "post_nonstop_stop_delay_count", 6)))
            ov_scale = int(100 * float(bkd.get("overshoot_scale_speed_ratio", 0.7)))
            ttg = int(camera_cfg.get("time_to_dest_speed_limit_frames", 10))

            tb("Stop_Direction_Change_Delay_Frames", 60, stop_dir_delay)
            tb("Cancel_Stop_On_Opposite_Direction", 1, cancel_stop)
            tb("Stop_Cancel_Hysteresis_Frames", 10, hyst)
            tb("Stop_Delay_Cooldown_Frames", 30, cooldown)
            tb("Overshoot_Stop_Delay_Frames", 60, ov_delay)
            tb("Post_Nonstop_Stop_Delay_Frames", 60, postns)
            tb("Overshoot_Speed_Ratio_x100", 200, ov_scale)
            tb("Time_To_Dest_Speed_Limit_Frames", 120, ttg)
            # Translation constraints and target selection
            # Apply to fast and/or follower boxes
            tb("Apply_To_Fast_Box", 1, 1)
            tb("Apply_To_Follower_Box", 1, 1)

            # --- Stitch rotate degrees (-90..+90 mapped to 0..180) ---
            try:
                # Prefer consolidated game_config runtime value; fall back to CLI/defaults
                rot_cfg = None
                try:
                    rot_cfg = self._args.game_config.get("game", {}).get("stitching", {}).get(
                        "stitch-rotate-degrees"
                    )
                except Exception:
                    rot_cfg = None
                if rot_cfg is None:
                    rot_cfg = getattr(self._args, "stitch_rotate_degrees", 0.0)
                rot_cfg = 0.0 if rot_cfg is None else float(rot_cfg)
                # Map -90..+90 -> 0..180
                tb("Stitch_Rotate_Degrees", 180, int(max(-90.0, min(90.0, rot_cfg)) + 90.0))
            except Exception:
                # Non-fatal if config missing
                tb("Stitch_Rotate_Degrees", 180, 90)
            # Speeds/accels (scale sliders by x10 to allow decimals)
            # Initialize from YAML if present, else use reasonable defaults
            msx = int(10 * float(self._args.game_config["rink"]["camera"].get("max_speed_x", 30.0)))
            msy = int(10 * float(self._args.game_config["rink"]["camera"].get("max_speed_y", 30.0)))
            maxx = int(10 * float(self._args.game_config["rink"]["camera"].get("max_accel_x", 10.0)))
            maxy = int(10 * float(self._args.game_config["rink"]["camera"].get("max_accel_y", 10.0)))
            tb("Max_Speed_X_x10", 2000, msx)
            tb("Max_Speed_Y_x10", 2000, msy)
            tb("Max_Accel_X_x10", 1000, maxx)
            tb("Max_Accel_Y_x10", 1000, maxy)
            # Save defaults for reset
            self._ui_defaults = dict(
                Stop_Direction_Change_Delay_Frames=stop_dir_delay,
                Cancel_Stop_On_Opposite_Direction=cancel_stop,
                Stop_Cancel_Hysteresis_Frames=hyst,
                Stop_Delay_Cooldown_Frames=cooldown,
                Overshoot_Stop_Delay_Frames=ov_delay,
                Post_Nonstop_Stop_Delay_Frames=postns,
                Overshoot_Speed_Ratio_x100=ov_scale,
                Time_To_Dest_Speed_Limit_Frames=ttg,
                Apply_To_Fast_Box=1,
                Apply_To_Follower_Box=1,
                Max_Speed_X_x10=msx,
                Max_Speed_Y_x10=msy,
                Max_Accel_X_x10=maxx,
                Max_Accel_Y_x10=maxy,
                Stitch_Rotate_Degrees=cv2.getTrackbarPos("Stitch_Rotate_Degrees", self._ui_window_name) if cv2.getWindowProperty(self._ui_window_name, 0) is not None else 90,
            )
            self._ui_inited = True
            # ---- Color controls window ----
            try:
                cv2.namedWindow(self._ui_color_window_name, cv2.WINDOW_NORMAL)
                try:
                    cv2.moveWindow(self._ui_color_window_name, 520, 50)
                except Exception:
                    pass
                def tb2(name, maxv, init):
                    cv2.createTrackbar(name, self._ui_color_window_name, int(init), int(maxv), lambda v: None)
                tb2("White_Balance_Kelvin_Enable", 1, 0)
                tb2("White_Balance_Kelvin_Temperature", 15000, 6500)
                tb2("White_Balance_Red_Gain_x100", 300, 100)
                tb2("White_Balance_Green_Gain_x100", 300, 100)
                tb2("White_Balance_Blue_Gain_x100", 300, 100)
                tb2("Brightness_Multiplier_x100", 300, 100)
                tb2("Contrast_Multiplier_x100", 300, 100)
                tb2("Gamma_Multiplier_x100", 300, 100)
                self._ui_color_inited = True
            except Exception:
                self._ui_color_inited = False
        except Exception:
            self._camera_ui_enabled = False

    def _apply_ui_controls(self):
        if not self._camera_ui_enabled or not self._ui_inited:
            return
        try:
            # Read trackbars
            dir_delay = int(cv2.getTrackbarPos("Stop_Direction_Change_Delay_Frames", self._ui_window_name))
            cancel_opp = bool(cv2.getTrackbarPos("Cancel_Stop_On_Opposite_Direction", self._ui_window_name))
            hyst = int(cv2.getTrackbarPos("Stop_Cancel_Hysteresis_Frames", self._ui_window_name))
            cooldown = int(cv2.getTrackbarPos("Stop_Delay_Cooldown_Frames", self._ui_window_name))
            ov_delay = int(cv2.getTrackbarPos("Overshoot_Stop_Delay_Frames", self._ui_window_name))
            postns = int(cv2.getTrackbarPos("Post_Nonstop_Stop_Delay_Frames", self._ui_window_name))
            ov_scal = cv2.getTrackbarPos("Overshoot_Speed_Ratio_x100", self._ui_window_name) / 100.0
            ttg = int(cv2.getTrackbarPos("Time_To_Dest_Speed_Limit_Frames", self._ui_window_name))

            # Update YAML-like config so all downstream reads are consistent
            camera_cfg = self._args.game_config["rink"]["camera"]
            camera_cfg["stop_on_dir_change_delay"] = dir_delay
            camera_cfg["cancel_stop_on_opposite_dir"] = cancel_opp
            camera_cfg["stop_cancel_hysteresis_frames"] = hyst
            camera_cfg["stop_delay_cooldown_frames"] = cooldown
            bkd = camera_cfg.setdefault("breakaway_detection", {})
            bkd["overshoot_stop_delay_count"] = ov_delay
            bkd["post_nonstop_stop_delay_count"] = postns
            bkd["overshoot_scale_speed_ratio"] = float(ov_scal)
            camera_cfg["time_to_dest_speed_limit_frames"] = ttg

            # Stitch rotation degrees (-90..+90)
            try:
                rot_slider = cv2.getTrackbarPos("Stitch_Rotate_Degrees", self._ui_window_name)
                rot_deg = float(rot_slider - 90)
                game_cfg = self._args.game_config.setdefault("game", {}).setdefault("stitching", {})
                game_cfg["stitch-rotate-degrees"] = rot_deg
            except Exception:
                pass
            # --- Color controls (second window) ---
            if self._ui_color_inited:
                try:
                    color_win = self._ui_color_window_name
                    # Read back color trackbars
                    wbk_enable = cv2.getTrackbarPos("White_Balance_Kelvin_Enable", color_win)
                    kelvin = cv2.getTrackbarPos("White_Balance_Kelvin_Temperature", color_win)
                    r100 = cv2.getTrackbarPos("White_Balance_Red_Gain_x100", color_win)
                    g100 = cv2.getTrackbarPos("White_Balance_Green_Gain_x100", color_win)
                    b100 = cv2.getTrackbarPos("White_Balance_Blue_Gain_x100", color_win)
                    br100 = cv2.getTrackbarPos("Brightness_Multiplier_x100", color_win)
                    ct100 = cv2.getTrackbarPos("Contrast_Multiplier_x100", color_win)
                    gm100 = cv2.getTrackbarPos("Gamma_Multiplier_x100", color_win)

                    color_cfg = camera_cfg.setdefault("color", {})
                    # Kelvin WB overrides manual gains when enabled
                    if int(wbk_enable) > 0:
                        color_cfg["white_balance_temp"] = f"{int(max(1000, min(40000, kelvin)))}k"
                        color_cfg.pop("white_balance", None)
                    else:
                        # Map 50..200 -> 0.5..2.0
                        rgain = max(1, r100) / 100.0
                        ggain = max(1, g100) / 100.0
                        bgain = max(1, b100) / 100.0
                        color_cfg["white_balance"] = [float(rgain), float(ggain), float(bgain)]
                        color_cfg.pop("white_balance_temp", None)
                    color_cfg["brightness"] = max(1, br100) / 100.0
                    color_cfg["contrast"] = max(1, ct100) / 100.0
                    color_cfg["gamma"] = max(1, gm100) / 100.0
                except Exception:
                    pass
            # Read selection + constraints
            apply_fast = bool(cv2.getTrackbarPos("Apply_To_Fast_Box", self._ui_window_name))
            apply_follower = bool(cv2.getTrackbarPos("Apply_To_Follower_Box", self._ui_window_name))
            msx = cv2.getTrackbarPos("Max_Speed_X_x10", self._ui_window_name) / 10.0
            msy = cv2.getTrackbarPos("Max_Speed_Y_x10", self._ui_window_name) / 10.0
            maxx = cv2.getTrackbarPos("Max_Accel_X_x10", self._ui_window_name) / 10.0
            maxy = cv2.getTrackbarPos("Max_Accel_Y_x10", self._ui_window_name) / 10.0
            # Apply to Python movers (live)
            if isinstance(self._current_roi, MovingBox) and apply_fast:
                mb = self._current_roi
                mb._max_speed_x = torch.tensor(msx, dtype=torch.float, device=mb.device)
                mb._max_speed_y = torch.tensor(msy, dtype=torch.float, device=mb.device)
                mb._max_accel_x = torch.tensor(maxx, dtype=torch.float, device=mb.device)
                mb._max_accel_y = torch.tensor(maxy, dtype=torch.float, device=mb.device)
            if isinstance(self._current_roi_aspect, MovingBox) and apply_follower:
                mb = self._current_roi_aspect
                mb._max_speed_x = torch.tensor(msx, dtype=torch.float, device=mb.device)
                mb._max_speed_y = torch.tensor(msy, dtype=torch.float, device=mb.device)
                mb._max_accel_x = torch.tensor(maxx, dtype=torch.float, device=mb.device)
                mb._max_accel_y = torch.tensor(maxy, dtype=torch.float, device=mb.device)

            # Apply to Python movers live (if available)
            if isinstance(self._current_roi_aspect, MovingBox):
                mba = self._current_roi_aspect
                mba._stop_on_dir_change_delay = torch.tensor(int(dir_delay), dtype=torch.int64, device=mba.device)
                mba._cancel_stop_on_opposite_dir = cancel_opp
                mba._cancel_hysteresis_frames = torch.tensor(int(hyst), dtype=torch.int64, device=mba.device)
                mba._stop_delay_cooldown_frames = torch.tensor(int(cooldown), dtype=torch.int64, device=mba.device)
                mba._post_nonstop_stop_delay = torch.tensor(int(postns), dtype=torch.int64, device=mba.device)
            if self._playtracker is not None:
                try:
                    if apply_fast:
                        lb = self._playtracker.get_live_box(0)
                        lb.set_braking(dir_delay, cancel_opp, hyst, cooldown, postns)
                        lb.set_translation_constraints(msx, msy, maxx, maxy)
                    if apply_follower:
                        lb = self._playtracker.get_live_box(1)
                        lb.set_braking(dir_delay, cancel_opp, hyst, cooldown, postns)
                        lb.set_translation_constraints(msx, msy, maxx, maxy)
                    self._playtracker.set_breakaway_braking(ov_delay, ov_scal)
                except Exception:
                    pass
            # For Python-only breakaway values, we read from args.game_config in calculate_breakaway
            self._handle_ui_keyboard()
        except Exception:
            pass

    def _draw_ui_overlay(self, img):
        if not self._camera_ui_enabled or not self._ui_inited:
            return img
        try:
            camera_cfg = self._args.game_config["rink"]["camera"]
            bkd = camera_cfg.get("breakaway_detection", {})
            # Show current panorama rotate degrees if set
            try:
                rot_deg = (
                    self._args.game_config.get("game", {}).get("stitching", {}).get("stitch-rotate-degrees", 0.0)
                )
                rot_text = f" Rot={float(rot_deg):+.1f}deg"
            except Exception:
                rot_text = ""
            text = (
                f"DirChangeDelay={camera_cfg.get('stop_on_dir_change_delay', 0)} "
                f"CancelOpposite={int(bool(camera_cfg.get('cancel_stop_on_opposite_dir', 0)))} "
                f"Hysteresis={camera_cfg.get('stop_cancel_hysteresis_frames', 0)} "
                f"Cooldown={camera_cfg.get('stop_delay_cooldown_frames', 0)} "
                f"OvershotDelay={bkd.get('overshoot_stop_delay_count', 0)} "
                f"PostNonstop={bkd.get('post_nonstop_stop_delay_count', 0)} "
                f"OvershotRatio={bkd.get('overshoot_scale_speed_ratio', 0.0):.2f}"
                f"{rot_text}"
            )
            img = vis.plot_text(
                img,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )
            # Key help
            img = vis.plot_text(
                img,
                "[R]eset  [S]ave",
                (20, 70),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                thickness=2,
            )
            self._handle_ui_keyboard()
        except Exception:
            pass
        return img

    def _handle_ui_keyboard(self):
        if not self._camera_ui_enabled or not self._ui_inited:
            return
        try:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('r') or k == ord('R'):
                self._reset_ui_controls()
            elif k == ord('s') or k == ord('S'):
                self._save_ui_config()
        except Exception:
            pass

    def _reset_ui_controls(self):
        if not self._camera_ui_enabled or not self._ui_inited:
            return
        try:
            for name, val in self._ui_defaults.items():
                cv2.setTrackbarPos(name, self._ui_window_name, int(val))
        except Exception:
            pass

    def _save_ui_config(self):
        # Save current game_config to private config.yaml if game_id is present
        try:
            game_id = getattr(self._args, "game_id", None)
            if not game_id:
                return
            priv = get_game_config_private(game_id=game_id) or {}
            # Merge in camera section from current args
            priv.setdefault("rink", {}).setdefault("camera", {})
            camera_cfg = self._args.game_config["rink"]["camera"]
            priv["rink"]["camera"] = camera_cfg
            # Also persist game.stitching.stitch-rotate-degrees if present
            try:
                stitch_cfg = self._args.game_config.get("game", {}).get("stitching", {})
                if isinstance(stitch_cfg, dict) and "stitch-rotate-degrees" in stitch_cfg:
                    priv.setdefault("game", {}).setdefault("stitching", {})[
                        "stitch-rotate-degrees"
                    ] = stitch_cfg["stitch-rotate-degrees"]
            except Exception:
                pass
            save_private_config(game_id=game_id, data=priv, verbose=True)
        except Exception:
            pass

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
                current_box = make_box_at_center(average_center, width(current_box), height(current_box))
            else:
                current_box = make_box_at_center(edge_center, width(current_box), height(current_box))

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
                        accel_x=group_x_velocity * self._breakaway_detection.group_velocity_speed_ratio,
                        accel_y=None,
                        scale_constraints=self._breakaway_detection.scale_speed_constraints,
                        nonstop_delay=self._breakaway_detection.nonstop_delay_count,
                    )
                else:
                    # Overshoot: either multiplicative damping or begin a stop-delay
                    overshoot_delay = int(
                        self._args.game_config["rink"]["camera"]["breakaway_detection"].get(
                            "overshoot_stop_delay_count", 0
                        )
                    )
                    if overshoot_delay > 0:
                        # If using Python-only path, MovingBox has begin_stop_delay
                        if isinstance(speed_adjust_box, MovingBox):
                            speed_adjust_box.begin_stop_delay(delay_x=overshoot_delay)
                        else:
                            # C++ LivingBox exposes begin_stop_delay via bindings
                            speed_adjust_box.begin_stop_delay(overshoot_delay, None)
                    else:
                        # Cut the speed quickly due to overshoot
                        speed_adjust_box.scale_speed(ratio_x=self._breakaway_detection.overshoot_scale_speed_ratio)
        #
        # END Breakway detection
        #
        return current_box, online_im
