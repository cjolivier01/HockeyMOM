"""High-level play tracker that controls camera pan/zoom from detections."""

from __future__ import absolute_import, division, print_function

import copy
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.actions.action_tracker import ActionTracker
from hmlib.bbox.box_functions import (
    center,
    center_batch,
    clamp_box,
    get_enclosing_box,
    height,
    make_box_at_center,
    remove_largest_bbox,
    scale_box,
    tlwh_to_tlbr_single,
    width,
)
from hmlib.builder import HM
from hmlib.camera.camera import HockeyMOM
from hmlib.camera.clusters import ClusterMan
from hmlib.camera.moving_box import MovingBox
from hmlib.config import get_game_config_private, get_nested_value, save_private_config
from hmlib.jersey.jersey_tracker import JerseyTracker
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.ui import show_image
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_last
from hmlib.utils.progress_bar import ProgressBar
from hockeymom.core import AllLivingBoxConfig, BBox, HmLogLevel, PlayTrackerConfig
from hockeymom.core import PlayTracker as CppPlayTracker

from .camera_transformer import (
    CameraNorm,
    CameraPanZoomTransformer,
    build_frame_features,
    unpack_checkpoint,
)
from .living_box import PyLivingBox, from_bbox, to_bbox

_CPP_BOXES: bool = True
# _CPP_BOXES: bool = False

_CPP_PLAYTRACKER: bool = True and _CPP_BOXES
# _CPP_PLAYTRACKER: bool = False and _CPP_BOXES

_MISSING = object()
_COLOR_TRACKBARS = {
    "White_Balance_Kelvin_Enable",
    "White_Balance_Kelvin_Temperature",
    "White_Balance_Red_Gain_x100",
    "White_Balance_Green_Gain_x100",
    "White_Balance_Blue_Gain_x100",
    "Brightness_Multiplier_x100",
    "Contrast_Multiplier_x100",
    "Gamma_Multiplier_x100",
}


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
        game_config: Dict[str, Any],
        game_id: Optional[str] = None,
        cam_ignore_largest: bool = True,
        no_wide_start: bool = False,
        track_ids: Optional[Union[str, List[int], Set[int]]] = None,
        debug_play_tracker: bool = False,
        plot_individual_player_tracking: bool = False,
        plot_boundaries: bool = False,
        plot_all_detections: Optional[float] = None,
        plot_trajectories: bool = False,
        plot_speed: bool = False,
        plot_jersey_numbers: bool = False,
        plot_actions: bool = False,
        plot_moving_boxes: bool = False,
        camera_ui: int = 0,
        camera_controller: str = "rule",
        camera_model: Optional[str] = None,
        camera_window: int = 8,
        force_stitching: bool = False,
        stitch_rotation_controller: Optional[Any] = None,
        cluster_centroids: Optional[torch.Tensor] = None,
        cpp_boxes: bool = _CPP_BOXES,
        cpp_playtracker: bool = _CPP_PLAYTRACKER,
        plot_cluster_tracking: bool = False,
    ):
        """Track play and drive camera box based on detections and configs.

        @param hockey_mom: `HockeyMOM` instance providing tracker and video state.
        @param play_box: Box describing the allowed play region in TLBR coords.
        @param device: Torch device for computations.
        @param original_clip_box: Clip box applied to original image, if any.
        @param progress_bar: Optional progress bar for CLI status.
        @param game_config: Consolidated game configuration dict.
        @param game_id: Optional game identifier (for saving UI edits).
        @param cam_ignore_largest: Whether to ignore largest bbox.
        @param no_wide_start: Whether to skip initial wide camera box.
        @param track_ids: Optional track id whitelist (str or collection).
        @param debug_play_tracker: Enable per-frame debug logging.
        @param plot_moving_boxes: Enable ROI/mover overlay plotting.
        @param plot_individual_player_tracking: Enable per-player overlays.
        @param plot_boundaries: Enable rink boundary overlays.
        @param plot_all_detections: Score threshold for plotting all dets.
        @param plot_trajectories: Enable track trajectory overlays.
        @param plot_speed: Enable velocity/acceleration overlays.
        @param plot_jersey_numbers: Enable jersey number overlays.
        @param plot_actions: Enable action overlays.
        @param camera_ui: Enable camera UI windows (0/1).
        @param camera_controller: Camera controller mode ('rule'/'transformer').
        @param camera_model: Optional camera transformer checkpoint path.
        @param camera_window: Transformer time window length.
        @param force_stitching: Enable stitching side color UI controls.
        @param stitch_rotation_controller: Optional stitching controller handle.
        @param cluster_centroids: Optional precomputed cluster centroids.
        @param cpp_boxes: If True, use C++ BBox types internally.
        @param cpp_playtracker: If True, use the C++ PlayTracker backend.
        """
        super(PlayTracker, self).__init__()
        self._game_config: Dict[str, Any] = game_config
        self._game_id: Optional[str] = game_id
        self._cam_ignore_largest: bool = bool(cam_ignore_largest)
        self._no_wide_start: bool = bool(no_wide_start)
        self._debug_play_tracker: bool = bool(debug_play_tracker)
        self._plot_moving_boxes: bool = bool(plot_moving_boxes) or debug_play_tracker
        self._plot_trajectories: bool = bool(plot_trajectories)
        self._plot_individual_player_tracking: bool = (
            bool(plot_individual_player_tracking) or debug_play_tracker
        )
        self._plot_boundaries: bool = bool(plot_boundaries)
        self._plot_all_detections: Optional[float] = plot_all_detections
        self._cpp_boxes = cpp_boxes
        self._cpp_playtracker = cpp_playtracker
        self._playtracker: Union[PlayTracker, None] = None
        self._ui_dirty_paths: Set[Tuple[str, ...]] = set()
        self._hockey_mom: HockeyMOM = hockey_mom
        self._plot_cluster_tracking = plot_cluster_tracking or debug_play_tracker
        self._plot_speed = bool(plot_speed)
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
        self._camera_ui_enabled = bool(camera_ui)
        self._ui_window_name = "Tracker Controls"
        self._ui_inited = False
        self._ui_color_window_name = "Tracker Controls (Color)"
        self._ui_color_left_window_name = "Tracker Controls (Left Color)"
        self._ui_color_right_window_name = "Tracker Controls (Right Color)"
        self._ui_color_inited = False
        self._ui_color_left_inited = False
        self._ui_color_right_inited = False
        # Per-window slider defaults: {window_name: {slider_name: default_value}}
        self._ui_defaults: Dict[str, Dict[str, int]] = {}
        self._ui_controls_dirty = True
        self._stitch_rotation_controller = stitch_rotation_controller
        self._force_stitching: bool = bool(force_stitching)
        self._stitch_slider_enabled = False

        # Optional transformer-based camera controller
        self._camera_controller = camera_controller or "rule"
        self._camera_model: Optional[CameraPanZoomTransformer] = None
        self._camera_norm: Optional[CameraNorm] = None
        self._camera_window: int = int(camera_window)
        self._camera_feat_buf: deque = deque(maxlen=self._camera_window)
        self._camera_prev_center: Optional[Tuple[float, float]] = None
        self._camera_prev_h: Optional[float] = None
        self._camera_aspect = float(self._final_aspect_ratio)

        if cluster_centroids is not None:
            centroids_tensor = torch.as_tensor(cluster_centroids, dtype=torch.float32).cpu()
            if centroids_tensor.ndim != 2 or centroids_tensor.shape[1] != 2:
                raise ValueError("cluster_centroids must be shaped (N, 2)")
            self._cluster_centroids = centroids_tensor

        self._jersey_tracker = JerseyTracker(show=bool(plot_jersey_numbers))
        self._action_tracker = ActionTracker(show=bool(plot_actions))
        # Cache for rink_profile (combined_mask, centroid, etc.) pulled from data samples meta
        self._rink_profile_cache = None

        camera_cfg = self._game_config.setdefault("rink", {}).setdefault("camera", {})
        self._camera_base_speed_x = float(
            self._hockey_mom._camera_box_max_speed_x.detach().cpu().item()
        )
        self._camera_base_speed_y = float(
            self._hockey_mom._camera_box_max_speed_y.detach().cpu().item()
        )
        self._camera_base_accel_x = float(
            self._hockey_mom._camera_box_max_accel_x.detach().cpu().item()
        )
        self._camera_base_accel_y = float(
            self._hockey_mom._camera_box_max_accel_y.detach().cpu().item()
        )
        camera_cfg.setdefault("max_speed_ratio_x", 1.0)
        camera_cfg.setdefault("max_speed_ratio_y", 1.0)
        camera_cfg.setdefault("max_accel_ratio_x", 1.0)
        camera_cfg.setdefault("max_accel_ratio_y", 1.0)
        self._max_speed_ratio_x = float(camera_cfg["max_speed_ratio_x"])
        self._max_speed_ratio_y = float(camera_cfg["max_speed_ratio_y"])
        self._max_accel_ratio_x = float(camera_cfg["max_accel_ratio_x"])
        self._max_accel_ratio_y = float(camera_cfg["max_accel_ratio_y"])
        self._camera_speed_x = self._hockey_mom._camera_box_max_speed_x * self._max_speed_ratio_x
        self._camera_speed_y = self._hockey_mom._camera_box_max_speed_y * self._max_speed_ratio_y
        self._camera_accel_x = self._hockey_mom._camera_box_max_accel_x * self._max_accel_ratio_x
        self._camera_accel_y = self._hockey_mom._camera_box_max_accel_y * self._max_accel_ratio_y
        self._validate_required_camera_config()

        # Tracking specific ids
        self._track_ids: Set[int] = set()
        if track_ids:
            if isinstance(track_ids, str):
                ids_iter = (int(i) for i in track_ids.split(",") if i)
            else:
                ids_iter = (int(i) for i in track_ids)  # type: ignore[arg-type]
            self._track_ids = set(ids_iter)

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
            current_roi_config.max_speed_x = self._camera_speed_x * 1.5 / speed_scale
            current_roi_config.max_speed_y = self._camera_speed_y * 1.5 / speed_scale
            current_roi_config.max_accel_x = self._camera_accel_x * 1.1 / speed_scale
            current_roi_config.max_accel_y = self._camera_accel_y * 1.1 / speed_scale

            # Resizing
            current_roi_config.max_speed_w = self._camera_speed_x * 1.5 / speed_scale / 1.8
            current_roi_config.max_speed_h = self._camera_speed_y * 1.5 / speed_scale / 1.8
            current_roi_config.max_accel_w = self._camera_accel_x * 1.1 / speed_scale
            current_roi_config.max_accel_h = self._camera_accel_y * 1.1 / speed_scale

            current_roi_config.max_width = play_width
            current_roi_config.max_height = play_height
            current_roi_config.min_height = play_height / 10

            current_roi_config.stop_resizing_on_dir_change = False
            current_roi_config.stop_translation_on_dir_change = False
            current_roi_config.arena_box = to_bbox(self.get_arena_box(), self._cpp_boxes)
            # Frames-to-destination speed limiting (scaled by fps)
            ttg_frames = int(
                self._game_config["rink"]["camera"].get("time_to_dest_speed_limit_frames", 10)
            )
            current_roi_config.time_to_dest_speed_limit_frames = int(
                ttg_frames * self._hockey_mom.fps_speed_scale
            )

            #
            # Create and configure `AllLivingBoxConfig` for `_current_roi_aspect`
            #
            current_roi_aspect_config = AllLivingBoxConfig()

            kEXTRA_FOLLOWING_SCALE_DOWN = 1

            current_roi_aspect_config.max_speed_x = (
                self._camera_speed_x * 1 / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )
            current_roi_aspect_config.max_speed_y = (
                self._camera_speed_y * 1 / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )
            current_roi_aspect_config.max_accel_x = (
                self._camera_accel_x / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )
            current_roi_aspect_config.max_accel_y = (
                self._camera_accel_y / speed_scale / kEXTRA_FOLLOWING_SCALE_DOWN
            )

            current_roi_aspect_config.max_speed_w = self._camera_speed_x * 1 / speed_scale / 1.8
            current_roi_aspect_config.max_speed_h = self._camera_speed_y * 1 / speed_scale / 1.8
            current_roi_aspect_config.max_accel_w = self._camera_accel_x / speed_scale
            current_roi_aspect_config.max_accel_h = self._camera_accel_y / speed_scale

            current_roi_aspect_config.max_width = play_width
            current_roi_aspect_config.max_height = play_height
            current_roi_aspect_config.min_height = play_height / 5

            current_roi_aspect_config.stop_resizing_on_dir_change = True
            current_roi_aspect_config.stop_translation_on_dir_change = True
            current_roi_aspect_config.sticky_translation = True
            # Prefer YAML; fall back to CLI hm_opts defaults/overrides
            stop_dir_delay = int(self._initial_camera_value("stop_on_dir_change_delay"))
            cancel_stop = bool(self._initial_camera_value("cancel_stop_on_opposite_dir"))
            cancel_hyst = int(self._initial_camera_value("stop_cancel_hysteresis_frames"))
            cooldown_frames = int(self._initial_camera_value("stop_delay_cooldown_frames"))
            current_roi_aspect_config.stop_translation_on_dir_change_delay = stop_dir_delay
            current_roi_aspect_config.cancel_stop_on_opposite_dir = cancel_stop
            current_roi_aspect_config.cancel_stop_hysteresis_frames = cancel_hyst
            current_roi_aspect_config.stop_delay_cooldown_frames = cooldown_frames

            # FIXME: get this from config
            current_roi_aspect_config.dynamic_acceleration_scaling = 1.0
            current_roi_aspect_config.arena_angle_from_vertical = 30.0

            current_roi_aspect_config.arena_box = to_bbox(self.get_arena_box(), self._cpp_boxes)
            cam_cfg = self._camera_cfg()
            current_roi_aspect_config.sticky_size_ratio_to_frame_width = cam_cfg[
                "sticky_size_ratio_to_frame_width"
            ]
            current_roi_aspect_config.sticky_translation_gaussian_mult = cam_cfg[
                "sticky_translation_gaussian_mult"
            ]
            current_roi_aspect_config.unsticky_translation_size_ratio = cam_cfg[
                "unsticky_translation_size_ratio"
            ]
            current_roi_aspect_config.sticky_sizing = True
            current_roi_aspect_config.scale_dest_width = cam_cfg["follower_box_scale_width"]
            current_roi_aspect_config.scale_dest_height = cam_cfg["follower_box_scale_height"]
            current_roi_aspect_config.fixed_aspect_ratio = self._final_aspect_ratio

            # If we are not using the C++ PlayTracker, or if the controller is the transformer,
            # initialize Python-side movers without creating the C++ PlayTracker.
            if not self._cpp_playtracker or self._camera_controller == "transformer":
                self._breakaway_detection = BreakawayDetection(self._game_config)
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
                pt_config.ignore_largest_bbox = self._cam_ignore_largest
                pt_config.no_wide_start = self._no_wide_start

                pt_config.ignore_outlier_players = True  # EXPERIMENTAL
                pt_config.ignore_left_and_right_extremes = False  # EXPERIMENTAL

                breakaway_detection = self._breakaway_cfg()
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
                # Optional: use stop-delay braking instead of multiplicative overshoot scale
                pt_config.play_detector.overshoot_stop_delay_count = int(
                    breakaway_detection["overshoot_stop_delay_count"]
                )
                # After nonstop ends, optionally brake to stop
                post_nonstop = int(breakaway_detection["post_nonstop_stop_delay_count"])
                current_roi_aspect_config.post_nonstop_stop_delay_count = post_nonstop
                self._breakaway_detection = None

                self._playtracker = CppPlayTracker(BBox(0, 0, play_width, play_height), pt_config)
                self._current_roi = None
                self._current_roi_aspect = None
        else:
            assert not self._cpp_playtracker

            self._breakaway_detection = BreakawayDetection(self._game_config)

            camera_cfg = self._camera_cfg()
            stop_dir_delay = int(self._initial_camera_value("stop_on_dir_change_delay"))
            cancel_stop = bool(self._initial_camera_value("cancel_stop_on_opposite_dir"))
            cancel_hyst = int(self._initial_camera_value("stop_cancel_hysteresis_frames"))
            cooldown_frames = int(self._initial_camera_value("stop_delay_cooldown_frames"))

            self._current_roi: Union[MovingBox, PyLivingBox] = MovingBox(
                label="Current ROI",
                bbox=start_box.clone(),
                arena_box=self.get_arena_box(),
                max_speed_x=self._camera_speed_x * 1.5 / speed_scale,
                max_speed_y=self._camera_speed_y * 1.5 / speed_scale,
                max_accel_x=self._camera_accel_x * 1.1 / speed_scale,
                max_accel_y=self._camera_accel_y * 1.1 / speed_scale,
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
                    self._require_camera_value("time_to_dest_speed_limit_frames") * speed_scale
                ),
            )

            post_nonstop = int(self._require_breakaway_value("post_nonstop_stop_delay_count"))
            self._current_roi_aspect: Union[MovingBox, PyLivingBox] = MovingBox(
                label="AspectRatio",
                bbox=start_box.clone(),
                arena_box=self.get_arena_box(),
                max_speed_x=self._camera_speed_x * 1 / speed_scale,
                max_speed_y=self._camera_speed_y * 1 / speed_scale,
                max_accel_x=self._camera_accel_x * 1 / speed_scale,
                max_accel_y=self._camera_accel_y * 1 / speed_scale,
                max_width=play_width,
                max_height=play_height,
                stop_on_dir_change=True,
                stop_on_dir_change_delay=stop_dir_delay,
                cancel_stop_on_opposite_dir=cancel_stop,
                sticky_translation=True,
                sticky_size_ratio_to_frame_width=camera_cfg["sticky_size_ratio_to_frame_width"],
                sticky_translation_gaussian_mult=camera_cfg["sticky_translation_gaussian_mult"],
                unsticky_translation_size_ratio=camera_cfg["unsticky_translation_size_ratio"],
                sticky_sizing=True,
                scale_width=camera_cfg["follower_box_scale_width"],
                scale_height=camera_cfg["follower_box_scale_height"],
                fixed_aspect_ratio=self._final_aspect_ratio,
                color=(255, 0, 255),
                thickness=5,
                device=self._device,
                min_height=play_height / 5,
                post_nonstop_stop_delay=post_nonstop,
                cancel_hysteresis_frames=cancel_hyst,
                stop_delay_cooldown_frames=cooldown_frames,
                time_to_dest_speed_limit_frames=int(
                    self._require_camera_value("time_to_dest_speed_limit_frames") * speed_scale
                ),
            )

        if self._camera_ui_enabled:
            self._init_ui_controls()

        cm_path = camera_model
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
                self._camera_window = int(window)
                self._camera_feat_buf = deque(maxlen=self._camera_window)
                logger.info(
                    f"Loaded camera transformer from {cm_path} (window={self._camera_window})"
                )
            except Exception as ex:
                logger.warning(
                    f"Failed to load camera model at {cm_path}: {ex}. Falling back to rule controller."
                )
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
        if (self._frame_counter > 1) and (not self._no_wide_start):
            raise AssertionError("Not currently meant for setting at runtime")
        frame_box = self.get_arena_box()
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

        original_images = unwrap_tensor(results.pop("original_images"))

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

        debug = self._debug_play_tracker

        for frame_index, video_data_sample in enumerate(track_data_sample.video_data_samples):
            scalar_frame_id = video_data_sample.frame_id
            frame_id = torch.tensor([scalar_frame_id], dtype=torch.int64)

            track_inst = video_data_sample.pred_track_instances

            # Ensure any tracker outputs that may have been produced on a
            # different CUDA stream are synchronized with the current stream.
            ids = track_inst.instances_id

            online_tlwhs = batch_tlbrs_to_tlwhs(unwrap_tensor(track_inst.bboxes))
            online_ids = unwrap_tensor(ids)
            track_mask = get_track_mask(track_inst)

            if False:
                # goes a few fps faster when async if this is on CPU
                frame_id = frame_id.cpu()
                online_tlwhs = online_tlwhs.cpu()
                online_ids = online_ids.cpu()
                if isinstance(track_mask, torch.Tensor):
                    track_mask = track_mask.cpu()
                    online_tlwhs = online_tlwhs[track_mask]
                    online_ids = online_ids[track_mask]

            self.process_jerseys_info(
                frame_index=frame_index, frame_id=scalar_frame_id, data=results
            )

            # Cache rink_profile once if available in metainfo
            try:
                if self._rink_profile_cache is None and hasattr(video_data_sample, "metainfo"):
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
            use_transformer = (
                self._camera_controller == "transformer" and self._camera_model is not None
            )

            # Always sync camera UI controls so sliders affect tracking even without plotting.
            self._apply_ui_controls()

            if self._playtracker is not None:
                assert not use_transformer, "Cannot use transformer with C++ PlayTracker"
                tracking_ids: List[int] = []
                online_bboxes: List[BBox] = []
                for tid, bbox in zip(
                    online_ids.cpu().tolist(), video_data_sample.pred_track_instances.bboxes
                ):
                    if tid < 0:
                        # Invalid ID
                        continue
                    tracking_ids.append(tid)
                    online_bboxes.append(BBox(*bbox))
                playtracker_results = self._playtracker.forward(tracking_ids, online_bboxes)
                for msg in getattr(playtracker_results, "log_messages", []):
                    try:
                        text = msg.message
                        level = msg.level
                    except Exception:
                        continue
                    if level == HmLogLevel.DEBUG:
                        logger.debug(text)
                    elif level == HmLogLevel.INFO:
                        logger.info(text)
                    elif level == HmLogLevel.WARNING:
                        logger.warning(text)
                    elif level == HmLogLevel.ERROR:
                        logger.error(text)
                    else:
                        logger.info(text)

                if playtracker_results.largest_tracking_bbox is not None:
                    largest_bbox = from_bbox(playtracker_results.largest_tracking_bbox.bbox)
                    largest_bbox = batch_tlbrs_to_tlwhs(largest_bbox.unsqueeze(0)).squeeze(0)
                    vis_ignored_tracking_ids = {
                        playtracker_results.largest_tracking_bbox.tracking_id
                    }
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
                # if debug:
                #     logger.info(
                #         f"  boxes: fast={fast_roi_bounding_box.tolist()} current={current_box.tolist()}"
                #     )

                if debug or self._plot_moving_boxes:
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
                    online_im = self._draw_ui_overlay(online_im)

                if (
                    # TODO: move this to the tracker
                    self._plot_individual_player_tracking
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
                if self._cam_ignore_largest and len(online_tlwhs):
                    # Don't remove unless we have at least 4 online items being tracked
                    online_tlwhs, mask, largest_bbox = remove_largest_bbox(
                        online_tlwhs, min_boxes=4
                    )
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
                        online_tlwhs.numpy()
                        if isinstance(online_tlwhs, torch.Tensor)
                        else online_tlwhs
                    )
                    # Build and append features
                    feat_np = (
                        build_frame_features(
                            tlwh=tlwh_np,
                            norm=self._camera_norm,
                            prev_cam_center=self._camera_prev_center,
                            prev_cam_h=self._camera_prev_h,
                        )
                        if self._camera_norm is not None
                        else None
                    )
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
                        current_box = torch.tensor(
                            [left, top, right, bottom], dtype=torch.float, device=image_device
                        )
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
                elif self._plot_cluster_tracking and not cluster_boxes_map:
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
                    if (
                        self._rink_profile_cache is not None
                        and "centroid" in self._rink_profile_cache
                    ):
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
            player_ids_list.append(
                online_ids.clone() if isinstance(online_ids, torch.Tensor) else torch.tensor([])
            )

            # if self._plot_boundaries and self._boundaries is not None:
            #     online_im = self._boundaries.draw(online_im)

            if self._plot_all_detections is not None:
                detections = video_data_sample.pred_instances.bboxes
                if not isinstance(detections, dict):
                    for detection, score in zip(
                        detections, video_data_sample.pred_instances.scores
                    ):
                        if score >= self._plot_all_detections:
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
            if self._plot_trajectories:
                for tid in online_ids:
                    hist = self._hockey_mom.get_history(tid)
                    if hist is not None:
                        hist.draw(online_im)

            if self._plot_individual_player_tracking:
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

            if self._plot_cluster_tracking:
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
            self.process_actions_info(
                frame_index=frame_index,
                frame_id=frame_id.item() if torch.is_tensor(frame_id) else int(frame_id),
                data=results,
            )
            online_im = self._jersey_tracker.draw(
                image=online_im, tracking_ids=online_ids, bboxes=online_tlwhs
            )
            online_im = self._action_tracker.draw(
                image=online_im, tracking_ids=online_ids, bboxes_tlwh=online_tlwhs
            )

            # Run ROI mover when using Python-only path; skip breakaway when C++ tracker or external cam boxes are used
            # use_external_cam = results.get("camera_boxes") is not None
            use_external_cam = None
            if (self._playtracker is None) or use_external_cam:
                # Only apply Python breakaway if no C++ tracker and no external camera controller and not transformer
                if (
                    (self._playtracker is None)
                    and (not use_external_cam)
                    and (self._camera_controller != "transformer")
                ):
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
                if self._no_wide_start and (not self._initial_box_applied):
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

                if self._plot_moving_boxes:
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

            if self._plot_speed:
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

        results["frame_ids"] = wrap_tensor(torch.stack(frame_ids_list))
        results["current_box"] = wrap_tensor(torch.stack(current_box_list))
        results["current_fast_box_list"] = wrap_tensor(torch.stack(current_fast_box_list))
        # Attach per-frame player bottom points and ids for downstream overlays
        results["player_bottom_points"] = wrap_tensor(torch.stack(player_bottom_points_list))
        results["player_ids"] = wrap_tensor(torch.stack(player_ids_list))
        if self._rink_profile_cache is not None:
            results["rink_profile"] = self._rink_profile_cache
        # print(f"FAST: {current_fast_box_list}")
        # print(f"CURRENT: {current_box_list}")

        # for img in online_images:
        #     show_image("Play Tracker", img, wait=False, scale=0.25)

        # We want to track if it's slow
        img = torch.stack(online_images)
        img = wrap_tensor(img)
        img._verbose = True
        results["img"] = img

        return results

    # ---------------------- UI Controls ----------------------
    def _kelvin_slider_value(self, value: Any) -> int:
        try:
            sval = str(value).lower()
            if sval.endswith("k"):
                sval = sval[:-1]
            return int(max(1000, min(15000, float(sval))))
        except Exception:
            return 6500

    def _on_trackbar(self, _value: Any):
        # Mark that UI controls changed so we can skip the heavy read when idle.
        self._ui_controls_dirty = True

    def _stitch_deg_to_slider(self, degrees: float) -> int:
        """Convert signed degrees (-90..+90) to slider position (left=positive)."""
        deg = max(-90.0, min(90.0, float(degrees)))
        return int(max(0, min(180, round(90 - deg))))

    def _slider_to_stitch_deg(self, position: int) -> float:
        """Convert slider position to signed degrees (-90..+90), left=positive."""
        return float(90 - position)

    def _current_stitch_rotation_degrees(self) -> Optional[float]:
        """Read the current post-stitch rotation from controller or config."""
        ctrl = self._stitch_rotation_controller
        if ctrl is not None:
            try:
                getter = getattr(ctrl, "get_post_stitch_rotate_degrees", None)
                if callable(getter):
                    return getter()
                val = getattr(ctrl, "post_stitch_rotate_degrees", None)
                if callable(val):
                    val = val()
                return val
            except Exception:
                pass
        try:
            return (
                self._game_config.get("game", {}).get("stitching", {}).get("stitch-rotate-degrees")
            )
        except Exception:
            return None

    def _set_stitch_rotation_degrees(self, degrees: Optional[float]) -> None:
        """Apply post-stitch rotation to the controller and keep config in sync."""
        ctrl = self._stitch_rotation_controller
        if ctrl is not None:
            try:
                setter = getattr(ctrl, "set_post_stitch_rotate_degrees", None)
                if callable(setter):
                    setter(degrees)
                else:
                    setattr(ctrl, "post_stitch_rotate_degrees", degrees)
            except Exception:
                pass
        self._set_config_value(
            self._game_config,
            ("game", "stitching", "stitch-rotate-degrees"),
            degrees,
            mark_dirty=True,
        )

    def _configure_stitch_slider(self, desired_degrees: float):
        """Set the 0..180 slider position based on desired degrees (-90..+90)."""
        if not self._stitch_slider_enabled:
            return
        slider_name = "Stitch_Rotate_Degrees"
        win = self._ui_window_name
        desired_degrees = max(-90.0, min(90.0, float(desired_degrees)))
        try:
            cv2.setTrackbarPos(slider_name, win, self._stitch_deg_to_slider(desired_degrees))
        except Exception:
            pass
        self._on_trackbar(None)

    def _base_color_slider_defaults(self, color_cfg: Dict[str, Any]) -> Dict[str, int]:
        defaults: Dict[str, int] = {
            "White_Balance_Kelvin_Enable": 0,
            "White_Balance_Kelvin_Temperature": 6500,
            "White_Balance_Red_Gain_x100": 100,
            "White_Balance_Green_Gain_x100": 100,
            "White_Balance_Blue_Gain_x100": 100,
            "Brightness_Multiplier_x100": 100,
            "Contrast_Multiplier_x100": 100,
            "Gamma_Multiplier_x100": 100,
        }
        wbk = color_cfg.get("white_balance_temp")
        wb = color_cfg.get("white_balance")
        if wbk is not None:
            defaults["White_Balance_Kelvin_Enable"] = 1
            defaults["White_Balance_Kelvin_Temperature"] = self._kelvin_slider_value(wbk)
        elif isinstance(wb, (list, tuple)) and len(wb) == 3:
            b, g, r = wb
            try:
                defaults["White_Balance_Red_Gain_x100"] = int(
                    max(1.0, min(300.0, float(r) * 100.0))
                )
                defaults["White_Balance_Green_Gain_x100"] = int(
                    max(1.0, min(300.0, float(g) * 100.0))
                )
                defaults["White_Balance_Blue_Gain_x100"] = int(
                    max(1.0, min(300.0, float(b) * 100.0))
                )
            except Exception:
                pass
        for cfg_key, slider_key in (
            ("brightness", "Brightness_Multiplier_x100"),
            ("contrast", "Contrast_Multiplier_x100"),
            ("gamma", "Gamma_Multiplier_x100"),
        ):
            if cfg_key in color_cfg:
                try:
                    val = color_cfg[cfg_key]
                    defaults[slider_key] = int(max(1.0, min(300.0, float(val) * 100.0)))
                except Exception:
                    pass
        return defaults

    def _color_slider_defaults(self) -> Dict[str, int]:
        # Global stitched camera color defaults from top-level + rink camera.
        try:
            base_cfg: Dict[str, Any] = {}
            top_camera = (
                self._game_config.get("camera", {}) if isinstance(self._game_config, dict) else {}
            )
            if isinstance(top_camera, dict):
                base_cfg.update(top_camera.get("color", {}) or top_camera)
            rink_camera = self._game_config.get("rink", {}).get("camera", {})  # type: ignore[assignment]
            if isinstance(rink_camera, dict):
                base_cfg.update(rink_camera.get("color", {}) or rink_camera)
            color_cfg = base_cfg
        except Exception:
            color_cfg = {}
        return self._base_color_slider_defaults(color_cfg)

    def _stitch_side_color_defaults(self, side: str) -> Dict[str, int]:
        # Per-side stitching defaults under game.stitching.<side>.color (if present).
        cfg: Dict[str, Any] = {}
        try:
            if isinstance(self._game_config, dict):
                game_cfg = self._game_config.get("game", {})
                stitching = game_cfg.get("stitching", {}) if isinstance(game_cfg, dict) else {}
                side_cfg = stitching.get(side, {}) if isinstance(stitching, dict) else {}
                if isinstance(side_cfg, dict):
                    color_sub = side_cfg.get("color", {}) or side_cfg
                    if isinstance(color_sub, dict):
                        cfg.update(color_sub)
        except Exception:
            pass
        return self._base_color_slider_defaults(cfg)

    def _set_ui_color_value_at_prefixes(
        self, prefixes: List[Tuple[str, ...]], key: str, value: Any
    ):
        for prefix in prefixes:
            self._set_ui_config_value(prefix + (key,), value)

    def _set_ui_color_value(self, key: str, value: Any):
        # Update rink-scoped camera color block; this is the canonical location
        # for runtime camera color configuration.
        self._set_ui_color_value_at_prefixes(
            prefixes=[("rink", "camera", "color")],
            key=key,
            value=value,
        )

    def _apply_color_window(self, window_name: str, prefixes: List[Tuple[str, ...]]) -> None:
        """Read color sliders from a window and update config under given prefixes."""
        try:
            color_win = window_name
            wbk_enable = cv2.getTrackbarPos("White_Balance_Kelvin_Enable", color_win)
            kelvin = cv2.getTrackbarPos("White_Balance_Kelvin_Temperature", color_win)
            r100 = cv2.getTrackbarPos("White_Balance_Red_Gain_x100", color_win)
            g100 = cv2.getTrackbarPos("White_Balance_Green_Gain_x100", color_win)
            b100 = cv2.getTrackbarPos("White_Balance_Blue_Gain_x100", color_win)
            br100 = cv2.getTrackbarPos("Brightness_Multiplier_x100", color_win)
            ct100 = cv2.getTrackbarPos("Contrast_Multiplier_x100", color_win)
            gm100 = cv2.getTrackbarPos("Gamma_Multiplier_x100", color_win)

            if int(wbk_enable) > 0:
                kelvin_val = f"{int(max(1000, min(40000, kelvin)))}k"
                self._set_ui_color_value_at_prefixes(prefixes, "white_balance_temp", kelvin_val)
                self._set_ui_color_value_at_prefixes(prefixes, "white_balance", _MISSING)
            else:
                rgain = max(1, r100) / 100.0
                ggain = max(1, g100) / 100.0
                bgain = max(1, b100) / 100.0
                self._set_ui_color_value_at_prefixes(
                    prefixes, "white_balance", [float(bgain), float(ggain), float(rgain)]
                )
                self._set_ui_color_value_at_prefixes(prefixes, "white_balance_temp", _MISSING)
            self._set_ui_color_value_at_prefixes(prefixes, "brightness", max(1, br100) / 100.0)
            self._set_ui_color_value_at_prefixes(prefixes, "contrast", max(1, ct100) / 100.0)
            self._set_ui_color_value_at_prefixes(prefixes, "gamma", max(1, gm100) / 100.0)
        except Exception:
            pass

    def _init_ui_controls(self):
        try:
            cv2.namedWindow(self._ui_window_name, cv2.WINDOW_NORMAL)

            # Trackbar ranges
            def tb(name, maxv, init):
                cv2.createTrackbar(
                    name, self._ui_window_name, int(init), int(maxv), self._on_trackbar
                )

            stop_dir_delay = int(self._initial_camera_value("stop_on_dir_change_delay"))
            cancel_stop = (
                1
                if bool(
                    self._initial_camera_value(
                        "cancel_stop_on_opposite_dir"
                    )
                )
                else 0
            )
            hyst = int(
                self._initial_camera_value("stop_cancel_hysteresis_frames")
            )
            cooldown = int(
                self._initial_camera_value("stop_delay_cooldown_frames")
            )
            ov_delay = int(
                self._initial_breakaway_value("overshoot_stop_delay_count")
            )
            postns = int(
                self._initial_breakaway_value("post_nonstop_stop_delay_count")
            )
            ov_scale = int(
                100
                * float(
                    self._initial_breakaway_value("overshoot_scale_speed_ratio")
                )
            )
            ttg = int(self._require_camera_value("time_to_dest_speed_limit_frames"))

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
            tb("Apply_To_Fast_Box", 1, 0)
            tb("Apply_To_Follower_Box", 1, 1)

            # --- Stitch rotate degrees (0..180 slider mapped to -90..+90 degrees) ---
            self._stitch_slider_enabled = self._stitch_rotation_controller is not None
            if self._stitch_slider_enabled:
                try:
                    rot_cfg = self._current_stitch_rotation_degrees()
                    if rot_cfg is None:
                        rot_cfg = 0.0
                    rot_cfg = float(rot_cfg)
                    slider_pos = self._stitch_deg_to_slider(rot_cfg)
                    tb("Stitch_Rotate_Degrees", 180, slider_pos)
                    self._configure_stitch_slider(rot_cfg)
                except Exception:
                    self._stitch_slider_enabled = False
            # Speeds/accels (scale sliders by x10 to allow decimals)
            camera_cfg = self._camera_cfg()
            msx = int(10 * self._camera_base_speed_x * float(camera_cfg["max_speed_ratio_x"]))
            msy = int(10 * self._camera_base_speed_y * float(camera_cfg["max_speed_ratio_y"]))
            maxx = int(10 * self._camera_base_accel_x * float(camera_cfg["max_accel_ratio_x"]))
            maxy = int(10 * self._camera_base_accel_y * float(camera_cfg["max_accel_ratio_y"]))
            tb("Max_Speed_X_x10", 2000, msx)
            tb("Max_Speed_Y_x10", 2000, msy)
            tb("Max_Accel_X_x10", 1000, maxx)
            tb("Max_Accel_Y_x10", 1000, maxy)
            # Save defaults for reset (per main controls window)
            self._ui_defaults[self._ui_window_name] = dict(
                Stop_Direction_Change_Delay_Frames=stop_dir_delay,
                Cancel_Stop_On_Opposite_Direction=cancel_stop,
                Stop_Cancel_Hysteresis_Frames=hyst,
                Stop_Delay_Cooldown_Frames=cooldown,
                Overshoot_Stop_Delay_Frames=ov_delay,
                Post_Nonstop_Stop_Delay_Frames=postns,
                Overshoot_Speed_Ratio_x100=ov_scale,
                Time_To_Dest_Speed_Limit_Frames=ttg,
                Apply_To_Fast_Box=0,
                Apply_To_Follower_Box=1,
                Max_Speed_X_x10=msx,
                Max_Speed_Y_x10=msy,
                Max_Accel_X_x10=maxx,
                Max_Accel_Y_x10=maxy,
            )
            if self._stitch_slider_enabled:
                try:
                    self._ui_defaults[self._ui_window_name]["Stitch_Rotate_Degrees"] = (
                        cv2.getTrackbarPos("Stitch_Rotate_Degrees", self._ui_window_name)
                        if cv2.getWindowProperty(self._ui_window_name, 0) is not None
                        else self._stitch_deg_to_slider(
                            self._current_stitch_rotation_degrees() or 0.0
                        )
                    )
                except Exception:
                    pass
            self._ui_inited = True
            # ---- Color controls window (stitched panorama) ----
            try:
                cv2.namedWindow(self._ui_color_window_name, cv2.WINDOW_NORMAL)
                try:
                    cv2.moveWindow(self._ui_color_window_name, 520, 50)
                except Exception:
                    pass

                def tb2(name, maxv, init):
                    cv2.createTrackbar(
                        name, self._ui_color_window_name, int(init), int(maxv), self._on_trackbar
                    )

                tb2("White_Balance_Kelvin_Enable", 1, 0)
                tb2("White_Balance_Kelvin_Temperature", 15000, 6500)
                tb2("White_Balance_Red_Gain_x100", 300, 100)
                tb2("White_Balance_Green_Gain_x100", 300, 100)
                tb2("White_Balance_Blue_Gain_x100", 300, 100)
                tb2("Brightness_Multiplier_x100", 300, 100)
                tb2("Contrast_Multiplier_x100", 300, 100)
                tb2("Gamma_Multiplier_x100", 300, 100)
                # Apply defaults from current config so UI reflects runtime values
                color_defaults = self._color_slider_defaults()
                for name, val in color_defaults.items():
                    try:
                        cv2.setTrackbarPos(name, self._ui_color_window_name, int(val))
                    except Exception:
                        pass
                try:
                    self._ui_defaults[self._ui_color_window_name] = dict(color_defaults)
                except Exception:
                    pass
                self._ui_color_inited = True
            except Exception:
                self._ui_color_inited = False

            # ---- Left/right stitching color controls (optional) ----
            enable_stitch_side_ui = bool(self._force_stitching)
            if enable_stitch_side_ui:
                # Left stitching color window
                try:
                    cv2.namedWindow(self._ui_color_left_window_name, cv2.WINDOW_NORMAL)
                    try:
                        cv2.moveWindow(self._ui_color_left_window_name, 520, 300)
                    except Exception:
                        pass

                    def tb_left(name, maxv, init):
                        cv2.createTrackbar(
                            name,
                            self._ui_color_left_window_name,
                            int(init),
                            int(maxv),
                            self._on_trackbar,
                        )

                    for name, maxv in (
                        ("White_Balance_Kelvin_Enable", 1),
                        ("White_Balance_Kelvin_Temperature", 15000),
                        ("White_Balance_Red_Gain_x100", 300),
                        ("White_Balance_Green_Gain_x100", 300),
                        ("White_Balance_Blue_Gain_x100", 300),
                        ("Brightness_Multiplier_x100", 300),
                        ("Contrast_Multiplier_x100", 300),
                        ("Gamma_Multiplier_x100", 300),
                    ):
                        tb_left(
                            name,
                            maxv,
                            100 if "Enable" not in name and "Temperature" not in name else 0,
                        )
                    left_defaults = self._stitch_side_color_defaults("left")
                    for name, val in left_defaults.items():
                        try:
                            cv2.setTrackbarPos(name, self._ui_color_left_window_name, int(val))
                        except Exception:
                            pass
                    self._ui_defaults[self._ui_color_left_window_name] = dict(left_defaults)
                    self._ui_color_left_inited = True
                except Exception:
                    self._ui_color_left_inited = False

                # Right stitching color window
                try:
                    cv2.namedWindow(self._ui_color_right_window_name, cv2.WINDOW_NORMAL)
                    try:
                        cv2.moveWindow(self._ui_color_right_window_name, 520, 550)
                    except Exception:
                        pass

                    def tb_right(name, maxv, init):
                        cv2.createTrackbar(
                            name,
                            self._ui_color_right_window_name,
                            int(init),
                            int(maxv),
                            self._on_trackbar,
                        )

                    for name, maxv in (
                        ("White_Balance_Kelvin_Enable", 1),
                        ("White_Balance_Kelvin_Temperature", 15000),
                        ("White_Balance_Red_Gain_x100", 300),
                        ("White_Balance_Green_Gain_x100", 300),
                        ("White_Balance_Blue_Gain_x100", 300),
                        ("Brightness_Multiplier_x100", 300),
                        ("Contrast_Multiplier_x100", 300),
                        ("Gamma_Multiplier_x100", 300),
                    ):
                        tb_right(
                            name,
                            maxv,
                            100 if "Enable" not in name and "Temperature" not in name else 0,
                        )
                    right_defaults = self._stitch_side_color_defaults("right")
                    for name, val in right_defaults.items():
                        try:
                            cv2.setTrackbarPos(name, self._ui_color_right_window_name, int(val))
                        except Exception:
                            pass
                    self._ui_defaults[self._ui_color_right_window_name] = dict(right_defaults)
                    self._ui_color_right_inited = True
                except Exception:
                    self._ui_color_right_inited = False
        except Exception:
            import traceback

            traceback.print_exc()
            print("Failed to initialize camera UI controls.", file=sys.stderr)
            self._camera_ui_enabled = False

    def _apply_ui_controls(self):
        if not self._camera_ui_enabled or not self._ui_inited:
            return
        try:
            # Still poll keyboard every frame for responsiveness
            self._handle_ui_keyboard()
        except Exception:
            pass
        if not self._ui_controls_dirty:
            return
        try:
            self._ui_controls_dirty = False
            # Read trackbars
            dir_delay = int(
                cv2.getTrackbarPos("Stop_Direction_Change_Delay_Frames", self._ui_window_name)
            )
            cancel_opp = bool(
                cv2.getTrackbarPos("Cancel_Stop_On_Opposite_Direction", self._ui_window_name)
            )
            hyst = int(cv2.getTrackbarPos("Stop_Cancel_Hysteresis_Frames", self._ui_window_name))
            cooldown = int(cv2.getTrackbarPos("Stop_Delay_Cooldown_Frames", self._ui_window_name))
            ov_delay = int(cv2.getTrackbarPos("Overshoot_Stop_Delay_Frames", self._ui_window_name))
            postns = int(cv2.getTrackbarPos("Post_Nonstop_Stop_Delay_Frames", self._ui_window_name))
            ov_scal = cv2.getTrackbarPos("Overshoot_Speed_Ratio_x100", self._ui_window_name) / 100.0
            ttg = int(cv2.getTrackbarPos("Time_To_Dest_Speed_Limit_Frames", self._ui_window_name))

            # Update YAML-like config so all downstream reads are consistent
            self._set_ui_config_value(
                ("rink", "camera", "stop_on_dir_change_delay"), int(dir_delay)
            )
            self._set_ui_config_value(
                ("rink", "camera", "cancel_stop_on_opposite_dir"), bool(cancel_opp)
            )
            self._set_ui_config_value(
                ("rink", "camera", "stop_cancel_hysteresis_frames"), int(hyst)
            )
            self._set_ui_config_value(
                ("rink", "camera", "stop_delay_cooldown_frames"), int(cooldown)
            )
            self._set_ui_config_value(
                ("rink", "camera", "breakaway_detection", "overshoot_stop_delay_count"),
                int(ov_delay),
            )
            self._set_ui_config_value(
                ("rink", "camera", "breakaway_detection", "post_nonstop_stop_delay_count"),
                int(postns),
            )
            self._set_ui_config_value(
                ("rink", "camera", "breakaway_detection", "overshoot_scale_speed_ratio"),
                float(ov_scal),
            )
            self._set_ui_config_value(
                ("rink", "camera", "time_to_dest_speed_limit_frames"),
                int(ttg),
            )

            # Stitch rotation degrees
            if self._stitch_slider_enabled:
                try:
                    rot_slider = cv2.getTrackbarPos("Stitch_Rotate_Degrees", self._ui_window_name)
                    rot_deg = self._slider_to_stitch_deg(rot_slider)
                    self._set_stitch_rotation_degrees(float(rot_deg))
                except Exception:
                    pass
            # --- Color controls (stitched + left/right stitching) ---
            if self._ui_color_inited:
                # Global stitched color adjustments
                self._apply_color_window(
                    self._ui_color_window_name,
                    prefixes=[("rink", "camera", "color")],
                )
            if self._ui_color_left_inited:
                # Left camera in stitching dataloader
                self._apply_color_window(
                    self._ui_color_left_window_name,
                    prefixes=[("game", "stitching", "left", "color")],
                )
            if self._ui_color_right_inited:
                # Right camera in stitching dataloader
                self._apply_color_window(
                    self._ui_color_right_window_name,
                    prefixes=[("game", "stitching", "right", "color")],
                )
            # Read selection + constraints
            apply_fast = bool(cv2.getTrackbarPos("Apply_To_Fast_Box", self._ui_window_name))
            apply_follower = bool(cv2.getTrackbarPos("Apply_To_Follower_Box", self._ui_window_name))
            msx = cv2.getTrackbarPos("Max_Speed_X_x10", self._ui_window_name) / 10.0
            msy = cv2.getTrackbarPos("Max_Speed_Y_x10", self._ui_window_name) / 10.0
            maxx = cv2.getTrackbarPos("Max_Accel_X_x10", self._ui_window_name) / 10.0
            maxy = cv2.getTrackbarPos("Max_Accel_Y_x10", self._ui_window_name) / 10.0
            if self._camera_base_speed_x > 0:
                self._set_ui_config_value(
                    ("rink", "camera", "max_speed_ratio_x"),
                    float(msx / max(self._camera_base_speed_x, 1e-6)),
                )
            if self._camera_base_speed_y > 0:
                self._set_ui_config_value(
                    ("rink", "camera", "max_speed_ratio_y"),
                    float(msy / max(self._camera_base_speed_y, 1e-6)),
                )
            if self._camera_base_accel_x > 0:
                self._set_ui_config_value(
                    ("rink", "camera", "max_accel_ratio_x"),
                    float(maxx / max(self._camera_base_accel_x, 1e-6)),
                )
            if self._camera_base_accel_y > 0:
                self._set_ui_config_value(
                    ("rink", "camera", "max_accel_ratio_y"),
                    float(maxy / max(self._camera_base_accel_y, 1e-6)),
                )
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
                mba._stop_on_dir_change_delay = torch.tensor(
                    int(dir_delay), dtype=torch.int64, device=mba.device
                )
                mba._cancel_stop_on_opposite_dir = cancel_opp
                mba._cancel_hysteresis_frames = torch.tensor(
                    int(hyst), dtype=torch.int64, device=mba.device
                )
                mba._stop_delay_cooldown_frames = torch.tensor(
                    int(cooldown), dtype=torch.int64, device=mba.device
                )
                mba._post_nonstop_stop_delay = torch.tensor(
                    int(postns), dtype=torch.int64, device=mba.device
                )
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
            # For Python-only breakaway values, we read from self._game_config in calculate_breakaway
        except Exception:
            # If we failed to read UI, try again next frame
            self._ui_controls_dirty = True
            pass

    def _draw_ui_overlay(self, img):
        if not self._camera_ui_enabled or not self._ui_inited:
            return img
        try:
            camera_cfg = self._camera_cfg()
            bkd = self._breakaway_cfg()
            # Show current panorama rotate degrees if set
            try:
                rot_deg = self._current_stitch_rotation_degrees()
                rot_text = f" Rot={float(rot_deg):+.1f}deg" if rot_deg is not None else ""
            except Exception:
                rot_text = ""
            text = (
                f"DirChangeDelay={camera_cfg['stop_on_dir_change_delay']} "
                f"CancelOpposite={int(bool(camera_cfg['cancel_stop_on_opposite_dir']))} "
                f"Hysteresis={camera_cfg['stop_cancel_hysteresis_frames']} "
                f"Cooldown={camera_cfg['stop_delay_cooldown_frames']} "
                f"OvershotDelay={bkd['overshoot_stop_delay_count']} "
                f"PostNonstop={bkd['post_nonstop_stop_delay_count']} "
                f"OvershotRatio={float(bkd['overshoot_scale_speed_ratio']):.2f}"
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
            if k == ord("r") or k == ord("R"):
                self._reset_ui_controls()
            elif k == ord("s") or k == ord("S"):
                self._save_ui_config()
        except Exception:
            pass

    def _reset_ui_controls(self):
        if not self._camera_ui_enabled or not self._ui_inited:
            return
        try:
            for win, sliders in self._ui_defaults.items():
                for name, val in sliders.items():
                    try:
                        cv2.setTrackbarPos(name, win, int(val))
                    except Exception:
                        # Ignore missing windows/trackbars
                        pass
            self._ui_controls_dirty = True
        except Exception:
            pass

    def _values_equal(self, a, b) -> bool:
        if a is _MISSING or b is _MISSING:
            return a is b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-6
        return a == b

    def _camera_cfg(self) -> Dict[str, Any]:
        cfg = self._game_config.get("rink", {}).get("camera")
        if not isinstance(cfg, dict):
            raise KeyError("Missing rink.camera configuration")
        return cfg

    def _breakaway_cfg(self) -> Dict[str, Any]:
        camera_cfg = self._camera_cfg()
        bkd = camera_cfg.get("breakaway_detection")
        if not isinstance(bkd, dict):
            raise KeyError("Missing rink.camera.breakaway_detection configuration")
        return bkd

    def _require_camera_value(self, key: str):
        camera_cfg = self._camera_cfg()
        if key not in camera_cfg:
            raise KeyError(f"Missing rink.camera.{key}")
        return camera_cfg[key]

    def _require_breakaway_value(self, key: str):
        bkd = self._breakaway_cfg()
        if key not in bkd:
            raise KeyError(f"Missing rink.camera.breakaway_detection.{key}")
        return bkd[key]

    def _initial_camera_value(self, config_key: str):
        return self._require_camera_value(config_key)

    def _initial_breakaway_value(self, key: str):
        return self._require_breakaway_value(key)

    def _validate_required_camera_config(self):
        required_keys = (
            "stop_on_dir_change_delay",
            "cancel_stop_on_opposite_dir",
            "stop_cancel_hysteresis_frames",
            "stop_delay_cooldown_frames",
            "time_to_dest_speed_limit_frames",
            "sticky_size_ratio_to_frame_width",
            "sticky_translation_gaussian_mult",
            "unsticky_translation_size_ratio",
            "follower_box_scale_width",
            "follower_box_scale_height",
        )
        for key in required_keys:
            self._require_camera_value(key)
        breakaway_keys = (
            "overshoot_stop_delay_count",
            "post_nonstop_stop_delay_count",
            "overshoot_scale_speed_ratio",
            "min_considered_group_velocity",
            "group_ratio_threshold",
            "group_velocity_speed_ratio",
            "scale_speed_constraints",
            "nonstop_delay_count",
        )
        for key in breakaway_keys:
            self._require_breakaway_value(key)
        color_cfg = self._camera_cfg().setdefault("color", {})
        color_cfg.setdefault("brightness", 1.0)
        color_cfg.setdefault("contrast", 1.0)
        color_cfg.setdefault("gamma", 1.0)
        if ("white_balance" not in color_cfg) and ("white_balance_temp" not in color_cfg):
            color_cfg["white_balance"] = [1.0, 1.0, 1.0]

    def _set_config_value(
        self,
        root: Dict[str, Any],
        path: Tuple[str, ...],
        value: Any,
        *,
        mark_dirty: bool = False,
        cleanup_empty: bool = False,
    ) -> bool:
        cur = root
        parents: List[Tuple[Dict[str, Any], str]] = []
        for key in path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                if value is _MISSING:
                    # nothing to delete
                    return False
                cur[key] = {}
            parents.append((cur, key))
            cur = cur[key]
        leaf = path[-1]
        current = cur.get(leaf, _MISSING)
        if value is _MISSING:
            if leaf not in cur:
                return False
            del cur[leaf]
            if cleanup_empty:
                while parents:
                    parent, key = parents.pop()
                    child = parent[key]
                    if isinstance(child, dict) and not child:
                        del parent[key]
                    else:
                        break
            if mark_dirty:
                self._ui_dirty_paths.add(path)
            return True
        if self._values_equal(current, value):
            return False
        cur[leaf] = copy.deepcopy(value)
        if mark_dirty:
            self._ui_dirty_paths.add(path)
        return True

    def _set_ui_config_value(self, path: Tuple[str, ...], value: Any):
        self._set_config_value(self._game_config, path, value, mark_dirty=True)

    def _get_config_path_value(self, path: Tuple[str, ...]):
        cur: Any = self._game_config
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return _MISSING
            cur = cur[key]
        return cur

    def _set_priv_path(self, priv: Dict[str, Any], path: Tuple[str, ...], value: Any) -> bool:
        return self._set_config_value(priv, path, value, mark_dirty=False, cleanup_empty=True)

    def _delete_priv_path(self, priv: Dict[str, Any], path: Tuple[str, ...]) -> bool:
        return self._set_config_value(priv, path, _MISSING, mark_dirty=False, cleanup_empty=True)

    def _save_ui_config(self):
        # Save current game_config to private config.yaml if game_id is present
        try:
            game_id = self._game_id
            if not game_id:
                return
            if not self._ui_dirty_paths:
                return
            priv = get_game_config_private(game_id=game_id) or {}

            # If the stitched rotation angle changed, clear cached rink geometry
            # that would become stale under a new rotation.
            stitch_path: Tuple[str, ...] = ("game", "stitching", "stitch-rotate-degrees")
            if stitch_path in self._ui_dirty_paths:
                # Helper: read a nested value from the private config.
                def _get_priv_path(path: Tuple[str, ...]):
                    cur: Any = priv
                    for key in path:
                        if not isinstance(cur, dict) or key not in cur:
                            return _MISSING
                        cur = cur[key]
                    return cur

                new_val = self._get_config_path_value(stitch_path)
                old_val = _get_priv_path(stitch_path)
                try:
                    changed = not self._values_equal(new_val, old_val)
                except Exception:
                    changed = True
                if changed:
                    for path in (
                        ("rink", "ice_contours_mask_count"),
                        ("rink", "ice_contours_mask_centroid"),
                        ("rink", "ice_contours_combined_bbox"),
                        ("rink", "scoreboard"),
                    ):
                        try:
                            self._delete_priv_path(priv, path)
                        except Exception:
                            pass

            dirty = False
            for path in list(self._ui_dirty_paths):
                current_value = self._get_config_path_value(path)
                if current_value is _MISSING:
                    dirty |= self._delete_priv_path(priv, path)
                else:
                    dirty |= self._set_priv_path(priv, path, current_value)
            if dirty:
                save_private_config(game_id=game_id, data=priv, verbose=True)
            self._ui_dirty_paths.clear()
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

            if self._plot_individual_player_tracking:
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
                if self._plot_individual_player_tracking:
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
                    # Overshoot: either multiplicative damping or begin a stop-delay
                    overshoot_delay = int(
                        self._require_breakaway_value("overshoot_stop_delay_count")
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
                        speed_adjust_box.scale_speed(
                            ratio_x=self._breakaway_detection.overshoot_scale_speed_ratio
                        )
        #
        # END Breakway detection
        #
        return current_box, online_im
