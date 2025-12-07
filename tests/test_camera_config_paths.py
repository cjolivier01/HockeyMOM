from __future__ import annotations

from typing import Any, Dict

import torch

from hmlib.camera.camera import HockeyMOM
from hmlib.camera.play_tracker import PlayTracker


def _base_game_config() -> Dict[str, Any]:
    return {
        "rink": {
            "camera": {
                "pan_smoothing_alpha": 0.18,
                "sticky_size_ratio_to_frame_width": 0.5,
                "sticky_translation_gaussian_mult": 1.0,
                "unsticky_translation_size_ratio": 0.2,
                "follower_box_scale_width": 1.0,
                "follower_box_scale_height": 1.0,
                "time_to_dest_speed_limit_frames": 10,
                "fixed_edge_scaling_factor": 1.0,
                "fixed_edge_rotation_angle": 0.0,
                "stop_on_dir_change_delay": 10,
                "cancel_stop_on_opposite_dir": True,
                "stop_cancel_hysteresis_frames": 2,
                "stop_delay_cooldown_frames": 2,
                "max_speed_ratio_x": 1.0,
                "max_speed_ratio_y": 1.0,
                "max_accel_ratio_x": 1.0,
                "max_accel_ratio_y": 1.0,
                "color": {
                    "white_balance": [1.0, 1.0, 1.0],
                    "brightness": 1.0,
                    "contrast": 1.0,
                    "gamma": 1.0,
                },
                "breakaway_detection": {
                    "min_considered_group_velocity": 1.0,
                    "group_ratio_threshold": 0.5,
                    "group_velocity_speed_ratio": 0.5,
                    "scale_speed_constraints": 1.0,
                    "nonstop_delay_count": 3,
                    "overshoot_scale_speed_ratio": 1.0,
                    "overshoot_stop_delay_count": 3,
                    "post_nonstop_stop_delay_count": 2,
                },
            },
            "tracking": {"cam_ignore_largest": False},
        }
    }


def should_store_camera_color_only_under_rink_level():
    cfg = _base_game_config()
    hockey_mom = HockeyMOM(
        image_width=1920,
        image_height=1080,
        fps=30.0,
        device=torch.device("cpu"),
        camera_name="GoPro",
    )
    play_box = torch.tensor([0.0, 0.0, 1920.0, 1080.0], dtype=torch.float32)
    tracker = PlayTracker(
        hockey_mom=hockey_mom,
        play_box=play_box,
        device=torch.device("cpu"),
        original_clip_box=None,
        progress_bar=None,
        game_config=cfg,
        cam_ignore_largest=False,
        no_wide_start=False,
        track_ids=None,
        debug_play_tracker=False,
        plot_individual_player_tracking=False,
        plot_boundaries=False,
        plot_all_detections=None,
        plot_trajectories=False,
        plot_speed=False,
        plot_jersey_numbers=False,
        plot_actions=False,
        plot_moving_boxes=False,
        camera_ui=1,
        camera_controller="rule",
        camera_model=None,
        camera_window=8,
        force_stitching=False,
        stitch_rotation_controller=None,
        cluster_centroids=None,
        cpp_boxes=False,
        cpp_playtracker=False,
        plot_cluster_tracking=False,
    )

    # Simulate UI updating brightness and white balance temperature via sliders.
    tracker._set_ui_color_value("brightness", 1.04)
    tracker._set_ui_color_value("white_balance_temp", "8536k")

    game_cfg = tracker._game_config
    rink_cam = game_cfg.get("rink", {}).get("camera", {})
    assert isinstance(rink_cam, dict)
    color = rink_cam.get("color", {})
    assert color.get("brightness") == 1.04
    assert color.get("white_balance_temp") == "8536k"

    # Root-level camera block should not be created/modified by the UI updates.
    root_cam = game_cfg.get("camera")
    assert not isinstance(root_cam, dict) or "color" not in root_cam

