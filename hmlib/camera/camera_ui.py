"""Camera UI control panel for interactive adjustment of tracking parameters."""

from __future__ import absolute_import, division, print_function

import argparse
from typing import Any, Callable, Dict, Optional, Protocol

import cv2
import torch

from hmlib.camera.moving_box import MovingBox
from hmlib.config import get_game_config_private, save_private_config
from hmlib.tracking_utils import visualization as vis


class CameraController(Protocol):
    """Abstract interface for objects that can be controlled by the CameraUi."""

    def apply_motion_constraints(
        self,
        fast_box: Optional[MovingBox],
        follower_box: Optional[MovingBox],
        apply_fast: bool,
        apply_follower: bool,
        max_speed_x: float,
        max_speed_y: float,
        max_accel_x: float,
        max_accel_y: float,
    ) -> None:
        """Apply motion constraints to the tracking boxes."""
        ...

    def apply_breakaway_constraints(
        self,
        overshoot_delay: int,
        overshoot_speed_ratio: float,
    ) -> None:
        """Apply breakaway detection constraints."""
        ...

    def apply_cpp_playtracker_constraints(
        self,
        apply_fast: bool,
        apply_follower: bool,
        dir_delay: int,
        cancel_opp: bool,
        hyst: int,
        cooldown: int,
        postns: int,
        max_speed_x: float,
        max_speed_y: float,
        max_accel_x: float,
        max_accel_y: float,
        ov_delay: int,
        ov_scal: float,
    ) -> None:
        """Apply constraints to C++ PlayTracker if available."""
        ...


class CameraUi:
    """Interactive camera UI control panel with trackbars for real-time parameter adjustment."""

    def __init__(
        self,
        args: argparse.Namespace,
        enabled: bool = True,
    ):
        """
        Initialize the camera UI.

        Args:
            args: Command-line arguments containing game_config
            enabled: Whether UI controls are enabled
        """
        self._args = args
        self._enabled = enabled
        self._ui_window_name = "Tracker Controls"
        self._ui_color_window_name = "Tracker Controls (Color)"
        self._ui_inited = False
        self._ui_color_inited = False
        self._ui_defaults: Dict[str, Any] = {}

        if self._enabled:
            try:
                self._init_ui_controls()
            except Exception:
                self._enabled = False

    def _init_ui_controls(self) -> None:
        """Initialize main control window with trackbars."""
        cv2.namedWindow(self._ui_window_name, cv2.WINDOW_NORMAL)
        camera_cfg = self._args.game_config["rink"]["camera"]
        bkd = camera_cfg.get("breakaway_detection", {})

        def tb(name: str, maxv: int, init: int) -> None:
            cv2.createTrackbar(name, self._ui_window_name, int(init), int(maxv), lambda v: None)

        # Load initial values from config
        stop_dir_delay = int(
            camera_cfg.get("stop_on_dir_change_delay", getattr(self._args, "stop_on_dir_change_delay", 10))
        )
        cancel_stop = (
            1
            if bool(
                camera_cfg.get(
                    "cancel_stop_on_opposite_dir",
                    bool(getattr(self._args, "cancel_stop_on_opposite_dir", 1)),
                )
            )
            else 0
        )
        hyst = int(camera_cfg.get("stop_cancel_hysteresis_frames", getattr(self._args, "stop_cancel_hysteresis_frames", 2)))
        cooldown = int(camera_cfg.get("stop_delay_cooldown_frames", getattr(self._args, "stop_delay_cooldown_frames", 2)))
        ov_delay = int(bkd.get("overshoot_stop_delay_count", getattr(self._args, "overshoot_stop_delay_count", 6)))
        postns = int(bkd.get("post_nonstop_stop_delay_count", getattr(self._args, "post_nonstop_stop_delay_count", 6)))
        ov_scale = int(100 * float(bkd.get("overshoot_scale_speed_ratio", 0.7)))
        ttg = int(camera_cfg.get("time_to_dest_speed_limit_frames", 10))

        # ======== BREAKAWAY DETECTION ========
        tb("Dir Change Delay (frames)", 60, stop_dir_delay)
        tb("Cancel Opposite Dir", 1, cancel_stop)
        tb("Hysteresis (frames)", 10, hyst)
        tb("Cooldown (frames)", 30, cooldown)
        tb("Overshoot Delay (frames)", 60, ov_delay)
        tb("Post Non-Stop Delay (frames)", 60, postns)
        tb("Overshoot Speed Ratio (x100)", 200, ov_scale)
        tb("Time to Dest Limit (frames)", 120, ttg)
        # ======== BOX SELECTION ========
        tb("Apply to Fast Box", 1, 1)
        tb("Apply to Follower Box", 1, 1)
        # ======== STITCHING ========
        try:
            rot_cfg = None
            try:
                rot_cfg = self._args.game_config.get("game", {}).get("stitching", {}).get("stitch-rotate-degrees")
            except Exception:
                rot_cfg = None
            if rot_cfg is None:
                rot_cfg = getattr(self._args, "stitch_rotate_degrees", 0.0)
            rot_cfg = 0.0 if rot_cfg is None else float(rot_cfg)
            tb("Stitch Rotation (degrees)", 180, int(max(-90.0, min(90.0, rot_cfg)) + 90.0))
        except Exception:
            tb("Stitch Rotation (degrees)", 180, 90)

        # ======== MOTION CONSTRAINTS ========
        msx = int(10 * float(self._args.game_config["rink"]["camera"].get("max_speed_x", 30.0)))
        msy = int(10 * float(self._args.game_config["rink"]["camera"].get("max_speed_y", 30.0)))
        maxx = int(10 * float(self._args.game_config["rink"]["camera"].get("max_accel_x", 10.0)))
        maxy = int(10 * float(self._args.game_config["rink"]["camera"].get("max_accel_y", 10.0)))
        tb("Max Speed X (x10 multiplier)", 2000, msx)
        tb("Max Speed Y (x10 multiplier)", 2000, msy)
        tb("Max Accel X (x10 multiplier)", 1000, maxx)
        tb("Max Accel Y (x10 multiplier)", 1000, maxy)

        # Save defaults
        self._ui_defaults = {
            "Dir Change Delay (frames)": stop_dir_delay,
            "Cancel Opposite Dir": cancel_stop,
            "Hysteresis (frames)": hyst,
            "Cooldown (frames)": cooldown,
            "Overshoot Delay (frames)": ov_delay,
            "Post Non-Stop Delay (frames)": postns,
            "Overshoot Speed Ratio (x100)": ov_scale,
            "Time to Dest Limit (frames)": ttg,
            "Apply to Fast Box": 1,
            "Apply to Follower Box": 1,
            "Max Speed X (x10 multiplier)": msx,
            "Max Speed Y (x10 multiplier)": msy,
            "Max Accel X (x10 multiplier)": maxx,
            "Max Accel Y (x10 multiplier)": maxy,
            "Stitch Rotation (degrees)": (
                cv2.getTrackbarPos("Stitch Rotation (degrees)", self._ui_window_name)
                if cv2.getWindowProperty(self._ui_window_name, 0) is not None
                else 90
            ),
        }
        self._ui_inited = True

        # ---- Color controls window ----
        try:
            cv2.namedWindow(self._ui_color_window_name, cv2.WINDOW_NORMAL)
            try:
                cv2.moveWindow(self._ui_color_window_name, 520, 50)
            except Exception:
                pass

            def tb2(name: str, maxv: int, init: int) -> None:
                cv2.createTrackbar(name, self._ui_color_window_name, int(init), int(maxv), lambda v: None)

            # ======== WHITE BALANCE ========
            tb2("White Balance Kelvin Enable", 1, 0)
            tb2("White Balance Kelvin Value", 15000, 6500)
            tb2("White Balance Red (x100 multiplier)", 300, 100)
            tb2("White Balance Green (x100 multiplier)", 300, 100)
            tb2("White Balance Blue (x100 multiplier)", 300, 100)
            # ======== TONE MAPPING ========
            tb2("Brightness (x100 multiplier)", 300, 100)
            tb2("Contrast (x100 multiplier)", 300, 100)
            tb2("Gamma (x100 multiplier)", 300, 100)
            self._ui_color_inited = True
        except Exception:
            self._ui_color_inited = False

    def apply_controls(self, controller: CameraController) -> None:
        """
        Read current trackbar values and apply them to the controller.

        Args:
            controller: Object implementing CameraController protocol
        """
        if not self._enabled or not self._ui_inited:
            return

        try:
            # Read trackbars
            dir_delay = int(cv2.getTrackbarPos("Dir Change Delay (frames)", self._ui_window_name))
            cancel_opp = bool(cv2.getTrackbarPos("Cancel Opposite Dir", self._ui_window_name))
            hyst = int(cv2.getTrackbarPos("Hysteresis (frames)", self._ui_window_name))
            cooldown = int(cv2.getTrackbarPos("Cooldown (frames)", self._ui_window_name))
            ov_delay = int(cv2.getTrackbarPos("Overshoot Delay (frames)", self._ui_window_name))
            postns = int(cv2.getTrackbarPos("Post Non-Stop Delay (frames)", self._ui_window_name))
            ov_scal = cv2.getTrackbarPos("Overshoot Speed Ratio (x100)", self._ui_window_name) / 100.0
            ttg = int(cv2.getTrackbarPos("Time to Dest Limit (frames)", self._ui_window_name))

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
                rot_slider = cv2.getTrackbarPos("Stitch Rotation (degrees)", self._ui_window_name)
                rot_deg = float(rot_slider - 90)
                game_cfg = self._args.game_config.setdefault("game", {}).setdefault("stitching", {})
                game_cfg["stitch-rotate-degrees"] = rot_deg
            except Exception:
                pass

            # --- Color controls (second window) ---
            if self._ui_color_inited:
                try:
                    color_win = self._ui_color_window_name
                    wbk_enable = cv2.getTrackbarPos("White Balance Kelvin Enable", color_win)
                    kelvin = cv2.getTrackbarPos("White Balance Kelvin Value", color_win)
                    r100 = cv2.getTrackbarPos("White Balance Red (x100 multiplier)", color_win)
                    g100 = cv2.getTrackbarPos("White Balance Green (x100 multiplier)", color_win)
                    b100 = cv2.getTrackbarPos("White Balance Blue (x100 multiplier)", color_win)
                    br100 = cv2.getTrackbarPos("Brightness (x100 multiplier)", color_win)
                    ct100 = cv2.getTrackbarPos("Contrast (x100 multiplier)", color_win)
                    gm100 = cv2.getTrackbarPos("Gamma (x100 multiplier)", color_win)

                    color_cfg = camera_cfg.setdefault("color", {})
                    if int(wbk_enable) > 0:
                        color_cfg["white_balance_temp"] = f"{int(max(1000, min(40000, kelvin)))}k"
                        color_cfg.pop("white_balance", None)
                    else:
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
            apply_fast = bool(cv2.getTrackbarPos("Apply to Fast Box", self._ui_window_name))
            apply_follower = bool(cv2.getTrackbarPos("Apply to Follower Box", self._ui_window_name))
            msx = cv2.getTrackbarPos("Max Speed X (x10 multiplier)", self._ui_window_name) / 10.0
            msy = cv2.getTrackbarPos("Max Speed Y (x10 multiplier)", self._ui_window_name) / 10.0
            maxx = cv2.getTrackbarPos("Max Accel X (x10 multiplier)", self._ui_window_name) / 10.0
            maxy = cv2.getTrackbarPos("Max Accel Y (x10 multiplier)", self._ui_window_name) / 10.0

            # Delegate to controller
            controller.apply_cpp_playtracker_constraints(
                apply_fast=apply_fast,
                apply_follower=apply_follower,
                dir_delay=dir_delay,
                cancel_opp=cancel_opp,
                hyst=hyst,
                cooldown=cooldown,
                postns=postns,
                max_speed_x=msx,
                max_speed_y=msy,
                max_accel_x=maxx,
                max_accel_y=maxy,
                ov_delay=ov_delay,
                ov_scal=ov_scal,
            )

            self._handle_keyboard()
        except Exception:
            pass

    def draw_overlay(self, img: Any) -> Any:
        """Draw UI overlay on image."""
        if not self._enabled or not self._ui_inited:
            return img

        try:
            camera_cfg = self._args.game_config["rink"]["camera"]
            bkd = camera_cfg.get("breakaway_detection", {})

            try:
                rot_deg = self._args.game_config.get("game", {}).get("stitching", {}).get("stitch-rotate-degrees", 0.0)
                rot_text = f" Rot={float(rot_deg):+.1f}deg"
            except Exception:
                rot_text = ""

            text = (
                f"DirDelay={camera_cfg.get('stop_on_dir_change_delay', 0)} "
                f"Cancel={int(bool(camera_cfg.get('cancel_stop_on_opposite_dir', 0)))} "
                f"Hyst={camera_cfg.get('stop_cancel_hysteresis_frames', 0)} "
                f"CD={camera_cfg.get('stop_delay_cooldown_frames', 0)} "
                f"OvDelay={bkd.get('overshoot_stop_delay_count', 0)} "
                f"PostNS={bkd.get('post_nonstop_stop_delay_count', 0)} "
                f"OvScale={bkd.get('overshoot_scale_speed_ratio', 0.0):.2f}"
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

            img = vis.plot_text(
                img,
                "[R]eset  [S]ave",
                (20, 70),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                thickness=2,
            )
        except Exception:
            pass

        return img

    def _handle_keyboard(self) -> None:
        """Handle keyboard input for UI controls."""
        if not self._enabled or not self._ui_inited:
            return

        try:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('r') or k == ord('R'):
                self._reset_controls()
            elif k == ord('s') or k == ord('S'):
                self._save_config()
        except Exception:
            pass

    def _reset_controls(self) -> None:
        """Reset all trackbars to their default values."""
        if not self._enabled or not self._ui_inited:
            return

        try:
            for name, val in self._ui_defaults.items():
                cv2.setTrackbarPos(name, self._ui_window_name, int(val))
        except Exception:
            pass

    def _save_config(self) -> None:
        """Save current configuration to persistent storage."""
        try:
            game_id = getattr(self._args, "game_id", None)
            if not game_id:
                return

            priv = get_game_config_private(game_id=game_id) or {}
            priv.setdefault("rink", {}).setdefault("camera", {})
            camera_cfg = self._args.game_config["rink"]["camera"]
            priv["rink"]["camera"] = camera_cfg

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

    @property
    def enabled(self) -> bool:
        """Whether UI is enabled."""
        return self._enabled
