"""Rust runtime controls for the stitching-only Aspen workflow."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Set, Tuple

from hmlib.builder import HM
from hmlib.camera.hm_ui_bridge import HmUiProcess
from hmlib.config import (
    get_config,
    get_game_config_private,
    get_nested_value,
    normalize_runtime_config,
    save_private_config,
)
from hmlib.log import logger

from .base import Plugin

_MISSING = object()
_EXPOSURE_EV_X10_MIN = -40
_EXPOSURE_EV_X10_MAX = 40
_EXPOSURE_EV_X10_SLIDER_MAX = _EXPOSURE_EV_X10_MAX - _EXPOSURE_EV_X10_MIN
_EXPOSURE_EV_X10_SLIDER_ZERO = -_EXPOSURE_EV_X10_MIN


@HM.register_module()
class StitchUiPlugin(Plugin):
    """Expose only controls that affect the stitched image."""

    disable_in_cuda_graph_pipeline = True

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._process: Optional[HmUiProcess] = None
        self._game_config: Dict[str, Any] = {}
        self._open_config: Dict[str, Any] = {}
        self._system_config: Dict[str, Any] = {}
        self._game_id: Optional[str] = None
        self._dirty_paths: Set[Tuple[str, ...]] = set()
        self._shared: Optional[Dict[str, Any]] = None
        self._active = False

    @staticmethod
    def _stitch_deg_to_slider(degrees: Any) -> int:
        value = max(-90.0, min(90.0, float(degrees or 0.0)))
        return int(max(0, min(180, round(90.0 - value))))

    @staticmethod
    def _slider_to_stitch_deg(position: int) -> float:
        return float(90 - int(position))

    @staticmethod
    def _exposure_ev_to_slider(value: Any) -> int:
        try:
            ev_x10 = int(round(float(value) * 10.0))
        except (TypeError, ValueError):
            ev_x10 = 0
        ev_x10 = max(_EXPOSURE_EV_X10_MIN, min(_EXPOSURE_EV_X10_MAX, ev_x10))
        return ev_x10 - _EXPOSURE_EV_X10_MIN

    @staticmethod
    def _slider_to_exposure_ev(position: int) -> float:
        position = max(0, min(_EXPOSURE_EV_X10_SLIDER_MAX, int(position)))
        return float(position + _EXPOSURE_EV_X10_MIN) / 10.0

    @classmethod
    def _color_defaults(cls, config: Dict[str, Any], path: str) -> Dict[str, int]:
        color_cfg = get_nested_value(config, path, {}) or {}
        if not isinstance(color_cfg, dict):
            color_cfg = {}
        defaults = {
            "White_Balance_Kelvin_Enable": 0,
            "White_Balance_Kelvin_Temperature": 6500,
            "White_Balance_Red_Gain_x100": 100,
            "White_Balance_Green_Gain_x100": 100,
            "White_Balance_Blue_Gain_x100": 100,
            "Brightness_Multiplier_x100": 100,
            "Exposure_EV_x10": _EXPOSURE_EV_X10_SLIDER_ZERO,
            "Contrast_Multiplier_x100": 100,
            "Gamma_Multiplier_x100": 100,
        }
        temperature = color_cfg.get("white_balance_temp")
        gains = color_cfg.get("white_balance")
        if temperature is not None:
            try:
                defaults["White_Balance_Kelvin_Enable"] = 1
                defaults["White_Balance_Kelvin_Temperature"] = int(
                    max(1000, min(15000, float(str(temperature).lower().removesuffix("k"))))
                )
            except (TypeError, ValueError):
                logger.warning("Invalid white balance temperature in %s: %r", path, temperature)
        elif isinstance(gains, (list, tuple)) and len(gains) == 3:
            try:
                blue, green, red = gains
                defaults["White_Balance_Red_Gain_x100"] = int(float(red) * 100.0)
                defaults["White_Balance_Green_Gain_x100"] = int(float(green) * 100.0)
                defaults["White_Balance_Blue_Gain_x100"] = int(float(blue) * 100.0)
            except (TypeError, ValueError):
                logger.warning("Invalid white balance gains in %s: %r", path, gains)
        defaults["Exposure_EV_x10"] = cls._exposure_ev_to_slider(color_cfg.get("exposure_ev", 0.0))
        for key, control in (
            ("brightness", "Brightness_Multiplier_x100"),
            ("contrast", "Contrast_Multiplier_x100"),
            ("gamma", "Gamma_Multiplier_x100"),
        ):
            try:
                defaults[control] = int(float(color_cfg.get(key, 1.0)) * 100.0)
            except (TypeError, ValueError):
                logger.warning("Invalid %s in %s: %r", key, path, color_cfg.get(key))
        return defaults

    def _add_color_window(self, window_name: str, defaults: Dict[str, int]) -> None:
        assert self._process is not None
        self._process.add_window(window_name)
        maximums = {
            "White_Balance_Kelvin_Enable": 1,
            "White_Balance_Kelvin_Temperature": 15000,
            "White_Balance_Red_Gain_x100": 300,
            "White_Balance_Green_Gain_x100": 300,
            "White_Balance_Blue_Gain_x100": 300,
            "Brightness_Multiplier_x100": 300,
            "Exposure_EV_x10": _EXPOSURE_EV_X10_SLIDER_MAX,
            "Contrast_Multiplier_x100": 300,
            "Gamma_Multiplier_x100": 300,
        }
        for name, value in defaults.items():
            self._process.add_slider(window_name, name, maximums[name], value)

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._process is not None or self._active:
            return
        shared = context.get("shared", {})
        if not isinstance(shared, dict):
            return
        self._active = bool(shared.get("camera_ui"))
        if not self._active:
            return

        self._shared = shared
        self._game_config = shared.get("game_config") or {}
        self._open_config = copy.deepcopy(self._game_config)
        self._game_id = shared.get("game_id")
        self._system_config = copy.deepcopy(self._game_config)
        if self._game_id:
            try:
                self._system_config = get_config(
                    game_id=self._game_id,
                    ignore_private_config=True,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as ex:
                logger.warning(
                    "Could not load system stitch UI defaults for %s; using open-time values: %s",
                    self._game_id,
                    ex,
                )

        title = f"HM Stitch UI - {self._game_id}" if self._game_id else "HM Stitch UI"
        self._process = HmUiProcess(title=title, preview_names=("Stitched",))
        alignment_window = "Stitch Alignment"
        self._process.add_window(alignment_window)
        rotation = get_nested_value(
            self._game_config,
            "stitching.post_stitch_rotate_degrees",
            0.0,
        )
        self._process.add_slider(
            alignment_window,
            "Stitch_Rotate_Degrees",
            180,
            self._stitch_deg_to_slider(rotation),
        )

        color_windows = {
            "Tracker Controls (Stitched Color)": "rink.camera.color",
            "Tracker Controls (Left Color)": "stitching.left.color",
            "Tracker Controls (Right Color)": "stitching.right.color",
        }
        system_defaults: Dict[str, Dict[str, int]] = {
            alignment_window: {
                "Stitch_Rotate_Degrees": self._stitch_deg_to_slider(
                    get_nested_value(
                        self._system_config,
                        "stitching.post_stitch_rotate_degrees",
                        0.0,
                    )
                ),
            }
        }
        for window_name, path in color_windows.items():
            defaults = self._color_defaults(self._game_config, path)
            system_defaults[window_name] = self._color_defaults(self._system_config, path)
            self._add_color_window(window_name, defaults)
        self._process.set_system_defaults(system_defaults)
        shared["hm_ui_process"] = self._process
        shared["hm_ui_preview_active"] = True

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left is _MISSING or right is _MISSING:
            return left is right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) < 1e-6
        return left == right

    @staticmethod
    def _path_value(config: Dict[str, Any], path: Tuple[str, ...]) -> Any:
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return _MISSING
            current = current[key]
        return current

    @classmethod
    def _set_path(cls, config: Dict[str, Any], path: Tuple[str, ...], value: Any) -> bool:
        current: Any = config
        parents = []
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                if value is _MISSING:
                    return False
                current[key] = {}
            parents.append((current, key))
            current = current[key]
        leaf = path[-1]
        previous = current.get(leaf, _MISSING)
        if value is _MISSING:
            if leaf not in current:
                return False
            del current[leaf]
            while parents:
                parent, key = parents.pop()
                if isinstance(parent.get(key), dict) and not parent[key]:
                    del parent[key]
                else:
                    break
            return True
        if cls._values_equal(previous, value):
            return False
        current[leaf] = copy.deepcopy(value)
        return True

    def _set_runtime_path(self, path: Tuple[str, ...], value: Any) -> None:
        if self._set_path(self._game_config, path, value):
            self._dirty_paths.add(path)

    def _apply_color_window(self, window_name: str, prefix: Tuple[str, ...]) -> None:
        assert self._process is not None

        def value(name: str) -> int:
            assert self._process is not None
            return self._process.get_value(window_name, name, poll=False)

        if value("White_Balance_Kelvin_Enable"):
            kelvin = max(1000, min(15000, value("White_Balance_Kelvin_Temperature")))
            self._set_runtime_path(prefix + ("white_balance_temp",), f"{kelvin}k")
            self._set_runtime_path(prefix + ("white_balance",), _MISSING)
        else:
            gains = [
                max(1, value("White_Balance_Blue_Gain_x100")) / 100.0,
                max(1, value("White_Balance_Green_Gain_x100")) / 100.0,
                max(1, value("White_Balance_Red_Gain_x100")) / 100.0,
            ]
            self._set_runtime_path(prefix + ("white_balance",), gains)
            self._set_runtime_path(prefix + ("white_balance_temp",), _MISSING)
        self._set_runtime_path(
            prefix + ("brightness",),
            max(1, value("Brightness_Multiplier_x100")) / 100.0,
        )
        self._set_runtime_path(
            prefix + ("exposure_ev",),
            self._slider_to_exposure_ev(value("Exposure_EV_x10")),
        )
        self._set_runtime_path(
            prefix + ("contrast",),
            max(1, value("Contrast_Multiplier_x100")) / 100.0,
        )
        self._set_runtime_path(
            prefix + ("gamma",),
            max(1, value("Gamma_Multiplier_x100")) / 100.0,
        )

    def _apply_controls(self) -> None:
        assert self._process is not None
        rotation = self._slider_to_stitch_deg(
            self._process.get_value("Stitch Alignment", "Stitch_Rotate_Degrees", poll=False)
        )
        self._set_runtime_path(("stitching", "post_stitch_rotate_degrees"), rotation)
        self._apply_color_window(
            "Tracker Controls (Stitched Color)",
            ("rink", "camera", "color"),
        )
        self._apply_color_window(
            "Tracker Controls (Left Color)",
            ("stitching", "left", "color"),
        )
        self._apply_color_window(
            "Tracker Controls (Right Color)",
            ("stitching", "right", "color"),
        )

    def _save(self) -> None:
        if not self._game_id:
            return
        private_config = get_game_config_private(game_id=self._game_id) or {}
        normalize_runtime_config(private_config)
        rotation_path = ("stitching", "post_stitch_rotate_degrees")
        if rotation_path in self._dirty_paths:
            for path in (
                ("rink", "ice_contours_mask_count"),
                ("rink", "ice_contours_mask_centroid"),
                ("rink", "ice_contours_combined_bbox"),
                ("rink", "scoreboard", "perspective_polygon"),
            ):
                self._set_path(private_config, path, _MISSING)

        changed = False
        for path in self._managed_paths() | self._dirty_paths:
            current = self._path_value(self._game_config, path)
            system = self._path_value(self._system_config, path)
            value = (
                _MISSING if current is _MISSING or self._values_equal(current, system) else current
            )
            changed |= self._set_path(private_config, path, value)
        if changed:
            save_private_config(self._game_id, private_config, verbose=True)
        self._dirty_paths.clear()

    @staticmethod
    def _managed_paths() -> Set[Tuple[str, ...]]:
        rotation_path = ("stitching", "post_stitch_rotate_degrees")
        color_keys = {
            "white_balance",
            "white_balance_temp",
            "brightness",
            "exposure_ev",
            "contrast",
            "gamma",
        }
        managed_paths: Set[Tuple[str, ...]] = {rotation_path}
        for prefix in (
            ("rink", "camera", "color"),
            ("stitching", "left", "color"),
            ("stitching", "right", "color"),
        ):
            managed_paths.update(prefix + (key,) for key in color_keys)
        return managed_paths

    def _restore_managed_config(self, source: Dict[str, Any]) -> None:
        for path in self._managed_paths():
            self._set_runtime_path(path, self._path_value(source, path))

    def _disable_ui(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None
        if self._shared is not None:
            self._shared["hm_ui_process"] = None
            self._shared["hm_ui_preview_active"] = False

    def input_keys(self) -> set[str]:
        return {"img", "shared"}

    def output_keys(self) -> set[str]:
        return {"img"}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {"img": context.get("img")}
        img = context.get("img")
        try:
            self._ensure_initialized(context)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            logger.exception("Failed to initialize stitch camera UI; disabling it")
            self._disable_ui()
            return {"img": img}
        if self._process is None:
            return {"img": img}
        try:
            controls_changed = self._process.poll()
            if self._process.closed:
                self._disable_ui()
                return {"img": img}
            actions = self._process.consume_actions(poll=False)
            if "reset-system" in actions:
                self._restore_managed_config(self._system_config)
            elif "reset-open" in actions:
                self._restore_managed_config(self._open_config)
            elif controls_changed:
                self._apply_controls()
            if "save" in actions:
                self._apply_controls()
                self._save()
            if img is not None:
                self._process.publish_preview(img, name="Stitched")
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            logger.exception("Stitch camera UI failed; disabling it for the rest of this run")
            self._disable_ui()
        return {"img": img}

    def finalize(self) -> None:
        self._disable_ui()


__all__ = ["StitchUiPlugin"]
