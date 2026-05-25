"""Resizable OpenCV controls for camera tuning dialogs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]
logger = logging.getLogger(__name__)


@dataclass
class SliderControl:
    name: str
    max_value: int
    value: int


@dataclass
class SliderLayout:
    row_rect: Rect
    track_rect: Rect
    knob_center: Tuple[int, int]
    knob_radius: int


class CameraControlDialog:
    """Draw scalable slider controls into an OpenCV image window."""

    _BG: Color = (28, 30, 34)
    _ROW_BG: Color = (48, 52, 59)
    _ROW_ALT: Color = (43, 47, 53)
    _TEXT: Color = (238, 241, 245)
    _MUTED: Color = (169, 176, 186)
    _TRACK: Color = (91, 99, 110)
    _ACTIVE: Color = (77, 171, 247)
    _KNOB: Color = (246, 248, 250)
    _KNOB_EDGE: Color = (19, 24, 31)

    def __init__(
        self,
        window_name: str,
        *,
        on_change: Optional[Callable[[int], None]] = None,
        initial_size: Tuple[int, int] = (900, 640),
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.window_name = window_name
        self._on_change = on_change
        self._initial_size = (max(260, int(initial_size[0])), max(180, int(initial_size[1])))
        self._position = position
        self._controls: Dict[str, SliderControl] = {}
        self._layouts: Dict[str, SliderLayout] = {}
        self._opened = False
        self._has_shown = False
        self._dragging_name: Optional[str] = None
        self._last_size = self._initial_size
        self._render_cache: Optional[np.ndarray] = None
        self._render_cache_size: Optional[Tuple[int, int]] = None
        self._render_dirty = True

    def add_slider(self, name: str, max_value: int, initial_value: int) -> None:
        max_value = max(1, int(max_value))
        value = self._clamp_value(initial_value, max_value)
        self._controls[name] = SliderControl(name=name, max_value=max_value, value=value)
        self._render_dirty = True

    def get_value(self, name: str) -> int:
        return self._controls[name].value

    def set_value(self, name: str, value: int, *, notify: bool = True) -> None:
        control = self._controls[name]
        new_value = self._clamp_value(value, control.max_value)
        if new_value == control.value:
            return
        control.value = new_value
        self._render_dirty = True
        if notify and self._on_change is not None:
            self._on_change(new_value)

    def open(self) -> None:
        if self._opened:
            return
        window_flags = (
            cv2.WINDOW_NORMAL
            | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            | getattr(cv2, "WINDOW_FREERATIO", 0)
        )
        cv2.namedWindow(self.window_name, window_flags)
        self._set_free_ratio()
        cv2.resizeWindow(self.window_name, self._initial_size[0], self._initial_size[1])
        if self._position is not None:
            try:
                cv2.moveWindow(self.window_name, self._position[0], self._position[1])
            except cv2.error as ex:
                logger.debug("Failed to position OpenCV camera control window: %s", ex)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._opened = True

    def show(self) -> None:
        if not self._opened:
            self.open()
        if self._has_shown and not self._window_visible():
            return
        width, height = self._current_window_size()
        size = (width, height)
        if self._render_dirty or self._render_cache_size != size or self._render_cache is None:
            self._render_cache = self.render(size)
            self._render_cache_size = size
            self._render_dirty = False
        cv2.imshow(self.window_name, self._render_cache)
        self._has_shown = True

    def render(self, size: Tuple[int, int]) -> np.ndarray:
        width = max(260, int(size[0]))
        height = max(180, int(size[1]))
        self._last_size = (width, height)

        canvas = np.full((height, width, 3), self._BG, dtype=np.uint8)
        count = max(1, len(self._controls))
        margin = max(8, int(min(width, height) * 0.018))
        header_h = max(28, min(48, int(height * 0.07)))
        footer_h = max(20, min(34, int(height * 0.045)))
        available_h = max(1, height - header_h - footer_h - margin)
        row_gap = max(1, min(4, int(height * 0.004)))
        content_left = margin
        content_right = width - margin
        content_w = max(1, content_right - content_left)
        min_row_h = 24
        desired_total_h = count * min_row_h + row_gap * max(0, count - 1)
        height_columns = max(1, (desired_total_h + available_h - 1) // available_h)
        columns = max(1, min(count, height_columns))
        rows_per_col = max(1, (count + columns - 1) // columns)
        col_gap = max(8, min(18, int(width * 0.012)))
        col_w = max(1, int((content_w - col_gap * max(0, columns - 1)) / columns))
        row_h = max(
            14,
            int((available_h - row_gap * max(0, rows_per_col - 1)) / rows_per_col),
        )

        self._draw_header(canvas, margin, header_h, content_right - content_left)
        top = header_h
        self._layouts.clear()
        for idx, control in enumerate(self._controls.values()):
            col_idx = idx // rows_per_col
            row_idx = idx % rows_per_col
            col_left = content_left + col_idx * (col_w + col_gap)
            col_right = min(content_right, col_left + col_w)
            row_top = top + row_idx * (row_h + row_gap)
            row_bottom = min(height - footer_h - margin, row_top + row_h)
            if row_bottom <= row_top:
                continue
            self._draw_control(
                canvas,
                control,
                idx,
                (col_left, row_top, col_right, row_bottom),
            )
        self._draw_footer(canvas, margin, height - footer_h, content_right - content_left)
        return canvas

    def _draw_header(self, canvas: np.ndarray, x: int, height: int, width: int) -> None:
        title_scale = self._fit_text_scale(self.window_name, width, max(0.45, height / 46.0))
        baseline_y = max(24, int(height * 0.68))
        cv2.putText(
            canvas,
            self.window_name,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            self._TEXT,
            max(1, int(title_scale * 2)),
            cv2.LINE_AA,
        )

    def _draw_footer(self, canvas: np.ndarray, x: int, y: int, width: int) -> None:
        text = "R reset   S save"
        scale = self._fit_text_scale(text, width, 0.55)
        cv2.putText(
            canvas,
            text,
            (x, y + max(18, int(24 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            self._MUTED,
            1,
            cv2.LINE_AA,
        )

    def _draw_control(
        self,
        canvas: np.ndarray,
        control: SliderControl,
        idx: int,
        row_rect: Rect,
    ) -> None:
        x1, y1, x2, y2 = row_rect
        row_h = y2 - y1
        width = x2 - x1
        if row_h <= 0 or width <= 0:
            return

        row_color = self._ROW_BG if idx % 2 == 0 else self._ROW_ALT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), row_color, thickness=-1)

        pad_x = max(6, int(width * 0.012))
        label = self._display_name(control.name)
        value_text = self._format_value(control)

        mid_y = (y1 + y2) // 2
        preferred_scale = max(0.28, min(0.82, row_h / 48.0))
        value_col_w = max(46, min(84, int(width * 0.09)))
        label_col_w = max(120, min(int(width * 0.42), width - value_col_w - pad_x * 5 - 90))
        track_x1 = x1 + pad_x * 2 + label_col_w
        track_x2 = x2 - pad_x * 2 - value_col_w

        if track_x2 - track_x1 < 90:
            self._draw_stacked_control(
                canvas,
                control,
                row_rect,
                label,
                value_text,
                preferred_scale,
            )
            return

        text_scale = self._fit_text_scale(label, label_col_w, preferred_scale)
        value_scale = min(
            text_scale,
            self._fit_text_scale(value_text, value_col_w, preferred_scale),
        )
        label_text = self._clip_text_to_width(label, label_col_w, text_scale)
        value_text = self._clip_text_to_width(value_text, value_col_w, value_scale)
        _, label_h = self._text_size(label_text, text_scale)
        value_width, value_h = self._text_size(value_text, value_scale)
        label_y = mid_y + label_h // 2 - 2
        value_y = mid_y + value_h // 2 - 2

        cv2.putText(
            canvas,
            label_text,
            (x1 + pad_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            self._TEXT,
            max(1, int(text_scale * 1.5)),
            cv2.LINE_AA,
        )

        track_y = mid_y
        track_h = max(3, min(8, int(row_h * 0.18)))
        knob_radius = max(4, min(9, int(row_h * 0.24)))
        track_y1 = max(y1 + 2, track_y - track_h // 2)
        track_y2 = min(y2 - 2, track_y + track_h // 2)
        track_rect = (track_x1, track_y1, track_x2, track_y2)

        knob_x = self._draw_slider_track(
            canvas,
            control,
            track_x1,
            track_x2,
            track_y,
            track_h,
            knob_radius,
        )

        cv2.putText(
            canvas,
            value_text,
            (x2 - pad_x - value_width, value_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            value_scale,
            self._MUTED,
            max(1, int(value_scale * 1.4)),
            cv2.LINE_AA,
        )

        self._layouts[control.name] = SliderLayout(
            row_rect=row_rect,
            track_rect=track_rect,
            knob_center=(knob_x, track_y),
            knob_radius=knob_radius,
        )

    def _draw_stacked_control(
        self,
        canvas: np.ndarray,
        control: SliderControl,
        row_rect: Rect,
        label: str,
        value_text: str,
        preferred_scale: float,
    ) -> None:
        x1, y1, x2, y2 = row_rect
        row_h = y2 - y1
        width = x2 - x1
        pad_x = max(4, int(width * 0.014))
        value_max_w = max(1, min(52, int(width * 0.28)))
        label_max_w = max(1, width - pad_x * 3 - value_max_w)
        text_scale = self._fit_text_scale(label, label_max_w, preferred_scale)
        value_scale = min(text_scale, self._fit_text_scale(value_text, value_max_w, text_scale))
        label_text = self._clip_text_to_width(label, label_max_w, text_scale)
        value_text = self._clip_text_to_width(value_text, value_max_w, value_scale)
        _, label_h = self._text_size(label_text, text_scale)
        value_w, _ = self._text_size(value_text, value_scale)
        text_y = y1 + max(label_h + 2, int(row_h * 0.42))

        cv2.putText(
            canvas,
            label_text,
            (x1 + pad_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            self._TEXT,
            max(1, int(text_scale * 1.5)),
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            value_text,
            (x2 - pad_x - value_w, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            value_scale,
            self._MUTED,
            max(1, int(value_scale * 1.4)),
            cv2.LINE_AA,
        )

        track_y = y1 + int(row_h * 0.74)
        track_h = max(3, min(7, int(row_h * 0.14)))
        knob_radius = max(4, min(8, int(row_h * 0.2)))
        track_x1 = x1 + pad_x
        track_x2 = x2 - pad_x
        track_y1 = max(y1 + 2, track_y - track_h // 2)
        track_y2 = min(y2 - 2, track_y + track_h // 2)
        track_rect = (track_x1, track_y1, track_x2, track_y2)
        knob_x = self._draw_slider_track(
            canvas,
            control,
            track_x1,
            track_x2,
            track_y,
            track_h,
            knob_radius,
        )
        self._layouts[control.name] = SliderLayout(
            row_rect=row_rect,
            track_rect=track_rect,
            knob_center=(knob_x, track_y),
            knob_radius=knob_radius,
        )

    def _draw_slider_track(
        self,
        canvas: np.ndarray,
        control: SliderControl,
        track_x1: int,
        track_x2: int,
        track_y: int,
        track_h: int,
        knob_radius: int,
    ) -> int:
        ratio = float(control.value) / float(control.max_value)
        knob_x = int(round(track_x1 + ratio * (track_x2 - track_x1)))
        cv2.line(
            canvas,
            (track_x1, track_y),
            (track_x2, track_y),
            self._TRACK,
            track_h,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (track_x1, track_y),
            (knob_x, track_y),
            self._ACTIVE,
            track_h,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (knob_x, track_y), knob_radius, self._KNOB_EDGE, thickness=-1)
        cv2.circle(canvas, (knob_x, track_y), max(1, knob_radius - 2), self._KNOB, thickness=-1)
        return knob_x

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            name = self._control_at(x, y)
            self._dragging_name = name
            if name is not None:
                self._set_from_point(name, x)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            name = self._dragging_name or self._control_at(x, y)
            if name is not None:
                self._set_from_point(name, x)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging_name is not None:
                self._set_from_point(self._dragging_name, x)
            self._dragging_name = None

    def _set_from_point(self, name: str, x: int) -> None:
        control = self._controls[name]
        layout = self._layouts.get(name)
        if layout is None:
            self.render(self._last_size)
            layout = self._layouts.get(name)
        if layout is None:
            return
        x1, _, x2, _ = layout.track_rect
        if x2 <= x1:
            return
        ratio = max(0.0, min(1.0, float(x - x1) / float(x2 - x1)))
        value = int(round(ratio * control.max_value))
        self.set_value(name, value)

    def _control_at(self, x: int, y: int) -> Optional[str]:
        for name, layout in self._layouts.items():
            x1, y1, x2, y2 = layout.row_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def _current_window_size(self) -> Tuple[int, int]:
        try:
            _, _, width, height = cv2.getWindowImageRect(self.window_name)
            if width > 0 and height > 0:
                return int(width), int(height)
        except cv2.error:
            return self._last_size
        except AttributeError:
            return self._last_size
        return self._last_size

    def _window_visible(self) -> bool:
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False
        except AttributeError:
            return True

    def _set_free_ratio(self) -> None:
        try:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_ASPECT_RATIO,
                getattr(cv2, "WINDOW_FREERATIO", 0),
            )
        except cv2.error as ex:
            logger.debug("Failed to set OpenCV camera control aspect ratio: %s", ex)
        except AttributeError:
            logger.debug("OpenCV build does not expose window aspect-ratio properties")

    @staticmethod
    def _clamp_value(value: int, max_value: int) -> int:
        return max(0, min(int(max_value), int(value)))

    @staticmethod
    def _display_name(name: str) -> str:
        return name.replace("_", " ")

    @staticmethod
    def _format_value(control: SliderControl) -> str:
        if control.max_value == 1:
            return "On" if control.value else "Off"
        return str(control.value)

    @staticmethod
    def _text_size(text: str, scale: float) -> Tuple[int, int]:
        (width, height), _baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(scale * 1.6))
        )
        return width, height

    @classmethod
    def _fit_text_scale(cls, text: str, max_width: int, preferred_scale: float) -> float:
        scale = max(0.22, min(1.9, float(preferred_scale)))
        width, _ = cls._text_size(text, scale)
        if width <= max_width:
            return scale
        if width <= 0:
            return scale
        return max(0.22, scale * (float(max_width) / float(width)))

    @classmethod
    def _clip_text_to_width(cls, text: str, max_width: int, scale: float) -> str:
        if max_width <= 0:
            return ""
        if cls._text_size(text, scale)[0] <= max_width:
            return text

        suffix = "..."
        if cls._text_size(suffix, scale)[0] > max_width:
            while suffix and cls._text_size(suffix, scale)[0] > max_width:
                suffix = suffix[:-1]
            return suffix

        best = suffix
        low = 0
        high = len(text)
        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid].rstrip() + suffix
            if cls._text_size(candidate, scale)[0] <= max_width:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best
