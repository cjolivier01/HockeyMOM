from __future__ import annotations

from typing import Any, Dict, Optional

from hmlib.builder import HM
from hmlib.config import get_nested_value, normalize_runtime_config
from hmlib.ui.shower import Shower

from .base import Plugin


@HM.register_module()
class VideoPreviewPlugin(Plugin):
    """Async, lossy preview sink for local display, headless preview, and RTMP."""

    disable_in_cuda_graph_pipeline = True

    def __init__(
        self,
        enabled: bool = True,
        cache_size: int = 1,
        preview_fps: Optional[float] = None,
        show_image: Optional[bool] = None,
        show_scaled: Optional[float] = None,
        show_youtube: Optional[bool] = None,
        youtube_stream_url: Optional[str] = None,
        youtube_stream_key: Optional[str] = None,
        headless_preview_host: Optional[str] = None,
        headless_preview_port: Optional[int] = None,
        always_stream: Optional[bool] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self._cache_size = max(1, int(cache_size or 1))
        self._preview_fps = float(preview_fps) if preview_fps not in (None, 0) else None
        self._show_image = bool(show_image) if show_image is not None else None
        self._show_scaled = show_scaled
        self._show_youtube = bool(show_youtube) if show_youtube is not None else None
        self._youtube_stream_url = youtube_stream_url
        self._youtube_stream_key = youtube_stream_key
        self._headless_preview_host = headless_preview_host
        self._headless_preview_port = headless_preview_port
        self._always_stream = bool(always_stream) if always_stream is not None else None
        self._shower: Optional[Shower] = None
        self._progress_bar_callback_installed = False

    def _resolve_preview_enabled(self, cfg: Dict[str, Any]) -> tuple[bool, bool]:
        show_image = bool(
            self._show_image
            if self._show_image is not None
            else get_nested_value(
                cfg,
                "video_out.show_image",
                default_value=get_nested_value(cfg, "aspen.video_out.show_image", False),
            )
        )
        show_youtube = bool(
            self._show_youtube
            if self._show_youtube is not None
            else get_nested_value(
                cfg,
                "video_out.show_youtube",
                default_value=get_nested_value(cfg, "aspen.video_out.show_youtube", False),
            )
        )
        return show_image, show_youtube

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._shower is not None:
            return

        shared = context.get("shared", {})
        cfg = shared.get("game_config") if isinstance(shared, dict) else {}
        cfg = cfg or {}
        normalize_runtime_config(cfg)

        show_image, show_youtube = self._resolve_preview_enabled(cfg)
        if not show_image and not show_youtube:
            return

        fps_val = context.get("fps")
        if fps_val is None and isinstance(shared, dict):
            fps_val = shared.get("fps")
        fps = self._preview_fps
        if fps is None:
            cfg_preview_fps = get_nested_value(
                cfg,
                "video_out.preview_fps",
                default_value=get_nested_value(cfg, "aspen.video_out.preview_fps", None),
            )
            if cfg_preview_fps is not None:
                try:
                    fps = float(cfg_preview_fps)
                except Exception:
                    fps = None
        if fps is None and fps_val is not None:
            try:
                fps = float(fps_val)
            except Exception:
                fps = None
        if fps is not None and fps <= 0:
            fps = None

        label = "Preview"
        if isinstance(shared, dict) and shared.get("game_id"):
            label = f"{shared['game_id']} preview"

        headless_preview_host = (
            self._headless_preview_host
            if self._headless_preview_host is not None
            else get_nested_value(
                cfg,
                "video_out.headless_preview_host",
                default_value=get_nested_value(
                    cfg, "aspen.video_out.headless_preview_host", "0.0.0.0"
                ),
            )
        )
        headless_preview_port = (
            self._headless_preview_port
            if self._headless_preview_port is not None
            else get_nested_value(
                cfg,
                "video_out.headless_preview_port",
                default_value=get_nested_value(cfg, "aspen.video_out.headless_preview_port", 0),
            )
        )

        self._shower = Shower(
            label=label,
            show_scaled=(
                self._show_scaled
                if self._show_scaled is not None
                else get_nested_value(
                    cfg,
                    "video_out.show_scaled",
                    default_value=get_nested_value(cfg, "aspen.video_out.show_scaled", None),
                )
            ),
            max_size=self._cache_size,
            fps=fps,
            profiler=self._profiler,
            skip_frame_when_full=True,
            drop_oldest_when_full=True,
            enable_local_display=show_image,
            show_youtube=show_youtube,
            youtube_stream_url=(
                self._youtube_stream_url
                if self._youtube_stream_url is not None
                else get_nested_value(
                    cfg,
                    "video_out.youtube_stream_url",
                    default_value=get_nested_value(cfg, "aspen.video_out.youtube_stream_url", None),
                )
            ),
            youtube_stream_key=(
                self._youtube_stream_key
                if self._youtube_stream_key is not None
                else get_nested_value(
                    cfg,
                    "video_out.youtube_stream_key",
                    default_value=get_nested_value(cfg, "aspen.video_out.youtube_stream_key", None),
                )
            ),
            headless_preview_host=str(headless_preview_host or "0.0.0.0"),
            headless_preview_port=int(headless_preview_port or 0),
            always_stream=bool(
                self._always_stream
                if self._always_stream is not None
                else get_nested_value(
                    cfg,
                    "video_out.always_stream",
                    default_value=get_nested_value(cfg, "aspen.video_out.always_stream", False),
                )
            ),
        )

        progress_bar = shared.get("progress_bar") if isinstance(shared, dict) else None
        if progress_bar is not None and not self._progress_bar_callback_installed:
            progress_bar.add_table_callback(self._shower.update_progress_table)
            self._progress_bar_callback_installed = True

    def input_keys(self):
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "shared",
                "fps",
            }
        return self._input_keys

    def is_output(self) -> bool:
        return self.enabled

    def output_keys(self):
        return set()

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        with self.profile_scope("video_preview.forward"):
            with self.profile_scope("video_preview.ensure_initialized"):
                self._ensure_initialized(context)
            if self._shower is None:
                return {}
            img = context.get("img")
            if img is None:
                return {}
            with self.profile_scope("video_preview.enqueue"):
                if getattr(img, "ndim", None) == 4:
                    for frame in img:
                        self._shower.show(frame, clone=True)
                else:
                    self._shower.show(img, clone=True)
        return {}

    def finalize(self) -> None:
        if self._shower is not None:
            with self.profile_scope("video_preview.finalize"):
                self._shower.close()
            self._shower = None


__all__ = ["VideoPreviewPlugin"]
