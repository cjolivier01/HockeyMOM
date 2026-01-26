from __future__ import annotations

import os
from typing import Any, Dict, Optional

from hmlib.builder import HM
from hmlib.log import logger
from hmlib.config import get_nested_value
from hmlib.audio import has_audio_stream, mux_audio_in_place
from hmlib.video.video_out import VideoOutput

from .base import Plugin


@HM.register_module()
class VideoOutPlugin(Plugin):
    """
    Plugin wrapping VideoOutput to process final overlays/cropping and save/show frames.

    Expects in context:
      - img: batched image tensor (from PlayTrackerPlugin)
      - current_box: TLBR camera boxes per frame
      - frame_ids: tensor[B]
      - play_box: TLBR arena/play box (for sizing when crop_play_box)
      - shared.game_config: full game config dict
      - shared.original_clip_box: optional clip box
      - shared.device: preferred device
      - data.fps: float

    Params:
      - output_video_path: str or None (if None, creates under CWD; skip_final_save honored)
      - cache_size: int for small internal buffering
      - video_out_device: torch.device or str
      - video_out_pipeline: dict/list pipeline; passed to VideoOutput
      - no_cuda_streams: bool
      - skip_final_save: bool
      - save_frame_dir: optional frame dump dir
      - encoder_backend: optional encoder/muxer backend selection
      - output_width: optional target output width
      - show_image: enable live preview window
      - show_scaled: optional preview scale factor
      - bit_rate: optional encoder bit rate
    """

    def __init__(
        self,
        enabled: bool = True,
        output_video_path: Optional[str] = None,
        cache_size: int = 2,
        video_out_device: Optional[str] = None,
        video_out_pipeline: Optional[Dict[str, Any]] = None,
        no_cuda_streams: bool = False,
        skip_final_save: bool = False,
        save_frame_dir: Optional[str] = None,
        encoder_backend: Optional[str] = None,
        output_width: Optional[str | int] = None,
        show_image: Optional[bool] = None,
        show_scaled: Optional[float] = None,
        bit_rate: Optional[int] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self._vo: Optional[VideoOutput] = None
        self._out_path = output_video_path
        self._cache = int(cache_size)
        self._vo_dev = video_out_device
        self._pipeline = video_out_pipeline
        self._no_cuda_streams = bool(no_cuda_streams)
        self._skip_final_save = bool(skip_final_save)
        self._save_dir = save_frame_dir
        self._encoder_backend = encoder_backend
        self._output_width = output_width
        self._show_image = bool(show_image) if show_image is not None else None
        self._show_scaled = show_scaled
        self._bit_rate = bit_rate
        self._last_shared: Optional[Dict[str, Any]] = None

    def is_output(self) -> bool:
        """If enabled, this node is an output."""
        return self.enabled

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._vo is not None:
            return
        shared = context.get("shared", {})
        img = context.get("img")
        if img is None:
            # Try fallbacks
            img = context.get("data", {}).get("original_images")
        assert img is not None, "VideoOutPlugin requires 'img' in context"

        device = shared.get("device")
        cfg = shared.get("game_config") or {}
        vo_dev = self._vo_dev if self._vo_dev is not None else device
        # Resolve the output video path:
        #   1. Explicit path passed via plugin params (_out_path)
        #   2. CLI/config-derived path in game_config.aspen.video_out.output_video_path
        #   3. Fallback to <work_dir>/tracking_output.mkv
        out_path = self._out_path
        if not out_path or "/" not in out_path:
            candidate = get_nested_value(
                cfg, "aspen.video_out.output_video_path", default_value=None
            )
            if not candidate:
                work_dir = context.get("work_dir") or os.path.join(os.getcwd(), "output_workdirs")
                candidate = os.path.join(str(work_dir), out_path or "tracking_output.mkv")
            out_path = candidate
        self._out_path = out_path

        fps = None
        try:
            fps = float(context.get("data", {}).get("fps"))
        except Exception:
            fps = None
        if fps is None:
            fps = 30.0

        bit_rate = self._bit_rate
        if bit_rate is None:
            bit_rate = get_nested_value(cfg, "aspen.video_out.bit_rate", default_value=None)
        if bit_rate is None:
            bit_rate = int(55e6)
        self._vo = VideoOutput(
            output_video_path=out_path,
            fps=fps,
            bit_rate=bit_rate,
            save_frame_dir=self._save_dir,
            name="TRACKING",
            skip_final_save=self._skip_final_save,
            cache_size=self._cache,
            device=vo_dev,
            output_width=self._output_width,
            show_image=bool(
                self._show_image
                if self._show_image is not None
                else get_nested_value(cfg, "aspen.video_out.show_image", default_value=False)
            ),
            show_scaled=(
                self._show_scaled
                if self._show_scaled is not None
                else get_nested_value(cfg, "aspen.video_out.show_scaled", default_value=None)
            ),
            profiler=shared.get("profiler", None),
            enable_end_zones=False,
            encoder_backend=self._encoder_backend,
        )
        self._vo = self._vo.to(device)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        shared = context.get("shared")
        if isinstance(shared, dict):
            self._last_shared = shared
        self._ensure_initialized(context)
        assert self._vo is not None
        # Call VideoOutput as a regular nn.Module (forward) to write frames.
        self._vo(context)
        return {}

    def finalize(self) -> None:
        """Release any UI/video resources held by VideoOutput."""
        if self._vo is not None:
            try:
                self._vo.stop()
            except Exception:
                # Finalization should be best-effort; ignore errors here.
                pass
        shared = self._last_shared or {}
        try:
            if bool(shared.get("no_audio", False)):
                return
            audio_sources = shared.get("audio_sources")
            if not audio_sources:
                return
            out_path = self._out_path
            if not out_path or not os.path.exists(out_path):
                return
            if has_audio_stream(out_path):
                return
            start_seconds = float(shared.get("audio_start_seconds") or 0.0)
            result = mux_audio_in_place(
                input_audio=audio_sources,
                video_path=out_path,
                start_seconds=start_seconds,
                shortest=True,
                keep_original=False,
            )
            if result:
                logger.info("Muxed audio into output video: %s", result)
        except Exception:
            logger.exception("Failed to mux audio into output video")

    def input_keys(self):
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "current_box",
                "frame_ids",
                "player_bottom_points",
                "player_ids",
                "rink_profile",
                "shared",
                "data",
                "game_id",
                "work_dir",
                "video_frame_cfg",
            }
        return self._input_keys

    def output_keys(self):
        return set()
