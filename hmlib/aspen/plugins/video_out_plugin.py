from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from hmlib.builder import HM
from hmlib.config import get_nested_value
from hmlib.utils.image import image_height, image_width
from hmlib.video.video_out import MAX_NEVC_VIDEO_WIDTH, VideoOutput

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
      - shared.cam_args: Namespace (optional)
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

    def is_output(self) -> bool:
        """If enabled, this node is an output."""
        return self.enabled

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._vo is not None:
            return
        shared = context.get("shared", {})
        cam_args: Optional[argparse.Namespace] = shared.get("cam_args")  # type: ignore[name-defined]
        if cam_args is None:
            # Fallback: construct a minimal args holder from config.initial_args
            cfg = shared.get("game_config") or {}
            init_args = cfg.get("initial_args") or {}
            try:
                cam_args = argparse.Namespace(**init_args)  # type: ignore[name-defined]
            except Exception:
                cam_args = argparse.Namespace()  # type: ignore[name-defined]
            cam_args.game_config = cfg
            if not hasattr(cam_args, "cam_ignore_largest"):
                cam_args.cam_ignore_largest = get_nested_value(
                    cfg, "rink.tracking.cam_ignore_largest", default_value=False
                )
            if not hasattr(cam_args, "crop_play_box"):
                cam_args.crop_play_box = bool(getattr(cam_args, "crop_play_box", False))
            if not hasattr(cam_args, "crop_output_image"):
                no_crop = bool(getattr(cam_args, "no_crop", False))
                cam_args.crop_output_image = not no_crop

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
        #   2. CLI/config-derived path on cam_args.output_video_path
        #   3. Fallback to <work_dir>/tracking_output.mkv
        out_path = self._out_path
        if not out_path:
            candidate = getattr(cam_args, "output_video_path", None)
            if not candidate:
                work_dir = context.get("work_dir") or os.path.join(os.getcwd(), "output_workdirs")
                candidate = os.path.join(str(work_dir), "tracking_output.mkv")
            out_path = candidate
        self._out_path = out_path

        # Configure NVENC / muxing backend via env var so PyNvVideoEncoder
        # can honor baseline.yaml and CLI overrides.
        backend = get_nested_value(cfg, "aspen.video_out.encoder_backend", default_value=None)
        if backend == "pyav":
            os.environ["HM_VIDEO_ENCODER_BACKEND"] = "pyav"
        elif backend == "ffmpeg":
            os.environ["HM_VIDEO_ENCODER_BACKEND"] = "ffmpeg"
        elif backend == "raw":
            os.environ["HM_VIDEO_ENCODER_BACKEND"] = "raw"
        elif backend == "auto":
            # Defer to library defaults / auto-detect; clear explicit override.
            os.environ.pop("HM_VIDEO_ENCODER_BACKEND", None)

        fps = None
        try:
            fps = float(context.get("data", {}).get("fps"))
        except Exception:
            fps = None
        if fps is None:
            fps = 30.0

        self._vo = VideoOutput(
            output_video_path=out_path,
            fps=fps,
            bit_rate=getattr(cam_args, "output_video_bit_rate", int(55e6)),
            save_frame_dir=self._save_dir,
            name="TRACKING",
            skip_final_save=self._skip_final_save,
            cache_size=self._cache,
            device=vo_dev,
            show_image=bool(getattr(cam_args, "show_image", False)),
            show_scaled=getattr(cam_args, "show_scaled", None),
            profiler=getattr(cam_args, "profiler", None),
            enable_end_zones=False,
        )
        self._vo = self._vo.to(device)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
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
