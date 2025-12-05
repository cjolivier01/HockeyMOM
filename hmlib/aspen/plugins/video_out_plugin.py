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
        self._final_w: Optional[int] = None
        self._final_h: Optional[int] = None

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

        # Decide final output size
        H = int(image_height(img))
        W = int(image_width(img))
        ar = 16.0 / 9.0
        play_box = context.get("play_box")

        if cam_args.no_crop:
            final_w = W
            final_h = H
        elif getattr(cam_args, "crop_play_box", False):
            if getattr(cam_args, "crop_output_image", True):
                final_h = H if play_box is None else int((play_box[3] - play_box[1]))
                final_w = int(final_h * ar)
                if final_w > MAX_NEVC_VIDEO_WIDTH:
                    final_w = MAX_NEVC_VIDEO_WIDTH
                    final_h = int(final_w / ar)
            else:
                final_h = H if play_box is None else int((play_box[3] - play_box[1]))
                final_w = W if play_box is None else int((play_box[2] - play_box[0]))
        else:
            if getattr(cam_args, "crop_output_image", True):
                final_h = H
                final_w = int(final_h * ar)
                if final_w > MAX_NEVC_VIDEO_WIDTH:
                    final_w = MAX_NEVC_VIDEO_WIDTH
                    final_h = int(final_w / ar)
            else:
                final_h = H
                final_w = W

        self._final_w = int(final_w)
        self._final_h = int(final_h)

        device = shared.get("device")
        vo_dev = self._vo_dev if self._vo_dev is not None else device
        if not self._out_path:
            self._out_path = context.get("work_dir")
        out_path = self._out_path or os.path.join(os.getcwd(), "tracking_output.mkv")
        fps = None
        try:
            fps = float(context.get("data", {}).get("fps"))
        except Exception:
            fps = None
        if fps is None:
            fps = 30.0

        self._vo = VideoOutput(
            output_video_path=out_path,
            output_frame_width=self._final_w,
            output_frame_height=self._final_h,
            fps=fps,
            bit_rate=getattr(cam_args, "output_video_bit_rate", int(55e6)),
            save_frame_dir=self._save_dir,
            name="TRACKING",
            skip_final_save=self._skip_final_save,
            original_clip_box=shared.get("original_clip_box"),
            cache_size=self._cache,
            no_cuda_streams=self._no_cuda_streams,
            device=vo_dev,
            show_image=bool(getattr(cam_args, "show_image", False)),
            show_scaled=getattr(cam_args, "show_scaled", None),
            profiler=getattr(cam_args, "profiler", None),
            game_config=None,
            enable_end_zones=False,
        )
        self._vo = self._vo.to(device)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        # We expect img/current_box/frame_ids present from prior trunk
        results = {
            "img": context.get("img"),
            "current_box": context.get("current_box"),
            "frame_ids": context.get("frame_ids"),
            "game_id": context.get("game_id"),
        }
        # Preserve optional extras for overlays
        for k in ("player_bottom_points", "player_ids", "rink_profile", "pano_size_wh"):
            if k in context:
                results[k] = context[k]

        self._ensure_initialized(context)
        assert self._vo is not None
        # Call VideoOutput as a regular nn.Module (forward) to write frames.
        self._vo(results)
        return {}

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
            }
        return self._input_keys

    def output_keys(self):
        return set()
