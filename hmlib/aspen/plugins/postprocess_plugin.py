import argparse
import os
from typing import Any, Dict, Optional

import torch

from hmlib.camera.cam_post_process import CamTrackPostProcessor
from hmlib.config import get_nested_value

from .base import Plugin


class CamPostProcessPlugin(Plugin):
    """
    Feeds results into CamTrackPostProcessor to render/save video and produce outputs.

    Expects in context:
      - shared.initial_args: original CLI args dict (required)
      - shared.game_config: full game config dict
      - shared.work_dir: output/results directory
      - shared.original_clip_box: optional clip box
      - data: dict from MMTrackingPlugin/PosePlugin (must include 'fps')
      - rink_profile: rink profile with combined_bbox (for play box seeding)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._postprocessor: Optional[CamTrackPostProcessor] = None

    def _ensure_postprocessor(self, context: Dict[str, Any]) -> None:
        if self._postprocessor is not None:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        data = context.get("data", {}) or {}
        game_cfg = shared.get("game_config") or {}
        init_args = shared.get("initial_args") or {}
        work_dir = shared.get("work_dir") or "."
        original_clip_box = shared.get("original_clip_box")
        device = shared.get("camera_device") or shared.get("device") or torch.device("cuda")
        video_out_device = shared.get("encoder_device") or device

        try:
            args_ns = argparse.Namespace(**init_args)
        except Exception:
            args_ns = argparse.Namespace()
        setattr(args_ns, "game_config", game_cfg)

        if not hasattr(args_ns, "cam_ignore_largest"):
            cam_ignore = get_nested_value(
                game_cfg, "rink.tracking.cam_ignore_largest", default_value=False
            )
            setattr(args_ns, "cam_ignore_largest", bool(cam_ignore))
        if not hasattr(args_ns, "crop_output_image"):
            no_crop = bool(getattr(args_ns, "no_crop", False))
            setattr(args_ns, "crop_output_image", not no_crop)
        if not hasattr(args_ns, "crop_play_box"):
            setattr(args_ns, "crop_play_box", bool(getattr(args_ns, "crop_play_box", False)))
        if not hasattr(args_ns, "plot_individual_player_tracking"):
            pit = bool(getattr(args_ns, "plot_tracking", False))
            setattr(args_ns, "plot_individual_player_tracking", pit)
        if not hasattr(args_ns, "plot_boundaries"):
            setattr(
                args_ns,
                "plot_boundaries",
                bool(getattr(args_ns, "plot_individual_player_tracking", False)),
            )
        if not hasattr(args_ns, "plot_cluster_tracking"):
            setattr(args_ns, "plot_cluster_tracking", False)
        if not hasattr(args_ns, "plot_speed"):
            setattr(args_ns, "plot_speed", False)

        fps = None
        try:
            fps = float(data.get("fps"))
        except Exception:
            fps = None
        if fps is None:
            fps = 30.0

        camera_name = None
        try:
            camera_name = get_nested_value(game_cfg, "camera.name", None)
        except Exception:
            camera_name = None

        output_video_path = getattr(args_ns, "output_video_path", None) or shared.get(
            "output_video_path"
        )
        if not output_video_path:
            output_video_path = os.path.join(work_dir, "tracking_output.mkv")

        video_out_pipeline = getattr(args_ns, "video_out_pipeline", None) or shared.get(
            "video_out_pipeline"
        )
        if video_out_pipeline is None:
            video_out_pipeline = {}

        save_frame_dir = getattr(args_ns, "save_frame_dir", None)
        no_cuda_streams = bool(getattr(args_ns, "no_cuda_streams", False))

        self._postprocessor = CamTrackPostProcessor(
            opt=args_ns,
            args=args_ns,
            device=device,
            fps=fps,
            save_dir=work_dir,
            output_video_path=output_video_path,
            save_frame_dir=save_frame_dir,
            original_clip_box=original_clip_box,
            video_out_pipeline=video_out_pipeline,
            data_type="mot",
            postprocess=True,
            video_out_device=video_out_device,
            no_cuda_streams=no_cuda_streams,
            camera_name=camera_name or "",
        )

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        postprocessor = context.get("postprocessor") or self._postprocessor
        if postprocessor is None:
            self._ensure_postprocessor(context)
            postprocessor = self._postprocessor
            if postprocessor is None:
                return {}
        data: Dict[str, Any] = context.get("data", {})
        postprocessor.process_tracking(results=data, context=context)
        return {}

    def input_keys(self):
        return {"data", "rink_profile", "shared"}

    def output_keys(self):
        return {"arena"}
