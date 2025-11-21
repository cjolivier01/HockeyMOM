from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import torch

from hmlib.bbox.box_functions import center, height, make_box_at_center, width
from hmlib.builder import HM
from hmlib.camera.cam_post_process import DefaultArguments
from hmlib.camera.camera import HockeyMOM
from hmlib.camera.play_tracker import PlayTracker as _PlayTracker
from hmlib.utils.image import image_height, image_width

from .base import Trunk


@HM.register_module()
class PlayTrackerTrunk(Trunk):
    """
    Trunk wrapping the PlayTracker to compute per-frame camera boxes and
    attach visualization images.

    Expects context keys (minimal_context ready):
      - data: dict with 'data_samples' and 'original_images'
      - shared.game_config: full game config dict
      - shared.original_clip_box: optional clip box
      - shared.device: torch.device
      - shared.cam_args: DefaultArguments (optional; created if missing)

    Produces:
      - img: image tensor batch after overlays (StreamCheckpoint)
      - current_box: TLBR camera boxes per frame (tensor[B,4])
      - frame_ids: tensor[B]
      - player_bottom_points, player_ids, rink_profile (when available)
      - play_box: current arena/play region tensor (TLBR)
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._hockey_mom: Optional[HockeyMOM] = None
        self._play_tracker: Optional[_PlayTracker] = None
        self._device: Optional[torch.device] = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._args: Optional[DefaultArguments] = None
        self._clip_box: Optional[torch.Tensor] = None

    # region helpers
    @staticmethod
    def _calc_seed_play_box(results: Dict[str, Any]) -> torch.Tensor:
        # Use rink_profile combined bbox if present; scale a bit and clamp
        track_container = results["data_samples"]
        vds = track_container.video_data_samples[0]
        prof = getattr(vds, "metainfo", {}).get("rink_profile", None)
        image = results["original_images"]
        if prof and "combined_bbox" in prof:
            play_box = torch.tensor(prof["combined_bbox"], dtype=torch.int64, device=image.device)
            ww, hh = width(play_box), height(play_box)
            cc = center(play_box)
            play_box = make_box_at_center(cc, ww * 1.3, hh * 1.3)
        else:
            # fallback to whole frame
            play_box = torch.tensor(
                [
                    0,
                    0,
                    image_width(image),
                    image_height(image),
                ],
                dtype=torch.int64,
                device=image.device,
            )
        # Clamp to image bounds
        zero = torch.zeros((), dtype=torch.int64, device=image.device)
        play_box[0] = torch.max(zero, play_box[0].long())
        play_box[1] = torch.max(zero, play_box[1].long())
        play_box[2] = torch.min(
            torch.tensor(image_width(image), dtype=torch.int64, device=image.device),
            play_box[2].long(),
        )
        play_box[3] = torch.min(
            torch.tensor(image_height(image), dtype=torch.int64, device=image.device),
            play_box[3].long(),
        )
        return play_box

    def _ensure_initialized(self, context: Dict[str, Any], results: Dict[str, Any]) -> None:
        if self._play_tracker is not None:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        game_cfg = shared.get("game_config") or {}
        self._clip_box = shared.get("original_clip_box")
        # Prefer provided device; fallback to image's device
        dev = context.get("device")
        if dev is None and isinstance(results.get("original_images"), torch.Tensor):
            dev = results["original_images"].device
        self._device = dev if isinstance(dev, torch.device) else torch.device(str(dev) if dev else "cuda")
        # Build cam args if not supplied
        cam_args = shared.get("cam_args")
        if cam_args is None:
            init_args = game_cfg.get("initial_args") or {}
            try:
                opt_ns = argparse.Namespace(**init_args)
            except Exception:
                opt_ns = argparse.Namespace()
            cam_args = DefaultArguments(
                game_config=game_cfg,
                basic_debugging=int(getattr(opt_ns, "debug", 0)),
                output_video_path=getattr(opt_ns, "output_file", None),
                opts=opt_ns,
            )
        self._args = cam_args

        # Determine video geometry and fps
        ori = results["original_images"]
        H = int(ori.shape[-2])
        W = int(ori.shape[-1])
        fps = None
        try:
            fps = float(context.get("data", {}).get("fps"))
        except Exception:
            fps = None
        if fps is None:
            fps = 30.0

        cam_name = None
        try:
            cam_name = game_cfg.get("camera", {}).get("name")
        except Exception:
            pass

        self._hockey_mom = HockeyMOM(
            image_width=W,
            image_height=H,
            fps=fps,
            device=self._device,
            camera_name=cam_name,
        )
        seed_box = self._calc_seed_play_box(results)
        self._play_tracker = _PlayTracker(
            hockey_mom=self._hockey_mom,
            play_box=seed_box,
            device=self._device,
            original_clip_box=self._clip_box,
            progress_bar=None,
            args=self._args,
        )
        self._play_tracker.eval()

    # endregion

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        data: Dict[str, Any] = context.get("data", {})
        # Build results dictionary expected by PlayTracker
        results: Dict[str, Any] = {
            "data_samples": data.get("data_samples"),
            "original_images": data.get("original_images"),
        }
        # Optional per-frame annotations propagated if present
        for k in ("jersey_results", "action_results"):
            if k in data:
                results[k] = data[k]

        self._ensure_initialized(context, results)
        assert self._play_tracker is not None
        with torch.no_grad():
            out = self._play_tracker.forward(results=results)
        # Surface current play_box for downstream video out sizing
        try:
            out["play_box"] = self._play_tracker.play_box
        except Exception:
            pass
        return out

    def input_keys(self):
        return {"data", "device", "shared"}

    def output_keys(self):
        return {"img", "current_box", "frame_ids", "player_bottom_points", "player_ids", "rink_profile", "play_box"}
