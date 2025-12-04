"""Camera post-processing utilities for arena estimation and video output.

This module computes the play/arena box from rink profiles and input frames,
configures `HockeyMOM` video geometry, and optionally drives `VideoOutput`
for camera debugging/training videos. Camera play tracking logic is handled
by `PlayTrackerPlugin` and the underlying `PlayTracker` module.
"""

from __future__ import absolute_import, division, print_function

import argparse
from typing import Any, Dict, List, Optional, Union

import torch

from hmlib.bbox.box_functions import (
    aspect_ratio,
    center,
    clamp_box,
    height,
    make_box_at_center,
    width,
)
from hmlib.builder import HM
from hmlib.camera.camera import HockeyMOM
from hmlib.log import logger
from hmlib.utils.image import image_height, image_width
from hmlib.video.video_stream import MAX_VIDEO_WIDTH


@HM.register_module()
class CamTrackPostProcessor:
    """Camera tracking head that owns HockeyMOM and the post-processing pipeline."""

    def __init__(
        self,
        opt: argparse.Namespace,
        args: argparse.Namespace,
        device: torch.device,
        fps: float,
        save_dir: str,
        output_video_path: Optional[str],
        camera_name: str,
        original_clip_box,
        video_out_pipeline: Dict[str, Any],
        save_frame_dir: Optional[str] = None,
        data_type: str = "mot",
        postprocess: bool = True,
        video_out_device: Optional[torch.device] = None,
        no_cuda_streams: bool = False,
    ):
        # Head-level configuration
        self._opt = opt
        self._args = args
        self._camera_name = camera_name
        self._data_type = data_type
        self._postprocess = postprocess
        self._video_out_pipeline = video_out_pipeline
        self._original_clip_box = original_clip_box
        self._fps = fps
        self._save_dir = save_dir
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._device = device
        self._video_out_device: Optional[torch.device] = video_out_device or device
        self._no_cuda_streams = no_cuda_streams
        self._counter = 0

        # Core post-processing state (initialized on first frame)
        self._hockey_mom: Optional[HockeyMOM] = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._arena_box: Optional[torch.Tensor] = None
        self.final_frame_width: Optional[int] = None
        self.final_frame_height: Optional[int] = None

    @property
    def data_type(self) -> str:
        return self._data_type

    def filter_outputs(self, outputs: torch.Tensor, output_results):
        return outputs, output_results

    def _maybe_init(
        self,
        frame_id: int,
        img_width: int,
        img_height: int,
        arena: List[int],
        device: torch.device,
    ) -> None:
        if not self.is_initialized():
            self.on_first_image(
                frame_id=frame_id,
                img_width=img_width,
                img_height=img_height,
                device=device,
                arena=arena,
            )

    def is_initialized(self) -> bool:
        return self._hockey_mom is not None and self._play_tracker is not None

    @staticmethod
    def calculate_play_box(
        results: Dict[str, Any], context: Dict[str, Any], scale: float = 1.3
    ) -> List[int]:
        play_box = torch.tensor(context["rink_profile"]["combined_bbox"], dtype=torch.int64)
        ww, hh = width(play_box), height(play_box)
        cc = center(play_box)
        play_box = make_box_at_center(cc, ww * scale, hh * scale)
        return clamp_box(
            play_box,
            [
                0,
                0,
                image_width(results["original_images"]),
                image_height(results["original_images"]),
            ],
        )

    def process_tracking(
        self,
        results: Dict[str, Any],
        context: Dict[str, Any],
    ):
        self._counter += 1
        if not self._postprocess:
            return results
        if self._hockey_mom is None:
            video_data_sample = results["data_samples"].video_data_samples[0]
            metainfo = video_data_sample.metainfo
            original_shape = metainfo["ori_shape"]

            arena = self.calculate_play_box(results, context)

            assert isinstance(original_shape, torch.Size)
            frame_id = getattr(video_data_sample, "frame_id", None)
            if frame_id is None:
                frame_id = metainfo.get("frame_id", 0)
            self._maybe_init(
                frame_id=int(frame_id),
                img_height=int(original_shape[0]),
                img_width=int(original_shape[1]),
                arena=arena,
                device=self._device,
            )
        # Expose arena/play box for downstream trunks (e.g., PlayTrackerPlugin, VideoOutPlugin)
        results["arena"] = self.get_arena_box()
        # results["final_frame_width"] = self.final_frame_width
        # results["final_frame_height"] = self.final_frame_height
        return results

    def on_first_image(
        self,
        frame_id: int,
        img_width: int,
        img_height: int,
        arena: List[int],
        device: torch.device,
    ) -> None:
        assert self._hockey_mom is None

        # Initialize HockeyMOM video geometry and cache arena box
        self._hockey_mom = HockeyMOM(
            image_width=img_width,
            image_height=img_height,
            fps=self._fps,
            device=device,
            camera_name=self._camera_name,
        )
        self._arena_box = (
            self._hockey_mom.video.bounding_box()
            if not getattr(self._args, "crop_play_box", False)
            else torch.as_tensor(arena, dtype=torch.float, device=device)
        )
        self.secondary_init()

    def secondary_init(self) -> None:
        assert self._hockey_mom is not None
        play_box: torch.Tensor = (
            self._arena_box
            if self._arena_box is not None
            else self._hockey_mom.video.bounding_box()
        )
        play_width, play_height = width(play_box), height(play_box)

        if getattr(self._args, "crop_play_box", False):
            if getattr(self._args, "crop_output_image", True):
                self.final_frame_height = play_height
                self.final_frame_width = play_height * self._final_aspect_ratio
                if self.final_frame_width > MAX_VIDEO_WIDTH:
                    self.final_frame_width = MAX_VIDEO_WIDTH
                    self.final_frame_height = self.final_frame_width / self._final_aspect_ratio

            else:
                self.final_frame_height = play_height
                self.final_frame_width = play_width
        else:
            if getattr(self._args, "crop_output_image", True):
                self.final_frame_height = self._hockey_mom.video.height
                self.final_frame_width = self._hockey_mom.video.height * self._final_aspect_ratio
                if self.final_frame_width > MAX_VIDEO_WIDTH:
                    self.final_frame_width = MAX_VIDEO_WIDTH
                    self.final_frame_height = self.final_frame_width / self._final_aspect_ratio

            else:
                self.final_frame_height = self._hockey_mom.video.height
                self.final_frame_width = self._hockey_mom.video.width

        self.final_frame_width = int(self.final_frame_width + 0.5)
        self.final_frame_height = int(self.final_frame_height + 0.5)

    @property
    def output_video_path(self) -> Optional[str]:
        return self._output_video_path

    def start(self) -> None:
        # CamTrackPostProcessor runs synchronously; kept for API compatibility.
        return None

    def stop(self) -> None:
        # Kept for API compatibility; nothing to stop in synchronous mode.
        return None

    def get_arena_box(self) -> torch.Tensor:
        assert self._hockey_mom is not None
        if self._arena_box is not None:
            return self._arena_box
        return self._hockey_mom.video.bounding_box()
