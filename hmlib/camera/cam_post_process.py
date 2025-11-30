"""Post-processing pipeline for camera tracking and play visualization.

This script ties together tracking CSVs, camera config and visualization to
produce debug or training videos for the camera controller.

@see @ref hmlib.camera.camera_dataframe.CameraTrackingDataFrame "CameraTrackingDataFrame"
@see @ref hmlib.camera.play_tracker.PlayTracker "PlayTracker"
"""

from __future__ import absolute_import, division, print_function

import argparse
import contextlib
import os
import time
import traceback
from threading import Thread
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
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.camera.play_tracker import PlayTracker
from hmlib.log import logger
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.ui import Shower
from hmlib.utils.containers import create_queue
from hmlib.utils.image import image_height, image_width
from hmlib.utils.progress_bar import ProgressBar
from hmlib.video.video_out import VideoOutput, get_open_files_count
from hmlib.video.video_stream import MAX_VIDEO_WIDTH


##
#  _____         __             _ _                                                     _
# |  __ \       / _|           | | |      /\                                           | |
# | |  | | ___ | |_  __ _ _   _| | |_    /  \    _ __  __ _ _   _ _ __ ___   ___  _ __ | |_  ___
# | |  | |/ _ \|  _|/ _` | | | | | __|  / /\ \  | '__|/ _` | | | | '_ ` _ \ / _ \| '_ \| __|/ __|
# | |__| |  __/| | | (_| | |_| | | |_  / ____ \ | |  | (_| | |_| | | | | | |  __/| | | | |_ \__ \
# |_____/ \___||_|  \__,_|\__,_|_|\__|/_/    \_\|_|   \__, |\__,_|_| |_| |_|\___||_| |_|\__||___/
#                                                      __/ |
#                                                     |___/
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
        async_post_processing: bool = False,
        video_out_device: Optional[torch.device] = None,
        video_out_cache_size: int = 2,
        async_video_out: bool = False,
        no_cuda_streams: bool = False,
    ):
        # Head-level configuration
        self._opt = opt
        self._args = args
        self._camera_name = camera_name
        self._data_type = data_type
        self._postprocess = postprocess
        self._video_out_pipeline = video_out_pipeline
        self._async_post_processing = async_post_processing
        self._async_video_out = async_video_out
        self._original_clip_box = original_clip_box
        self._fps = fps
        self._save_dir = save_dir
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._device = device
        self._video_out_device: Optional[torch.device] = video_out_device or device
        self._video_out_cache_size = video_out_cache_size
        self._no_cuda_streams = no_cuda_streams
        self._counter = 0

        # Core post-processing state (initialized on first frame)
        self._hockey_mom: Optional[HockeyMOM] = None
        self._start_frame_id: Optional[int] = None
        self._queue = create_queue(mp=False, name="CamPostProcess-Queue")
        self._thread: Optional[Thread] = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._output_video = None
        self._timer = Timer()
        self._camera_tracking_data: Optional[CameraTrackingDataFrame] = None
        self._video_output_campp: Optional[VideoOutput] = None
        self._queue_timer = Timer()
        self._send_to_timer_post_process = Timer()
        self._exception: Optional[BaseException] = None
        self._shower: Union[None, Shower] = None
        self._play_tracker: Optional[PlayTracker] = None
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
        if self._counter % 100 == 0:
            logger.info(f"open file count: {get_open_files_count()}")
        if not self._postprocess:
            return results
        if not self.is_initialized():
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
        # results = self.send(results)
        results["arena"] = self.get_arena_box()
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

        self._hockey_mom = HockeyMOM(
            image_width=img_width,
            image_height=img_height,
            fps=self._fps,
            device=device,
            camera_name=self._camera_name,
        )
        self._start_frame_id = frame_id

        play_box_tensor = (
            self._hockey_mom.video.bounding_box()
            if not getattr(self._args, "crop_play_box", False)
            else torch.as_tensor(arena, dtype=torch.float, device=device)
        )

        self._play_tracker = PlayTracker(
            hockey_mom=self._hockey_mom,
            play_box=play_box_tensor,
            device=device,
            original_clip_box=self._original_clip_box,
            progress_bar=None,
            args=self._args,
        )
        self.secondary_init()
        self.eval()
        if self._async_post_processing:
            self.start()

    def secondary_init(self) -> None:
        assert self._play_tracker is not None
        play_box: torch.Tensor = self._play_tracker.play_box
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
                assert self._hockey_mom is not None
                self.final_frame_height = self._hockey_mom.video.height
                self.final_frame_width = self._hockey_mom.video.height * self._final_aspect_ratio
                if self.final_frame_width > MAX_VIDEO_WIDTH:
                    self.final_frame_width = MAX_VIDEO_WIDTH
                    self.final_frame_height = self.final_frame_width / self._final_aspect_ratio

            else:
                assert self._hockey_mom is not None
                self.final_frame_height = self._hockey_mom.video.height
                self.final_frame_width = self._hockey_mom.video.width

        self.final_frame_width = int(self.final_frame_width + 0.5)
        self.final_frame_height = int(self.final_frame_height + 0.5)

        if getattr(self._args, "save_camera_data", False) and self._save_dir:
            self._camera_tracking_data = CameraTrackingDataFrame(
                output_file=os.path.join(self._save_dir, "camera.csv"),
                input_batch_size=self._args.batch_size,
            )

        if not getattr(self._args, "no_frame_postprocessing", False) and self.output_video_path:
            assert self._video_output_campp is None
            self._video_output_campp = VideoOutput(
                name="TRACKING",
                args=self._args,
                output_video_path=self.output_video_path,
                fps=self._fps,
                start=False,
                bit_rate=self._args.output_video_bit_rate,
                output_frame_width=self.final_frame_width,
                output_frame_height=self.final_frame_height,
                save_frame_dir=self._save_frame_dir,
                original_clip_box=self._original_clip_box,
                cache_size=self._video_out_cache_size,
                async_output=self._async_video_out,
                video_out_pipeline=self._video_out_pipeline,
                device=self._video_out_device,
                skip_final_save=self._args.skip_final_video_save,
                no_cuda_streams=self._no_cuda_streams,
            )
            self._video_output_campp.start()
        elif getattr(self._args, "show_image", False):
            self._shower = Shower("CamTrackPostProcessor", self._args.show_scaled, max_size=1)

    def eval(self) -> None:
        if self._play_tracker is not None:
            self._play_tracker.eval()

    @property
    def output_video_path(self) -> Optional[str]:
        return self._output_video_path

    def start(self) -> None:
        self._thread = Thread(target=self._start, name="CamPostProc")
        self._thread.start()

    def _start(self) -> None:
        self.postprocess_frame_worker()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        if self._video_output_campp is not None:
            self._video_output_campp.stop()

    def send(
        self,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._exception is not None:
            raise self._exception
        try:
            if self._async_post_processing:
                with TimeTracker(
                    "Send to cam post process queue",
                    self._send_to_timer_post_process,
                    print_interval=50,
                ):
                    wait_count = 0
                    while self._queue.qsize() > 1:
                        if not getattr(self._args, "debug", False) and not getattr(
                            self._args, "show_image", False
                        ):
                            wait_count += 1
                            if wait_count % 100 == 0:
                                logger.info("Cam post-process queue too large")
                        time.sleep(0.001)
                    self._queue.put(data)
            else:
                assert self._play_tracker is not None
                with torch.no_grad():
                    prof = getattr(self._args, "profiler", None)
                    ctx = (
                        prof.rf("play_tracker.forward")
                        if getattr(prof, "enabled", False)
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        results = self._play_tracker.forward(results=data)
                del data
                for frame_id, current_box in zip(results["frame_ids"], results["current_box"]):
                    assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
                    if self._camera_tracking_data is not None:
                        self._camera_tracking_data.add_frame_records(
                            frame_id=frame_id,
                            tlbr=current_box if current_box.ndim == 4 else current_box.unsqueeze(0),
                        )
                if self._video_output_campp is not None:
                    prof = getattr(self._args, "profiler", None)
                    ctx = (
                        prof.rf("video_out.append")
                        if getattr(prof, "enabled", False)
                        else contextlib.nullcontext()
                    )
                    with ctx:
                        self._video_output_campp.append(results)
                elif self._shower is not None and "img" in results:
                    self._shower.show(results["img"].cpu())
                return results
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise

    def postprocess_frame_worker(self) -> None:
        try:
            self._postprocess_frame_worker()
        except Exception as ex:
            self._exception = ex
            print(ex)
            traceback.print_exc()
            raise
        finally:
            if self._video_output_campp is not None:
                self._video_output_campp.stop()

    def _postprocess_frame_worker(self) -> None:
        while True:
            results = self._queue.get()
            if results is None:
                break

            assert self._play_tracker is not None
            with torch.no_grad():
                prof = getattr(self._args, "profiler", None)
                ctx = (
                    prof.rf("play_tracker.forward")
                    if getattr(prof, "enabled", False)
                    else contextlib.nullcontext()
                )
                with ctx:
                    results = self._play_tracker.forward(results=results)

            for frame_id, current_box in zip(results["frame_ids"], results["current_box"]):
                assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
                if self._camera_tracking_data is not None:
                    self._camera_tracking_data.add_frame_records(
                        frame_id=frame_id,
                        tlbr=current_box if current_box.ndim == 4 else current_box.unsqueeze(0),
                    )
            if self._video_output_campp is not None:
                prof = getattr(self._args, "profiler", None)
                ctx = (
                    prof.rf("video_out.append")
                    if getattr(prof, "enabled", False)
                    else contextlib.nullcontext()
                )
                with ctx:
                    self._video_output_campp.append(results)
            elif self._shower is not None and "img" in results:
                self._shower.show(results["img"])

    def get_arena_box(self) -> torch.Tensor:
        assert self._play_tracker is not None
        return self._play_tracker.play_box
