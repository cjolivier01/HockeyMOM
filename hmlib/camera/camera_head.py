from __future__ import absolute_import, division, print_function

"""Camera tracking head that bridges detection outputs to post-processing.

This module wires the `HockeyMOM` camera logic and the `CamTrackPostProcessor`
into an `HM`-registered head that can be used in hm2 configs.
"""

import os
from typing import Any, Dict, List, Optional

import torch

from hmlib.bbox.box_functions import center, clamp_box, height, make_box_at_center, width
from hmlib.builder import HM
from hmlib.camera.cam_post_process import CamTrackPostProcessor
from hmlib.camera.camera import HockeyMOM
from hmlib.log import logger
from hmlib.utils.image import image_height, image_width


def to_rgb_non_planar(image):
    """Convert CHW tensors to HWC tensors for OpenCV/visualization."""
    if isinstance(image, torch.Tensor):
        if (
            len(image.shape) == 4
            and image.shape[0] == 1
            and image.shape[1] == 3
            and image.shape[-1] > 1
        ):
            # Assuming it is planar
            image = torch.squeeze(image, dim=0).permute(1, 2, 0)
        elif len(image.shape) == 3 and image.shape[0] == 3:
            # Assuming it is planar
            image = image.permute(1, 2, 0)
    return image


def _pt_tensor(t, device):
    if not isinstance(t, torch.Tensor):
        return torch.from_numpy(t).to(device, non_blocking=True)
    return t


def scale_tlwhs(tlwhs: List, scale: float):
    return tlwhs


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))


@HM.register_module()
class CamTrackHead:
    """Entry head that owns `HockeyMOM` and its tracking post-processor."""
    # TODO: Get rid of this class entirely, since it's just a
    # bounce to the postprocessor and al the init crap
    # can be a function or somthing in the postprocessor file?
    def __init__(
        self,
        opt,
        args,
        device,
        fps: float,
        save_dir: str,
        output_video_path: Optional[str],
        camera_name: str,
        original_clip_box: torch.Tensor,
        video_out_pipeline: Dict[str, Any],
        save_frame_dir: str = None,
        data_type: str = "mot",
        postprocess: bool = True,
        async_post_processing: bool = False,
        video_out_device: str = None,
        video_out_cache_size: int = 2,
        async_video_out: bool = False,
        no_cuda_streams: bool = False,
    ):
        self._opt = opt
        self._no_cuda_streams = no_cuda_streams
        self._args = args
        self._camera_name = camera_name
        self._data_type = data_type
        self._postprocess = postprocess
        self._postprocessor = None
        self._video_out_pipeline = video_out_pipeline
        self._async_post_processing = async_post_processing
        self._async_video_out = async_video_out
        self._original_clip_box = original_clip_box
        self._fps = fps
        self._save_dir = save_dir
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._hockey_mom = None
        self._device = device
        self._video_out_device = video_out_device
        if self._video_out_device is None:
            self._video_out_device = self._device
        self._video_out_cache_size = video_out_cache_size
        self._counter = 0

    @property
    def data_type(self):
        return self._data_type

    def filter_outputs(self, outputs: torch.Tensor, output_results):
        return outputs, output_results

    def _maybe_init(self, frame_id, img_width: int, img_height: int, arena: List[int], device: torch.device):
        if self._postprocessor is None:
            self.on_first_image(
                frame_id=frame_id, img_width=img_width, img_height=img_height, device=device, arena=arena
            )

    def is_initialized(self) -> bool:
        return not self._hockey_mom is None

    @staticmethod
    def calculate_play_box(results: Dict[str, Any], context: Dict[str, Any], scale: float = 1.3) -> List[int]:
        # Use the first video_data_sample's rink_profile to seed the play box
        track_container = results["data_samples"]
        video_data_sample = track_container.video_data_samples[0]
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
        """Run camera tracking post-processing for a batch of detections."""
        self._counter += 1
        if self._counter % 100 == 0:
            logger.info(f"open file count: {get_open_files_count()}")
        if not self._postprocess:
            return results
        if not self.is_initialized():
            video_data_sample = results["data_samples"].video_data_samples[0]
            metainfo = video_data_sample.metainfo
            original_shape = metainfo["ori_shape"]
            # torch.Size will be (H, W)

            # play_box: List[int] = [0, 0, original_shape[1], original_shape[0]]
            # first_data_sample = results['data_samples'][0]
            # play_box = first_data_sample.metainfo['rink_profile']['combined_bbox']

            arena = self.calculate_play_box(results, context)

            assert isinstance(original_shape, torch.Size)
            frame_id = getattr(video_data_sample, "frame_id", None)
            if frame_id is None:
                frame_id = metainfo.get("frame_id", 0)
            self._maybe_init(
                frame_id=frame_id,
                img_height=original_shape[0],
                img_width=original_shape[1],
                arena=arena,
                device=self._device,
            )
        results = self._postprocessor.send(results)
        return results

    def on_first_image(self, frame_id, img_width: int, img_height: int, arena: List[int], device: torch.device):
        """Initialize `HockeyMOM` and the post-processor on the first frame."""
        assert self._hockey_mom is None

        self._hockey_mom = HockeyMOM(
            image_width=img_width,
            image_height=img_height,
            fps=self._fps,
            device=device,
            camera_name=self._camera_name,
        )
        assert self._postprocessor is None

        self._postprocessor = CamTrackPostProcessor(
            self._hockey_mom,
            start_frame_id=frame_id,
            data_type=self._data_type,
            fps=self._fps,
            save_dir=self._save_dir,
            output_video_path=self._output_video_path,
            save_frame_dir=self._save_frame_dir,
            device=device,
            original_clip_box=self._original_clip_box,  # TODO: Put in args
            play_box=arena,
            args=self._args,
            async_post_processing=self._async_post_processing,
            video_out_device=self._video_out_device,
            video_out_cache_size=self._video_out_cache_size,
            async_video_out=self._async_video_out,
            video_out_pipeline=self._video_out_pipeline,
            no_cuda_streams=self._no_cuda_streams,
        )
        self._postprocessor.eval()
        if self._async_post_processing:
            self._postprocessor.start()

    def stop(self):
        if self._postprocessor is not None:
            self._postprocessor.stop()
