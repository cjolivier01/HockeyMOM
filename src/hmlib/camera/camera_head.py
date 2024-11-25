from __future__ import absolute_import, division, print_function

import os
from typing import Any, Dict, List, Optional

import torch

from hmlib.builder import HM
from hmlib.camera.cam_post_process import CamTrackPostProcessor
from hmlib.camera.camera import HockeyMOM
from hmlib.log import logger
from hmlib.utils.image import image_height, image_width


def to_rgb_non_planar(image):
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

    def _maybe_init(self, frame_id, img_width: int, img_height: int, device: torch.device):
        if self._postprocessor is None:
            self.on_first_image(
                frame_id=frame_id, img_width=img_width, img_height=img_height, device=device
            )

    def is_initialized(self) -> bool:
        return not self._hockey_mom is None

    def process_tracking(
        self,
        results: Dict[str, Any],
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
            # torch.Size will be (H, W)
            assert isinstance(original_shape, torch.Size)
            self._maybe_init(
                frame_id=video_data_sample.frame_id,
                img_height=original_shape[0],
                img_width=original_shape[1],
                device=self._device,
            )
        results = self._postprocessor.send(results)
        return results

    def on_first_image(self, frame_id, img_width: int, img_height: int, device: torch.device):
        if self._hockey_mom is None:
            self._hockey_mom = HockeyMOM(
                image_width=img_width,
                image_height=img_height,
                fps=self._fps,
                device=device,
                camera_name=self._camera_name,
            )

        if self._postprocessor is None:
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
