from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch

from typing import Dict, List

#from yolox.evaluators.mot_evaluator import TrackingHead
from hmlib.camera.camera import HockeyMOM
from hmlib.camera.cam_post_process import CamTrackPostProcessor


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


class CamTrackHead:
    def __init__(
        self,
        opt,
        args,
        device,
        fps: float,
        save_dir: str,
        original_clip_box: torch.Tensor,
        save_frame_dir: str = None,
        data_type: str = "mot",
        postprocess: bool = True,
        use_fork: bool = False,
        async_post_processing: bool = False,
        video_out_device: str = None,
    ):
        self._opt = opt
        self._args = args
        self._data_type = data_type
        self._postprocess = postprocess
        self._postprocessor = None
        self._use_fork = use_fork
        self._async_post_processing = async_post_processing
        self._original_clip_box = original_clip_box
        self._fps = fps
        self._save_dir = save_dir
        self._save_frame_dir = save_frame_dir
        self._hockey_mom = None
        self._device = device
        self._video_out_device = video_out_device
        if self._video_out_device is None:
            self._video_out_device = self._device
        self._counter = 0

    @property
    def data_type(self):
        return self._data_type

    def filter_outputs(self, outputs: torch.Tensor, output_results):
        return outputs, output_results

    def _maybe_init(
        self,
        frame_id,
        letterbox_img,
        original_img,
    ):
        if self._postprocessor is None:
            self.on_first_image(
                frame_id,
                letterbox_img,
                original_img,
                device=self._device,
            )

    def process_tracking(
        self,
        frame_id,
        online_tlwhs,
        online_ids,
        detections,
        info_imgs,
        letterbox_img,
        original_img,
        online_scores,
    ):
        self._counter += 1
        if self._counter % 100 == 0:
            print(f"open file count: {get_open_files_count()}")
        if not self._postprocess:
            return detections, online_tlwhs
        if letterbox_img is not None:
            letterbox_img = to_rgb_non_planar(letterbox_img).cpu()
        original_img = to_rgb_non_planar(original_img)
        if isinstance(online_tlwhs, list) and len(online_tlwhs) != 0:
            online_tlwhs = torch.stack(
                [_pt_tensor(t, device=self._device) for t in online_tlwhs]
            ).to(self._device)
        self._maybe_init(
            frame_id,
            letterbox_img,
            original_img,
        )
        assert isinstance(online_ids, torch.Tensor) or (
            isinstance(online_ids, list) and len(online_ids) == 0
        )
        self._postprocessor.send(
            online_tlwhs,
            online_ids,
            detections,
            info_imgs,
            None,
            original_img,
        )
        return detections, online_tlwhs

    def on_first_image(self, frame_id, letterbox_img, original_img, device):
        if self._hockey_mom is None:
            if len(original_img.shape) == 4:
                original_img = original_img[0]
            self._hockey_mom = HockeyMOM(
                image_width=original_img.shape[1],
                image_height=original_img.shape[0],
                device=device,
            )

        if self._postprocessor is None:
            self._postprocessor = CamTrackPostProcessor(
                self._hockey_mom,
                start_frame_id=frame_id,
                data_type=self._data_type,
                fps=self._fps,
                save_dir=self._save_dir,
                save_frame_dir=self._save_frame_dir,
                device=device,
                original_clip_box=self._original_clip_box,  # TODO: Put in args
                args=self._args,
                use_fork=self._use_fork,
                async_post_processing=self._async_post_processing,
                video_out_device=self._video_out_device,
            )
            self._postprocessor.start()

    def stop(self):
        if self._postprocessor is not None:
            self._postprocessor.stop()
