from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import cv2

# from PIL import Image

from typing import Dict, List

from yolox.evaluators.mot_evaluator import write_results_no_score

from hmlib.tracker.multitracker import JDETracker, torch_device
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.tracking_utils.evaluation import Evaluator
from hmlib.tracking_utils.io import write_results, read_results, append_results
import hmlib.datasets.dataset.jde as datasets

from hmlib.tracking_utils.utils import mkdir_if_missing
from hmlib.opts import opts

from hmlib.camera.camera import HockeyMOM
from hmlib.camera.cam_post_process import (
    FramePostProcessor,
)


def get_tracker(tracker_name: str, opt, frame_rate: int):
    if tracker_name == "jde":
        return JDETracker(opt, frame_rate=frame_rate)
    return None


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
        return torch.from_numpy(t).to(device)
    return t


def scale_tlwhs(tlwhs: List, scale: float):
    return tlwhs


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))


class HmPostProcessor:
    def __init__(
        self,
        opt,
        args,
        device,
        fps: float,
        save_dir: str,
        save_frame_dir: str = None,
        data_type: str = "mot",
        postprocess: bool = True,
        use_fork: bool = False,
        async_post_processing: bool = False,
    ):
        self._opt = opt
        self._args = args
        self._data_type = data_type
        self._postprocess = postprocess
        self._postprocessor = None
        self._use_fork = use_fork
        self._async_post_processing = async_post_processing
        self._fps = fps
        self._save_dir = save_dir
        self._save_frame_dir = save_frame_dir
        self._hockey_mom = None
        self._device = device
        self._counter = 0

    @property
    def data_type(self):
        return self._data_type

    def filter_outputs(self, outputs: torch.Tensor, output_results):
        # TODO: for batches, will be total length of N batches combined
        # assert len(outputs) == len(output_results)
        return outputs, output_results

    def _maybe_init(
        self,
        frame_id,
        letterbox_img,
        inscribed_img,
        original_img,
    ):
        if self._postprocessor is None:
            self.on_first_image(
                frame_id,
                letterbox_img,
                inscribed_img,
                original_img,
                device=self._device,
            )

    def online_callback(
        self,
        frame_id,
        online_tlwhs,
        online_ids,
        detections,
        info_imgs,
        letterbox_img,
        inscribed_img,
        original_img,
        online_scores,
    ):
        self._counter += 1
        if self._counter % 100 == 0:
            print(f"open file count: {get_open_files_count()}")
        if not self._postprocess:
            return detections, online_tlwhs
        letterbox_img = to_rgb_non_planar(letterbox_img).cpu()
        original_img = to_rgb_non_planar(original_img)
        inscribed_img = to_rgb_non_planar(inscribed_img)
        if isinstance(online_tlwhs, list) and len(online_tlwhs) != 0:
            online_tlwhs = torch.stack(
                [_pt_tensor(t, device=inscribed_img.device) for t in online_tlwhs]
            ).to(self._device)
        self._maybe_init(
            frame_id,
            letterbox_img,
            inscribed_img,
            original_img,
        )
        if (
            not self._args.scale_to_original_image
            and isinstance(letterbox_img, torch.Tensor)
            and letterbox_img.dtype != torch.uint8
        ):
            letterbox_img *= 255
            letterbox_img = letterbox_img.clip(min=0, max=255).to(torch.uint8)

        self._postprocessor.send(
            online_tlwhs,
            torch.tensor(online_ids, dtype=torch.int64),
            detections,
            info_imgs,
            letterbox_img,
            original_img,
        )
        return detections, online_tlwhs

    def on_first_image(
        self, frame_id, letterbox_img, inscribed_image, original_img, device
    ):
        if self._hockey_mom is None:
            if self._args.scale_to_original_image:
                self._hockey_mom = HockeyMOM(
                    image_width=original_img.shape[1],
                    image_height=original_img.shape[0],
                    device=device,
                )
            else:
                assert False  # Don't do this anymore
                self._hockey_mom = HockeyMOM(
                    image_width=letterbox_img.shape[1],
                    image_height=letterbox_img.shape[0],
                    device=device,
                )

        if self._postprocessor is None:
            self._postprocessor = FramePostProcessor(
                self._hockey_mom,
                start_frame_id=frame_id,
                data_type=self._data_type,
                fps=self._fps,
                save_dir=self._save_dir,
                save_frame_dir=self._save_frame_dir,
                device=device,
                opt=self._opt,
                args=self._args,
                use_fork=self._use_fork,
                async_post_processing=self._async_post_processing,
            )
            self._postprocessor.start()

    def stop(self):
        if self._postprocessor is not None:
            self._postprocessor.stop()


def track_sequence(
    opt,
    args,
    dataloader,
    tracker_name: str,
    result_filename,
    postprocessor: HmPostProcessor,
    save_dir: str = None,
    use_cuda: bool = True,
    data_type: str = "mot",
):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = get_tracker(tracker_name, opt=opt, frame_rate=dataloader.fps)
    dataset_timer = Timer()
    timer = Timer()

    incremental_results = False
    frame_id = 0

    if result_filename:
        results = read_results(result_filename, postprocessor.data_type)

    using_precomputed_results = len(results) != 0

    # origin_imgs,
    # letterbox_imgs,
    # inscribed_images,
    # info_imgs,
    # ids,

    for i, (
        # _, letterbox_img, img0, original_img
        original_img,
        letterbox_img,
        img0,
        info_imgs,
        ids,
    ) in enumerate(dataloader):
        if i:
            dataset_timer.toc()

        if frame_id % 20 == 0:
            logger.info(
                "Dataset frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, dataset_timer.average_time)
                )
            )

        frame_id = i
        if frame_id > 0 and frame_id <= args.skip_frame_count:
            timer.toc()

        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )

        # run tracking
        timer.tic()

        if frame_id < args.skip_frame_count:
            continue

        if use_cuda:
            blob = letterbox_img.cuda(torch_device())
        else:
            blob = letterbox_img

        online_tlwhs = []
        online_ids = []
        online_scores = []

        if using_precomputed_results:
            assert frame_id + 1 in results
            frame_results = results[frame_id + 1]
            for tlwh, target_id, score in frame_results:
                online_ids.append(target_id)
                online_tlwhs.append(tlwh)
                online_scores.append(score)
        else:
            # online_targets = tracker.update(blob, img0)
            blob = blob.permute(0, 2, 3, 1).contiguous()
            original_img = original_img.squeeze(0).permute(1, 2, 0).contiguous()
            online_targets = tracker.update(blob, original_img, dataloader=dataloader)

            # TODO: move this back to model portion so we can reuse results.txt
            for _, t in enumerate(online_targets):
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if vertical:
                    print("VERTICAL!")
                    vertical = False
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                else:
                    print(
                        f"Box area too small (< {opt.min_box_area}): {tlwh[2] * tlwh[3]} or vertical (vertical={vertical})"
                    )
            # save results
            # results.append((frame_id + 1, online_tlwhs, online_ids))
            results[frame_id + 1] = (online_tlwhs, online_ids)

            if postprocessor is not None:
                info_imgs = [
                    torch.tensor([img0.shape[0]], dtype=torch.int64),
                    torch.tensor([img0.shape[1]], dtype=torch.int64),
                    torch.tensor([frame_id], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64),
                ]

                postprocessor.online_callback(
                    frame_id=frame_id,
                    online_tlwhs=online_tlwhs,
                    online_ids=online_ids,
                    online_scores=online_scores,
                    detections=[],
                    info_imgs=info_imgs,
                    letterbox_img=torch.from_numpy(img0),
                    inscribed_img=torch.from_numpy(letterbox_img),
                    original_img=torch.from_numpy(original_img),
                )

            # save results
            if incremental_results and result_filename and (i + 1) % 25 == 0:
                results.append((frame_id + 1, online_tlwhs, online_ids))

        timer.toc()

        # if postprocessor is not None:
        #     postprocessor.send(online_tlwhs, online_ids, info_imgs, letterbox_img, original_img)

        if args.stop_at_frame and frame_id >= args.stop_at_frame:
            break

        # Last thing, tic the dataset timer before we wrap around and next the iter
        dataset_timer.tic()

    if postprocessor is not None:
        postprocessor.stop()

    # save results
    if result_filename:
        if incremental_results:
            append_results(result_filename, results, data_type)
        else:
            write_results(result_filename, results, data_type)

    return frame_id, timer.average_time, timer.calls
