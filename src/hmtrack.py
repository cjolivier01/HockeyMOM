from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch

from threading import Thread
from multiprocessing import Queue

# from PIL import Image

from typing import Dict, List

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
    make_scale_array,
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
    return image


def scale_tlwhs(tlwhs: List, scale: float):
    return tlwhs


class HmPostProcessor:
    def __init__(
        self,
        opt,
        args,
        fps: int,
        save_dir: str,
        data_type: str = "mot",
        postprocess: bool = True,
    ):
        self._opt = opt
        self._args = args
        self._data_type = data_type
        self._postprocess = postprocess
        self._postprocessor = None
        self._fps = fps
        self._save_dir = save_dir
        self._hockey_mom = None
        self._image_scale_array = None
        self.dw = 0
        self.dh = 0
        self._scale_inscribed_to_original = 1
        self._timer = None
        self._counter = 0

    @property
    def data_type(self):
        return self._data_type

    def online_callback(
        self,
        frame_id,
        online_tlwhs,
        online_ids,
        detections,
        info_imgs,
        img,
        inscribed_image,
        original_img,
        online_scores=None,
    ):
        if not self._postprocess:
            return
        if self._timer is not None:
            self._timer.toc()
        if isinstance(img, torch.Tensor):
            img = to_rgb_non_planar(img).cpu()
            original_img = to_rgb_non_planar(original_img)
        if self._postprocessor is None:
            self.on_first_image(frame_id, info_imgs, img, inscribed_image, original_img)
        if self._args.scale_to_original_image:
            scaled_online_tlwhs = []
            for tlwh in online_tlwhs:
                tlwh[0] -= self.dw
                tlwh[1] -= self.dh
                tlwh /= self._scale_inscribed_to_original
                scaled_online_tlwhs.append(tlwh)
            online_tlwhs = scaled_online_tlwhs
        if (
            not self._args.scale_to_original_image
            and isinstance(img, torch.Tensor)
            and img.dtype != torch.uint8
        ):
            img *= 255
            img = img.clip(min=0, max=255).to(torch.uint8)

        self._postprocessor.send(
            online_tlwhs,
            online_ids,
            detections,
            info_imgs,
            img,
            original_img,
        )
        if self._timer is None:
            self._timer = Timer()
        self._counter += 1
        if self._counter % 20 == 0:
            logger.info(
                "Model Proc frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, self._timer.average_time)
                )
            )
        self._timer.tic()

    def on_first_image(self, frame_id, info_imgs, img, inscribed_image, original_img):
        _, _, self.dw, self.dh = datasets.calculate_letterbox(
            shape=inscribed_image.shape, height=img.shape[0], width=img.shape[1]
        )
        if self._args.scale_to_original_image and self._image_scale_array is None:
            self._scale_processed_to_inscribed = make_scale_array(
                from_img=img,
                to_img=inscribed_image,
            )
            self._scale_processed_to_inscribed = torch.cat(
                (
                    self._scale_processed_to_inscribed,
                    self._scale_processed_to_inscribed,
                ),
                dim=0,
            ).numpy()
            self._scale_original_to_processed = make_scale_array(
                from_img=original_img,
                to_img=img,
            )
            self._scale_original_to_processed = torch.cat(
                (self._scale_original_to_processed, self._scale_original_to_processed),
                dim=0,
            ).numpy()
            self._scale_inscribed_to_original = make_scale_array(
                from_img=inscribed_image,
                to_img=original_img,
            )
            self._scale_inscribed_to_original = torch.cat(
                (self._scale_inscribed_to_original, self._scale_inscribed_to_original),
                dim=0,
            ).numpy()

            self._image_scale_array = make_scale_array(
                from_img=inscribed_image,
                to_img=original_img,
            )
            self._image_scale_array = torch.cat(
                (self._image_scale_array, self._image_scale_array), dim=0
            )
        if self._hockey_mom is None:
            if self._args.scale_to_original_image:
                self._hockey_mom = HockeyMOM(
                    image_width=original_img.shape[1],
                    image_height=original_img.shape[0],
                )
            else:
                # assert not isinstance(img, torch.Tensor) or img.dtype == torch.uint8
                self._hockey_mom = HockeyMOM(
                    image_width=img.shape[1],
                    image_height=img.shape[0],
                )

        if self._postprocessor is None:
            self._postprocessor = FramePostProcessor(
                self._hockey_mom,
                start_frame_id=frame_id,
                data_type=self._data_type,
                fps=self._fps,
                save_dir=self._save_dir,
                opt=self._opt,
                args=self._args,
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
    output_video_path: str = None,
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
    image_scale_array = None

    if result_filename:
        results = read_results(result_filename, postprocessor.data_type)

    using_precomputed_results = len(results) != 0

    for i, (_, img, img0, original_img) in enumerate(dataloader):
        if i:
            dataset_timer.toc()

        info_imgs = [
            np.array([img0.shape[0]], dtype=np.int),
            np.array([img0.shape[1]], dtype=np.int),
            np.array([frame_id], dtype=np.int),
            np.array([], dtype=np.int),
        ]

        if frame_id % 20 == 0:
            logger.info(
                "Dataset frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, dataset_timer.average_time)
                )
            )

        # if args.scale_to_original_image and image_scale_array is None:
        #     image_scale_array = make_scale_array(from_img=img0, to_img=original_img)
        #     image_scale_array = torch.cat(
        #         [image_scale_array, image_scale_array]
        #     ).numpy()

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
            blob = torch.from_numpy(img).cuda(torch_device()).unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        online_tlwhs = []
        online_ids = []

        if using_precomputed_results:
            assert frame_id + 1 in results
            frame_results = results[frame_id + 1]
            for tlwh, target_id, score in frame_results:
                online_ids.append(target_id)
                online_tlwhs.append(tlwh)
        else:
            online_targets = tracker.update(blob, img0)

            # online_scores = []

            # TODO: move this back to model portion so we can reuse results.txt
            for _, t in enumerate(online_targets):
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if vertical:
                    print("VERTICAL!")
                    vertical = False
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    # if args.scale_to_original_image:
                    #     tlwh *= image_scale_array
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    # online_scores.append(t.score)
                else:
                    print(
                        f"Box area too small (< {opt.min_box_area}): {tlwh[2] * tlwh[3]} or vertical (vertical={vertical})"
                    )
            # save results
            # results.append((frame_id + 1, online_tlwhs, online_ids))
            results[frame_id + 1] = (online_tlwhs, online_ids)

            if postprocessor is not None:
                # cv2.imshow("img0", img0)
                # #cv2.imshow("img", img)
                # #cv2.imshow("img", original_img)
                # cv2.waitKey(1)
                # continue

                postprocessor.online_callback(
                    frame_id=frame_id,
                    online_tlwhs=online_tlwhs,
                    online_ids=online_ids,
                    detections=[],
                    info_imgs=info_imgs,
                    img=img0,
                    inscribed_image=img,
                    original_img=original_img,
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
