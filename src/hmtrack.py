from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
# import math
# import time
# import copy
# import os
# import os.path as osp
# import cv2
# import logging
# import argparse
# import motmetrics as mm
import numpy as np
import torch

# import traceback
# import typing
# import multiprocessing
# import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans

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
    DefaultArguments,
    make_scale_array,
)


def get_tracker(tracker_name: str, opt, frame_rate: int):
    if tracker_name == "jde":
        return JDETracker(opt, frame_rate=frame_rate)
    return None


class HmPostProcessor:
    def __init__(self, opt, args, fps: int, save_dir: str, data_type: str = "mot"):
        self._opt = opt
        self._args = args
        self._data_type = data_type
        self._postprocessor = None
        self._fps = fps
        self._save_dir = save_dir
        self._hockey_mom = None

    @property
    def data_type(self):
        return self._data_type

    def online_callback(
        self,
        frame_id,
        online_tlwhs,
        online_ids,
        info_imgs,
        img,
        original_img,
        online_scores=None,
    ):
        if self._postprocessor is None:
            self.on_first_image(frame_id, info_imgs, img, original_img)
        self._postprocessor.send(online_tlwhs, online_ids, img, original_img)

    def on_first_image(self, frame_id, info_imgs, img, original_img):
        if self._hockey_mom is None:
            if self._args.scale_to_original_image:
                self._hockey_mom = HockeyMOM(
                    image_width=original_img.shape[1],
                    image_height=original_img.shape[0],
                )
            else:
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
    save_dir=None,
    use_cuda=True,
):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = get_tracker(tracker_name, opt=opt, frame_rate=dataloader.fps)
    dataset_timer = Timer()
    timer = Timer()

    # args = DefaultArguments()

    # do_postprocessing = True
    incremental_results = False
    args.stop_at_frame = 1000

    # show_image = args.show_image

    frame_id = 0
    # hockey_mom = None
    # postprocessor = None
    image_scale_array = None
    # first_frame_id = 0

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

        if args.scale_to_original_image and image_scale_array is None:
            image_scale_array = make_scale_array(from_img=img0, to_img=original_img)

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
                    if args.scale_to_original_image:
                        tlwh *= image_scale_array
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
                postprocessor.online_callback(
                    frame_id=frame_id,
                    online_tlwhs=online_tlwhs,
                    online_ids=online_ids,
                    info_imgs=info_imgs,
                    img=img0,
                    original_img=original_img,
                )

            # save results
            if incremental_results and result_filename and (i + 1) % 25 == 0:
                results.append((frame_id + 1, online_tlwhs, online_ids))

        timer.toc()

        # if postprocessor is not None:
        #     postprocessor.send(online_tlwhs, online_ids, img0, original_img)

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
