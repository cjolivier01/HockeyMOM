import glob
import math
import os
import os.path as osp
import random
import traceback
import time
import multiprocessing
import threading
from collections import OrderedDict
from typing import List

import cv2
import json
import numpy as np
import torch
import copy

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import torch.nn.functional as F
from cython_bbox import bbox_overlaps as bbox_ious
from hmlib.opts import opts
from hmlib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from hmlib.utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from hmlib.tracking_utils.timer import Timer
from hmlib.datasets.dataset.jde import letterbox
from hmlib.tracking_utils.log import logger
from .stitching import StitchDataset

from yolox.data import MOTDataset


class MOTLoadVideoWithOrig(MOTDataset):  # for inference
    def __init__(
        self,
        path,
        img_size,
        clip_original=None,
        video_id: int = 1,
        data_dir=None,
        json_file: str = "train.json",
        name: str = "train",
        preproc=None,
        return_origin_img=False,
        max_frames: int = 0,
        batch_size: int = 1,
        start_frame_number: int = 0,
        multi_width_img_info: bool = True,
        embedded_data_loader=None,
    ):
        super().__init__(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            preproc=preproc,
            return_origin_img=return_origin_img,
        )
        self._path = path
        self._start_frame_number = start_frame_number
        self.clip_original = clip_original
        self.process_height = img_size[0]
        self.process_width = img_size[1]
        self._multi_width_img_info = multi_width_img_info
        self.width_t = None
        self.height_t = None
        self._count = torch.tensor([0], dtype=torch.int32)
        self.video_id = torch.tensor([video_id], dtype=torch.int32)
        self._last_size = None
        self._max_frames = max_frames
        self._batch_size = batch_size
        self._timer = None
        self._to_worker_queue = multiprocessing.Queue()
        self._from_worker_queue = multiprocessing.Queue()
        self.vn = None
        self.cap = None
        self._thread = None
        self._embedded_data_loader = embedded_data_loader
        assert self._embedded_data_loader is None or path is None

        self._open_video()
        self._close_video()

        self._start_worker()

    def _open_video(self):
        if self._embedded_data_loader is None:
            self.cap = cv2.VideoCapture(self._path)
            self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not self.vn:
                raise RuntimeError(
                    f"Video {self._path} either does not exist or has no usable video content"
                )
            assert self._start_frame_number >= 0 and self._start_frame_number < self.vn
            if self._start_frame_number:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame_number)
        else:
            self._embedded_data_loader_iter = iter(self._embedded_data_loader)
            self.vn = len(self._embedded_data_loader)
            # self.vw = self._embedded_data_loader.width
            # self.vh = self._embedded_data_loader.height
            self.vw = None
            self.vh = None
            self.fps = self._embedded_data_loader.fps
        print("Lenth of the video: {:d} frames".format(self.vn))

    def _close_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._embedded_data_loader is not None:
            self._embedded_data_loader.close()

    @property
    def dataset(self):
        return self

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def letterbox_dw_dh(self):
        return self.letterbox_dw, self.letterbox_dh

    def _next_frame_worker(self):
        try:
            self._open_video()
            while True:
                cmd = self._to_worker_queue.get()
                if cmd != "ok":
                    break
                next_batch = self._get_next_batch()
                self._from_worker_queue.put(next_batch)
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                print(ex)
                traceback.print_exc()
            self._from_worker_queue.put(ex)
            return
        finally:
            self._close_video()
        self._from_worker_queue.put(StopIteration())
        return

    def _start_worker(self):
        self._thread = threading.Thread(target=self._next_frame_worker)
        self._thread.start()
        # if not os.fork():
        #     try:
        #         self._next_frame_worker()
        #     except Exception as ex:
        #         print(e)
        #         traceback.print_exc()
        #         raise
        #     os._exit(0)
        # else:
        #     self._to_worker_queue.put("ok")
        self._to_worker_queue.put("ok")
        self._next_counter = 0

    def close(self):
        if self._thread is None:
            return
        self._to_worker_queue.put("stop")
        self._thread.join()

    def __delete__(self):
        self.close()

    def __iter__(self):
        self._timer = Timer()
        return self

    def _read_next_image(self):
        if self.cap is not None:
            return self.cap.read()
        else:
            try:
                img = next(self._embedded_data_loader_iter)
                if self.vw is None:
                    self.vw = img.shape[1]
                    self.vh = img.shape[0]
                return True, img

            except StopIteration:
                return False, None

    def _get_next_batch(self):
        # TODO: Support other batch sizes
        # if self._count and self._count % 20 == 0:
        #     logger.info(
        #         "Dataset delivery frame {} ({:.2f} fps)".format(
        #             self._count + 1, 1.0 / max(1e-5, self._timer.average_time)
        #         )
        #     )
        # self._timer.tic()
        current_count = self._count.item()
        if current_count == len(self) or (
            self._max_frames and current_count >= self._max_frames
        ):
            raise StopIteration
        frames_inscribed_images = []
        frames_imgs = []
        frames_original_imgs = []
        ids = []
        # frame_sizes = []
        for batch_item_number in range(self._batch_size):
            # Read image
            #_, img0 = self.cap.read()  # BGR
            _, img0 = self._read_next_image()
            if img0 is None:
                print(f"Error loading frame: {self._count + self._start_frame_number}")
                raise StopIteration()

            if self.clip_original:
                img0 = img0[
                    self.clip_original[1] : self.clip_original[3],
                    self.clip_original[0] : self.clip_original[2],
                    :,
                ]

            (
                img,
                inscribed_image,
                self.letterbox_ratio,
                self.letterbox_dw,
                self.letterbox_dh,
            ) = letterbox(
                img0,
                height=self.process_height,
                width=self.process_width,
            )
            if self.width_t is None:
                self.width_t = torch.tensor([img.shape[1]], dtype=torch.int64)
                self.height_t = torch.tensor([img.shape[0]], dtype=torch.int64)
            frames_inscribed_images.append(
                torch.from_numpy(inscribed_image.transpose(2, 0, 1))
            )
            frames_imgs.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
            frames_original_imgs.append(torch.from_numpy(img0.transpose(2, 0, 1)))
            ids.append(self._count + 1 + batch_item_number)
            # frame_sizes.append(torch.tensor([self.height_t, self.width_t], dtype=torch.int64, device=self.height_t.device))

        inscribed_image = torch.stack(frames_inscribed_images, dim=0)
        img = torch.stack(frames_imgs, dim=0).to(torch.float32).contiguous()
        original_img = torch.stack(frames_original_imgs, dim=0).contiguous()
        # Does this need to be in imgs_info this way as an array?
        ids = torch.cat(ids, dim=0)
        # frame_sizes = torch.stack(frame_sizes, dim=0)

        imgs_info = [
            self.height_t.repeat(len(ids))
            if self._multi_width_img_info
            else self.height_t,
            self.width_t.repeat(len(ids))
            if self._multi_width_img_info
            else self.width_t,
            ids,
            self.video_id.repeat(len(ids)),
            [self._path],
        ]

        # TODO: remove ascontiguousarray?
        img /= 255.0

        self._count += self._batch_size
        # self._timer.toc()
        return original_img, img, inscribed_image, imgs_info, ids

    def __next__(self):
        self._timer.tic()
        self._to_worker_queue.put("ok")
        results = self._from_worker_queue.get()
        if isinstance(results, Exception):
            if not isinstance(results, StopIteration):
                print(results)
                traceback.print_exc()
            self.close()
            raise results
        self._timer.toc()

        if self._next_counter and self._next_counter % 20 == 0:
            logger.info(
                "Dataset delivery frame {} ({:.2f} fps)".format(
                    self._count + 1, 1.0 / max(1e-5, self._timer.average_time)
                )
            )
        self._next_counter += 1
        return results

    def __len__(self):
        if self.vn is None:
            cap = cv2.VideoCapture(self._path)
            self.vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return self.vn  # number of frames
