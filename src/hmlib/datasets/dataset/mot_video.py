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
        mot_eval_mode: bool = False,
        video_id: int = 1,
        data_dir=None,
        json_file: str = "train_half.json",
        name: str = "train",
        preproc=None,
        return_origin_img=False,
        batch_size: int = 1,
        start_frame_number: int = 0,
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
        self.width_t = None
        self.height_t = None
        self.count = torch.tensor([0], dtype=torch.int32)
        self.video_id = torch.tensor([video_id], dtype=torch.int32)
        self._last_size = None
        self._mot_eval_mode = mot_eval_mode
        self._batch_size = batch_size
        self._timer = None
        self._to_worker_queue = multiprocessing.Queue()
        self._from_worker_queue = multiprocessing.Queue()
        self.vn = None

        assert self._batch_size == 1 and "Only batch size of one supported atm"

        self._start_worker()

    def _open_video(self):
        self.cap = cv2.VideoCapture(self._path)
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self._start_frame_number >= 0 and self._start_frame_number < self.vn
        if self._start_frame_number:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame_number)
        print("Lenth of the video: {:d} frames".format(self.vn))

    @property
    def dataset(self):
        return self

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def letterbox_dw_dh(self):
        return self.letterbox_dw, self.letterbox_dh

    # def set_frame_number(self, frame_id: int):
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    # def get_size(self, vw, vh, dw, dh):
    #     wa, ha = float(dw) / vw, float(dh) / vh
    #     a = min(wa, ha)
    #     size = int(vw * a), int(vh * a)
    #     if self._last_size is not None:
    #         assert size == self._last_size
    #     else:
    #         self._last_size = size
    #     return size

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
            self._from_worker_queue.put(ex)
            return
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

    def __iter__(self):
        self.count = torch.tensor([-1], dtype=torch.int32)
        self._timer = Timer()
        return self

    def _get_next_batch(self):
        # if self.count and self.count % 20 == 0:
        #     logger.info(
        #         "Dataset delivery frame {} ({:.2f} fps)".format(
        #             self.count + 1, 1.0 / max(1e-5, self._timer.average_time)
        #         )
        #     )
        # self._timer.tic()
        self.count += 1
        if self.count.item() == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        if img0 is None:
            print(f"Error loading frame: {self.count}")
            raise StopIteration()

        if self.clip_original:
            img0 = img0[
                self.clip_original[1] : self.clip_original[3],
                self.clip_original[0] : self.clip_original[2],
                :,
            ]

        original_img = img0.copy()

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

        if self._mot_eval_mode:
            imgs_info = [
                self.height_t,
                self.width_t,
                self.count + 1,
                self.video_id,
                [self._path],
            ]

        if self._mot_eval_mode:
            img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(
                2, 0, 1
            )
            img = img.unsqueeze(0)
        else:
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
        img /= 255.0

        if self._mot_eval_mode:
            original_img = torch.from_numpy(original_img.transpose(2, 0, 1))
            original_img = original_img.unsqueeze(0)
            ids = torch.tensor(imgs_info[2]).unsqueeze(0)
            # self._timer.toc()
            return original_img, img, inscribed_image, imgs_info, ids
        else:
            # self._timer.toc()
            return self.count, img, img0, original_img

    def __next__(self):
        self._timer.tic()
        self._to_worker_queue.put("ok")
        results = self._from_worker_queue.get()
        if isinstance(results, Exception):
            raise results
        self._timer.toc()

        if self._next_counter and self._next_counter % 20 == 0:
            logger.info(
                "Dataset delivery frame {} ({:.2f} fps)".format(
                    self.count + 1, 1.0 / max(1e-5, self._timer.average_time)
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


# class MOTLoadDatasetVideoWithOrig(MOTDataset):  # for inference
#     def __init__(
#         self,
#         dataset,
#         img_size,
#         clip_original=None,
#         mot_eval_mode: bool = False,
#         video_id: int = 1,
#         data_dir=None,
#         json_file: str = "train_half.json",
#         name: str = "train",
#         preproc=None,
#         return_origin_img=False,
#         batch_size: int = 1,
#     ):
#         super().__init__(
#             data_dir=data_dir,
#             json_file=json_file,
#             name=name,
#             preproc=preproc,
#             return_origin_img=return_origin_img,
#         )
#         # self._path = path
#         # self.cap = cv2.VideoCapture(path)
#         self._dataset = dataset
#         # self.vw = self._dataset.vw
#         # self.vh = self._dataset.vh
#         # self.vn = self._dataset.vn
#         self.fps = dataset.fps
#         self.clip_original = clip_original
#         self.process_height = img_size[0]
#         self.process_width = img_size[1]
#         self.width_t = None
#         self.height_t = None
#         self.count = torch.tensor([0], dtype=torch.int32)
#         self.video_id = torch.tensor([video_id], dtype=torch.int32)
#         self._last_size = None
#         self._mot_eval_mode = mot_eval_mode
#         self._batch_size = batch_size
#         self._timer = None

#         assert self._batch_size == 1 and "Only batch size of one supported atm"
#         print("Lenth of the video: {:d} frames".format(self.vn))

#     @property
#     def dataset(self):
#         return self

#     @property
#     def batch_size(self):
#         return self._batch_size

#     @property
#     def letterbox_dw_dh(self):
#         return self.letterbox_dw, self.letterbox_dh

#     def set_frame_number(self, frame_id: int):
#         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

#     def get_size(self, vw, vh, dw, dh):
#         assert False  # does this get called?
#         wa, ha = float(dw) / vw, float(dh) / vh
#         a = min(wa, ha)
#         size = int(vw * a), int(vh * a)
#         if self._last_size is not None:
#             assert size == self._last_size
#         else:
#             self._last_size = size
#         return size

#     def __iter__(self):
#         self.count = torch.tensor([-1], dtype=torch.int32)
#         self._timer = Timer()
#         return self

#     def __next__(self):
#         if self.count and self.count % 20 == 0:
#             logger.info(
#                 "Dataset delivery frame {} ({:.2f} fps)".format(
#                     self.count + 1, 1.0 / max(1e-5, self._timer.average_time)
#                 )
#             )
#         self._timer.tic()
#         self.count += 1
#         if self.count.item() == len(self):
#             raise StopIteration
#         # Read image
#         res, img0 = self.cap.read()  # BGR
#         if img0 is None:
#             print(f"Error loading frame: {self.count}")
#             raise StopIteration()

#         if self.clip_original:
#             img0 = img0[
#                 self.clip_original[1] : self.clip_original[3],
#                 self.clip_original[0] : self.clip_original[2],
#                 :,
#             ]

#         original_img = img0.copy()

#         (
#             img,
#             inscribed_image,
#             self.letterbox_ratio,
#             self.letterbox_dw,
#             self.letterbox_dh,
#         ) = _letterbox(img0, self.process_height, self.process_width)

#         if self.width_t is None:
#             self.width_t = torch.tensor([img.shape[1]], dtype=torch.int64)
#             self.height_t = torch.tensor([img.shape[0]], dtype=torch.int64)

#         if self._mot_eval_mode:
#             imgs_info = [
#                 self.height_t,
#                 self.width_t,
#                 self.count + 1,
#                 self.video_id,
#                 [self._path],
#             ]

#         if self._mot_eval_mode:
#             img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(
#                 2, 0, 1
#             )
#             img = img.unsqueeze(0)
#         else:
#             # Normalize RGB
#             img = img[:, :, ::-1].transpose(2, 0, 1)
#         img /= 255.0

#         if self._mot_eval_mode:
#             # Make the image planar RGB
#             # img = torch.from_numpy(img).permute(0, 2, 1).unsqueeze(0)
#             # print(original_img.shape)
#             # original_img = torch.from_numpy(np.ascontiguousarray(original_img)).permute(
#             #     2, 0, 1
#             # )
#             original_img = torch.from_numpy(
#                 np.ascontiguousarray(original_img.transpose(2, 0, 1))
#             )
#             original_img = original_img.unsqueeze(0)
#             # print(original_img.shape)
#             # for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
#             ids = torch.tensor(imgs_info[2]).unsqueeze(0)
#             self._timer.toc()
#             return original_img, img, inscribed_image, imgs_info, ids
#         else:
#             self._timer.toc()
#             # cv2.imwrite(img_path + '._letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save _letterbox image
#             return self.count, img, img0, original_img

#     def __len__(self):
#         return self.vn  # number of files
