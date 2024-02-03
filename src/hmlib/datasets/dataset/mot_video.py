import traceback
import multiprocessing
import threading
import numpy as np
from typing import List, Tuple

import cv2
import torch

import torch

from hmlib.tracking_utils.timer import Timer
from hmlib.datasets.dataset.jde import letterbox, py_letterbox
from hmlib.tracking_utils.log import logger
from hmlib.video_out import make_visible_image
from yolox.data import MOTDataset
from hmlib.utils.utils import create_queue

from hmlib.utils.image import (
    make_channels_last,
    make_channels_first,
    image_height,
    image_width,
)


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
        original_image_only: bool = False,
        image_channel_adjustment: Tuple[float, float, float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            preproc=preproc,
            return_origin_img=return_origin_img,
        )
        self._path = path
        # The delivery device of the letterbox image
        self._device = device
        self._start_frame_number = start_frame_number
        self.clip_original = clip_original
        self.calculated_clip_box = None
        self.process_height = img_size[0]
        self.process_width = img_size[1]
        self._multi_width_img_info = multi_width_img_info
        self._original_image_only = original_image_only
        self.width_t = None
        self.height_t = None
        self._image_channel_adjustment = image_channel_adjustment
        self._scale_color_tensor = None
        self._count = torch.tensor([0], dtype=torch.int32)
        self._next_frame_id = torch.tensor([start_frame_number], dtype=torch.int32)
        self.video_id = torch.tensor([video_id], dtype=torch.int32)
        self._last_size = None
        self._max_frames = max_frames
        self._batch_size = batch_size
        self._timer = None
        self._timer_counter = 0
        self._to_worker_queue = create_queue(mp=False)
        self._from_worker_queue = create_queue(mp=False)
        self.vn = None
        self.vw = None
        self.vh = None
        self.cap = None
        self._mapping_offset = None
        self._thread = None
        self._scale_inscribed_to_original = None
        self._embedded_data_loader = embedded_data_loader
        self._embedded_data_loader_iter = None
        assert self._embedded_data_loader is None or path is None

        # Optimize the clip box
        if self.clip_original is not None:
            if isinstance(self.clip_original, (list, tuple)):
                if not any(item is not None for item in self.clip_original):
                    self.clip_original = None

        if self._image_channel_adjustment:
            assert len(self._image_channel_adjustment) == 3

        self._open_video()
        self._close_video()

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
            self.vn = len(self._embedded_data_loader)
            self.fps = self._embedded_data_loader.fps
        if self.vn is not None:
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
        self._thread = threading.Thread(
            target=self._next_frame_worker, name="MOTVideoNextFrameWorker"
        )
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

    def __iter__(self):
        self._timer = Timer()
        if self._embedded_data_loader is not None:
            self._embedded_data_loader_iter = iter(self._embedded_data_loader)
        if self._thread is None:
            self._start_worker()
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

    # def maybe_scale_image_colors(self, image: torch.Tensor):
    #     if not self._image_channel_adjustment:
    #         return image
    #     if self._scale_color_tensor is None:
    #         if isinstance(image, torch.Tensor):
    #             self._scale_color_tensor = torch.tensor(
    #                 self._image_channel_adjustment,
    #                 dtype=torch.float32,
    #                 device=image.device,
    #             )
    #             self._scale_color_tensor = self._scale_color_tensor.view(1, 1, 3)
    #         else:
    #             self._scale_color_tensor = np.array(
    #                 self._image_channel_adjustment, dtype=np.float32
    #             )
    #             self._scale_color_tensor = np.expand_dims(
    #                 np.expand_dims(self._scale_color_tensor, 0), 0
    #             )
    #     if isinstance(image, torch.Tensor):
    #         image = torch.clamp(
    #             image.to(torch.float32) * self._scale_color_tensor, min=0, max=255.0
    #         ).to(torch.uint8)
    #     else:
    #         image = np.clip(
    #             image.astype(np.float32) * self._scale_color_tensor,
    #             a_min=0,
    #             a_max=255.0,
    #         ).astype(np.uint8)
    #     return image

    def scale_letterbox_to_original_image_coordinates(self, yolox_detections):
        assert False
        # Offset the boxes
        if self._mapping_offset is None:
            device = yolox_detections[0].device
            self._mapping_offset = torch.tensor(
                [
                    self.letterbox_dw,
                    self.letterbox_dh,
                    self.letterbox_dw,
                    self.letterbox_dh,
                ],
                dtype=yolox_detections[0].dtype,
                device=device,
            )
            if self._scale_inscribed_to_original.device != device:
                self._scale_inscribed_to_original = (
                    self._scale_inscribed_to_original.to(device)
                )
        if len(yolox_detections):
            # [0:4] detections are tlbr
            for i in range(len(yolox_detections)):
                dets = yolox_detections[i]
                if dets is None:
                    continue
                yolox_detections[i][:, 0:4] -= self._mapping_offset
                # Scale the width and height
                yolox_detections[i][:, 0:4] /= self._scale_inscribed_to_original
        return yolox_detections

    def make_letterbox_images(self, img: torch.Tensor):
        (
            img,
            _,
            self.letterbox_ratio,
            self.letterbox_dw,
            self.letterbox_dh,
        ) = py_letterbox(
            img,
            height=self.process_height,
            width=self.process_width,
        )
        # cv2.imshow("online_im", make_visible_image(img[0]))
        # cv2.waitKey(1)
        return img

    def _get_next_batch(self):
        current_count = self._count.item()
        if current_count == len(self) or (
            self._max_frames and current_count >= self._max_frames
        ):
            raise StopIteration

        ALL_NON_BLOCKING = True
        # ALL_NON_BLOCKING = False

        # frames_inscribed_images = []
        frames_imgs = []
        frames_original_imgs = []
        ids = []
        for batch_item_number in range(self._batch_size):
            # Read image
            res, img0 = self._read_next_image()
            if not res or img0 is None:
                print(f"Error loading frame: {self._count + self._start_frame_number}")
                raise StopIteration()

            inner_batch_size = 1
            if not isinstance(img0, torch.Tensor):
                img0 = torch.from_numpy(img0)
            else:
                assert img0.ndim == 4
                inner_batch_size = len(img0)
            original_img0 = img0.clone()
            img0 = img0.to(self._device, non_blocking=ALL_NON_BLOCKING)

            if self.clip_original is not None:
                # assert False  # do this on GPU
                # Clipping not handled now due to "original_img = img0.clone()" above
                if self.calculated_clip_box is None:
                    self.calculated_clip_box = fix_clip_box(
                        self.clip_original, [image_height(img0), image_width(img0)]
                    )
                if len(img0.shape) == 4:
                    img0 = img0[
                        :,
                        self.calculated_clip_box[1] : self.calculated_clip_box[3],
                        self.calculated_clip_box[0] : self.calculated_clip_box[2],
                        :,
                    ]
                else:
                    assert len(img0.shape) == 3
                    img0 = img0[
                        self.calculated_clip_box[1] : self.calculated_clip_box[3],
                        self.calculated_clip_box[0] : self.calculated_clip_box[2],
                        :,
                    ]
                # self.clip_original = fix_clip_box(self.clip_original, img0.shape[:2])
                # img0 = img0[
                #     self.clip_original[1] : self.clip_original[3],
                #     self.clip_original[0] : self.clip_original[2],
                #     :,
                # ]
                # original_img0 = img0.to("cpu", non_blocking=True)
                # original_img0 = img0.to("cpu")
                original_img0 = img0.clone()

            if not self._original_image_only:
                img0 = img0.to(torch.float32, non_blocking=ALL_NON_BLOCKING)

            if not self._original_image_only:
                if img0.ndim != 4:
                    img0 = img0.unsqueeze(0)
                img = self.make_letterbox_images(make_channels_first(img0))
            else:
                img = img0

            if self.width_t is None:
                self.width_t = torch.tensor([img.shape[-1]], dtype=torch.int64)
                self.height_t = torch.tensor([img.shape[-2]], dtype=torch.int64)

            if not self._original_image_only:
                frames_imgs.append(img.to(torch.float32))

            frames_original_imgs.append(
                make_channels_first(original_img0)
                # make_channels_first(original_img0).to(
                #     "cpu", non_blocking=ALL_NON_BLOCKING
                # )
            )
            for _ in range(inner_batch_size):
                self._next_frame_id += 1
                ids.append(self._next_frame_id.clone())

        if not self._original_image_only:
            if frames_imgs[0].ndim == 3:
                img = torch.stack(frames_imgs, dim=0).to(torch.float32)
            else:
                img = torch.cat(frames_imgs, dim=0).to(torch.float32)

        if frames_original_imgs[0].ndim == 3:
            original_img = torch.stack(frames_original_imgs, dim=0)
        else:
            original_img = torch.cat(frames_original_imgs, dim=0)
        # Does this need to be in imgs_info this way as an array?
        ids = torch.cat(ids, dim=0)

        imgs_info = [
            (
                self.height_t.repeat(len(ids))
                if self._multi_width_img_info
                else self.height_t
            ),
            (
                self.width_t.repeat(len(ids))
                if self._multi_width_img_info
                else self.width_t
            ),
            ids,
            self.video_id.repeat(len(ids)),
            [self._path if self._path is not None else "external"],
        ]

        if not self._original_image_only:
            img /= 255.0

        self._count += self._batch_size
        if self._original_image_only:
            # return original_img.cpu(), None, None, imgs_info, ids
            return original_img, None, None, imgs_info, ids
        else:
            return original_img, img, None, imgs_info, ids

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

        self._timer_counter += self._batch_size
        if self._next_counter and self._next_counter % 20 == 0:
            logger.info(
                "Video Dataset frame delivery {} ({:.2f} fps)".format(
                    self._timer_counter,
                    self._batch_size * 1.0 / max(1e-5, self._timer.average_time),
                )
            )
            if self._next_counter and self._next_counter % 100 == 0:
                self._timer = Timer()
                self._timer_counter = 0
        self._next_counter += 1
        return results

    def __len__(self):
        if self.vn is None:
            cap = cv2.VideoCapture(self._path)
            self.vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return self.vn  # number of frames


def is_none(val):
    if isinstance(val, str) and val == "None":
        return True
    return val is None


def fix_clip_box(clip_box, hw: List[int]):
    if isinstance(clip_box, list):
        if is_none(clip_box[0]):
            clip_box[0] = 0
        if is_none(clip_box[1]):
            clip_box[1] = 0
        if is_none(clip_box[2]):
            clip_box[2] = hw[1]
        if is_none(clip_box[3]):
            clip_box[3] = hw[0]
        clip_box = np.array(clip_box, dtype=np.int64)
    return clip_box
