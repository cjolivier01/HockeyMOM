import traceback
import threading
from contextlib import contextmanager
import numpy as np
from typing import List, Tuple

import cv2
import torch

from mmdet.datasets.pipelines import Compose

from yolox.data import MOTDataset
from yolox.data.datasets.datasets_wrapper import Dataset

from hmlib.tracking_utils.timer import Timer
from hmlib.datasets.dataset.jde import py_letterbox
from hmlib.tracking_utils.log import logger
from hmlib.utils.utils import create_queue
from hmlib.utils.image import make_channels_last
from hmlib.video_stream import VideoStreamReader
from hmlib.utils.gpu import (
    StreamTensor,
    # StreamTensorToDtype,
    StreamTensorToDevice,
    CachedIterator,
)
from hmlib.video_out import quick_show

from hmlib.utils.image import (
    make_channels_first,
    image_height,
    image_width,
)


@contextmanager
def optional_with(resource):
    """A context manager that works even if the resource is None."""
    if resource is None:
        # If the resource is None, yield nothing but still enter the with block
        yield None
    else:
        # If the resource is not None, use it as a normal context manager
        with resource as r:
            yield r


class MOTLoadVideoWithOrig(Dataset):  # for inference
    # class MOTLoadVideoWithOrig(MOTDataset):  # for inference
    def __init__(
        self,
        path,
        img_size,
        game_id: str = None,
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
        embedded_data_loader_cache_size: int = 6,
        original_image_only: bool = False,
        image_channel_adjustment: Tuple[float, float, float] = None,
        device: torch.device = torch.device("cpu"),
        decoder_device: torch.device = torch.device("cpu"),
        device_for_original_image: torch.device = None,
        stream_tensors: bool = False,
        log_messages: bool = False,
        dtype: torch.dtype = None,
        data_pipeline: Compose = None,
        # scale_rgb_down: bool = False,
        # output_type: torch.dtype = None
    ):
        # super().__init__(
        #     input_dimension=img_size,
        # )
        # super().__init__(
        #     data_dir=data_dir,
        #     json_file=json_file,
        #     name=name,
        #     preproc=preproc,
        #     return_origin_img=return_origin_img,
        # )
        assert not isinstance(img_size, str)
        self._path = path
        self._game_id = game_id
        self._embedded_data_loader_cache_size = embedded_data_loader_cache_size
        # The delivery device of the letterbox image
        self._device = device
        self._decoder_device = decoder_device
        self._preproc = preproc
        self._dtype = dtype
        self._data_pipeline = data_pipeline
        # self._scale_rgb_down = scale_rgb_down
        self._log_messages = log_messages
        self._device_for_original_image = device_for_original_image
        self._start_frame_number = start_frame_number
        # self._output_type = output_type
        self.clip_original = clip_original
        self.calculated_clip_box = None
        if img_size is None:
            self.process_height = None
            self.process_width = None
        else:
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
        if self._next_frame_id == 0:
            self._next_frame_id = 1
        self.video_id = torch.tensor([video_id], dtype=torch.int32)
        self._last_size = None
        self._max_frames = max_frames
        self._batch_size = batch_size
        self._timer = Timer()
        self._timer_counter = 0
        self._to_worker_queue = create_queue(mp=False)
        self._from_worker_queue = create_queue(mp=False)
        self.vn = None
        self.vw = None
        self.vh = None
        self.cap = None
        self._vid_iter = None
        self._mapping_offset = None
        self._thread = None
        self._scale_inscribed_to_original = None
        self._embedded_data_loader = embedded_data_loader
        self._embedded_data_loader_iter = None
        self._stream_tensors = stream_tensors
        # assert self._embedded_data_loader is None or path is None

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
            self.cap = VideoStreamReader(
                filename=self._path,
                type="cv2",
                batch_size=self._batch_size,
                device=self._decoder_device,
            )
            self.vw = self.cap.width
            self.vh = self.cap.height
            self.vn = len(self.cap)
            self.fps = self.cap.fps

            if not self.vn:
                raise RuntimeError(
                    f"Video {self._path} either does not exist or has no usable video content"
                )
            assert self._start_frame_number >= 0 and self._start_frame_number < self.vn
            if self._start_frame_number:
                self.cap.seek(frame_number=self._start_frame_number)
            self._vid_iter = iter(self.cap)
        else:
            self.vn = len(self._embedded_data_loader)
            self.fps = self._embedded_data_loader.fps
        if self.vn is not None and self._log_messages:
            print("Lenth of the video: {:d} frames".format(self.vn))

    def _close_video(self):
        if self.cap is not None:
            self.cap.close()
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
            if self._embedded_data_loader_cache_size:
                self._embedded_data_loader_iter = CachedIterator(
                    iterator=self._embedded_data_loader_iter,
                    cache_size=self._embedded_data_loader_cache_size,
                )
        if self._thread is None:
            self._start_worker()
        return self

    def _read_next_image(self):
        if self.cap is not None:
            img = next(self._vid_iter)
            return True, img
        else:
            try:
                if self._batch_size == 1:
                    img = next(self._embedded_data_loader_iter)
                    if self.vw is None:
                        self.vw = image_width(img)
                        self.vh = image_height(img)
                else:
                    imgs = []
                    for _ in range(self._batch_size):
                        imgs.append(next(self._embedded_data_loader_iter))
                    assert imgs[0].ndim == 4
                    if isinstance(imgs[0], np.ndarray):
                        img = np.concatenate(imgs, axis=0)
                    else:
                        img = torch.cat(imgs, dim=0)
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
    #                 dtype=torch.float,
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
    #             image.to(torch.float) * self._scale_color_tensor, min=0, max=255.0
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

    def _is_cuda(self):
        return self._device.type == "cuda"

    def _get_next_batch(self):
        current_count = self._count.item()
        if current_count == len(self) or (
            self._max_frames and current_count >= self._max_frames
        ):
            raise StopIteration

        ALL_NON_BLOCKING = True

        cuda_stream = None
        if self._stream_tensors and self._is_cuda():
            cuda_stream = torch.cuda.Stream(self._device)

        with optional_with(
            torch.cuda.stream(cuda_stream) if cuda_stream is not None else None
        ):
            # Read image
            res, img0 = self._read_next_image()
            if not res or img0 is None:
                print(f"Error loading frame: {self._count + self._start_frame_number}")
                raise StopIteration()

            if isinstance(img0, StreamTensor):
                img0 = img0.get()
            elif isinstance(img0, np.ndarray):
                img0 = torch.from_numpy(img0)

            if img0.ndim == 3:
                assert self._batch_size == 1
                img0 = img0.unsqueeze(0)
            assert img0.ndim == 4

            if self._preproc is not None:
                img0 = self._preproc(img0)

            if self._device.type != "cpu" and img0.device != self._device:
                img0 = img0.to(self._device, non_blocking=ALL_NON_BLOCKING)

            # Does this need to be in imgs_info this way as an array?
            ids = torch.tensor(
                [int(self._next_frame_id + i) for i in range(len(img0))],
                dtype=torch.int64,
            )
            self._next_frame_id += len(ids)

            if self._data_pipeline is not None:
                original_img0 = img0
                if torch.is_floating_point(img0):
                    # TODO: Can we have a floating point version?
                    # minp = torch.min(img0)
                    # maxp = torch.max(img0)
                    #img0 = torch.clamp(img0 * 255, min=0, max=255).to(torch.uint8)
                    # Normalize expects 0-255 values
                    
                    #mmin, mmax = torch.min(img0), torch.max(img0)
                    
                    img0 = img0 * 255
                    #quick_show(img0, wait=True)
                    
                    #mmin, mmax = torch.min(img0), torch.max(img0)
                    
                    pass
                else:
                    #mmin, mmax = torch.min(img0), torch.max(img0)
                    pass
                #assert img0.shape[0] == 1
                data_item = dict(
                    #img=make_channels_last(original_img0.squeeze(0)).cpu().numpy(),
                    #img=make_channels_last(img0.squeeze(0)),
                    img=make_channels_last(img0),
                    #img=make_channels_first(img0.squeeze(0)),
                    img_info=dict(frame_id=ids[0]),
                    img_prefix=None,
                )
                # Data pipeline is going to expect a uint8 image
                #data_item["img"] = torch.clamp(data_item["img"] * 255, min=0, max=255)
                data_item = self._data_pipeline(data_item)
                # data_item["img"] /= 255
                img = data_item["img"]
                
                if isinstance(img, list):
                    img = img[0]
                    data = data_item
                else:
                    assert False # ?
                    # atm, it isn't test_pipeline
                    assert isinstance(img, torch.Tensor)
                    data_item["img"] = make_channels_first(data_item["img"])
                    assert isinstance(img, torch.Tensor)  # not a list
                    data = dict()
                    for key, val in data_item.items():
                        data[key] = [val]
  
                #mmin, mmax = torch.min(img), torch.max(img)

                #quick_show(torch.clamp(img0 * 255, min=0, max=255).to(torch.uint8), wait=True)
                
            else:
                if self.clip_original is not None:
                    if self.calculated_clip_box is None:
                        self.calculated_clip_box = fix_clip_box(
                            self.clip_original, [image_height(img0), image_width(img0)]
                        )
                    if len(img0.shape) == 4:
                        img0 = img0[
                            :,
                            :,
                            self.calculated_clip_box[1] : self.calculated_clip_box[3],
                            self.calculated_clip_box[0] : self.calculated_clip_box[2],
                        ]
                    else:
                        assert len(img0.shape) == 3
                        img0 = img0[
                            :,
                            self.calculated_clip_box[1] : self.calculated_clip_box[3],
                            self.calculated_clip_box[0] : self.calculated_clip_box[2],
                        ]

                if not self._original_image_only:
                    original_img0 = img0
                    if not torch.is_floating_point(img0):
                        img0 = (
                            img0.to(torch.float, non_blocking=ALL_NON_BLOCKING) / 255.0
                        )
                    img = self.make_letterbox_images(make_channels_first(img0))
                else:
                    original_img0 = img0
                    if self._dtype is not None and self._dtype != original_img0.dtype:
                        was_fp = torch.is_floating_point(original_img0)
                        original_img0 = original_img0.to(
                            self._dtype, non_blocking=ALL_NON_BLOCKING
                        )
                        is_fp = torch.is_floating_point(original_img0)
                        if not was_fp and is_fp:
                            original_img0 /= 255.0
                    img = original_img0

            if self.width_t is None:
                self.width_t = torch.tensor([image_width(img)], dtype=torch.int64)
                self.height_t = torch.tensor([image_height(img)], dtype=torch.int64)

            original_img0 = make_channels_first(original_img0)

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
                [self._path if self._path is not None else self._game_id],
            ]

            # if (
            #     not self._original_image_only
            #     and not img_was_fp
            #     and torch.is_floating_point(img)
            # ):
            #     img /= 255.0
            #     if original_img0.dtype == img.dtype:
            #         original_img0 /= 255.0

            # if (
            #     self._device_for_original_image is not None
            #     and original_img0.device != self._device_for_original_image
            # ):
            #     if original_img0.device.type == "cuda":
            #         # print("Warning: original image is on a different cuda device")
            #         original_img0 = StreamTensorToDevice(
            #             tensor=original_img0,
            #             device=self._device_for_original_image,
            #         )
            #     if True:
            #         original_img0 = StreamTensorToDevice(
            #             tensor=original_img0, device=self._device_for_original_image
            #         )
            #     else:
            #         original_img0 = original_img0.to(
            #             self._device_for_original_image, non_blocking=True
            #         )

        self._count += self._batch_size

        if cuda_stream is not None:
            # if not self._original_image_only:
            #     original_img0 = original_img0.to("cpu", non_blocking=True)
            original_img0 = StreamTensor(
                tensor=original_img0, stream=cuda_stream, event=torch.cuda.Event()
            )

        if self._data_pipeline is not None:
            if cuda_stream is not None:
                for i, img in enumerate(data["img"]):
                    img = StreamTensor(
                        tensor=img, stream=cuda_stream, event=torch.cuda.Event()
                    )
                    data["img"][i] = img
            return original_img0, data, None, imgs_info, ids
        if self._original_image_only:
            return original_img0, None, None, imgs_info, ids
        else:
            if cuda_stream is not None:
                img = StreamTensor(
                    tensor=img, stream=cuda_stream, event=torch.cuda.Event()
                )
            return original_img0, img, None, imgs_info, ids

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
        if self._log_messages and self._next_counter and self._next_counter % 20 == 0:
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
