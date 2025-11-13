import contextlib as _contextlib
import threading
import traceback
from contextlib import contextmanager
from threading import Lock
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hmlib.log import get_root_logger
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import show_image
from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import StreamCheckpoint, StreamTensor, cuda_stream_scope
from hmlib.utils.image import image_height, image_width, make_channels_first, make_channels_last
from hmlib.utils.iterators import CachedIterator
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import VideoStreamReader


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

    def __init__(
        self,
        path: Union[str, List[str]],
        game_id: str = None,
        video_id: int = 1,
        preproc=None,
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
        log_messages: bool = False,
        dtype: torch.dtype = None,
        data_pipeline: Callable = None,
        frame_step: int = 1,
        result_as_dict: bool = False,
        adjust_exposure: Optional[float] = None,
        no_cuda_streams: bool = False,
    ):
        if isinstance(path, list):
            self._path_list = path
        elif path:
            self._path_list = [path]
        else:
            self._path_list = None

        if original_image_only and device_for_original_image is None:
            device_for_original_image = device

        self._current_path_index = 0
        self._game_id = game_id
        self._no_cuda_streams = no_cuda_streams
        self._embedded_data_loader_cache_size = embedded_data_loader_cache_size
        # The delivery device of the letterbox image
        self._device = device
        self._decoder_device = decoder_device
        self._preproc = preproc
        self._logger = get_root_logger()
        self._dtype = dtype
        self._frame_step = frame_step
        self._data_pipeline = data_pipeline
        self._log_messages = log_messages
        self._device_for_original_image = device_for_original_image
        self._start_frame_number = start_frame_number
        self.calculated_clip_box = None
        self._result_as_dict = result_as_dict
        self._adjust_exposure = adjust_exposure
        self._multi_width_img_info = multi_width_img_info
        self._original_image_only = original_image_only
        self.width_t = None
        self.height_t = None
        self._scale_color_tensor = None
        self._count = torch.tensor([0], dtype=torch.int64)
        self._next_frame_id = torch.tensor([start_frame_number], dtype=torch.int32)
        if self._next_frame_id == 0:
            # frame number is 1-based
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
        self._cuda_stream = None
        self._decoder_type = None
        self._next_counter = 0
        self._frame_read_count = 0
        self._video_info = None
        self._seek_lock = Lock()

        self.load_video_info()

    def load_video_info(self) -> None:
        if not self._path_list:
            return
        if self._video_info is not None:
            return
        assert self._embedded_data_loader is None
        self._video_info = BasicVideoInfo(",".join(self._path_list))

    def _open_video(self):
        self.load_video_info()
        if self._embedded_data_loader is None:
            self._decoder_type = "cv2"
            if self._decoder_device is not None and self._decoder_device.type == "cuda":
                self._decoder_type = "torchaudio"
            self.cap = VideoStreamReader(
                filename=str(self._path_list[self._current_path_index]),
                type=self._decoder_type,
                batch_size=self._batch_size,
                device=self._decoder_device,
            )
            self.vw = self.cap.width
            self.vh = self.cap.height
            assert self._video_info.width == self.vw and self._video_info.height == self.cap.height
            self.vn = self._video_info.frame_count
            if not self.vn:
                raise RuntimeError(
                    f"Video {str(self._path_list[self._current_path_index])} either does not exist or has no usable video content"
                )
            if self._start_frame_number < 0:
                raise ValueError("Start frame number cannot be negative")
            elif self._start_frame_number >= self.vn:
                raise ValueError(
                    f"Start frame {int(self._start_frame_number)} is beyond the end of the video, which has only {self.vn} frames"
                )
            if self._start_frame_number and self._current_path_index == 0:
                self.cap.seek(frame_number=self._start_frame_number * self._frame_step)
            self._vid_iter = iter(self.cap)
        else:
            self.vn = len(self._embedded_data_loader)
        if self.vn is not None and self._log_messages:
            print("Lenth of the video: {:d} frames".format(self.vn))

    def _close_video(self):
        if self.cap is not None:
            self.cap.close()
            self.cap = None
        if self._embedded_data_loader is not None:
            self._embedded_data_loader.close()

    def goto_next_video(self) -> bool:
        if not self._path_list:
            return False
        if self._current_path_index + 1 >= len(self._path_list):
            return False
        assert self._embedded_data_loader is None
        self._current_path_index += 1
        self._close_video()
        self._open_video()
        return True

    @property
    def fps(self) -> float:
        if self._embedded_data_loader is not None:
            return self._embedded_data_loader.fps
        return self._video_info.fps

    @property
    def bit_rate(self) -> int:
        if self._embedded_data_loader is not None:
            return self._embedded_data_loader.bit_rate
        return self._video_info.bit_rate

    @property
    def dataset(self):
        return self

    @property
    def batch_size(self):
        if self._embedded_data_loader is not None:
            return self._embedded_data_loader.batch_size * self._batch_size
        return self._batch_size

    @property
    def letterbox_dw_dh(self):
        return self.letterbox_dw, self.letterbox_dh

    def _next_frame_worker(self):
        try:
            self._open_video()

            # Create the CUDA stream outside of the main loop
            if self._cuda_stream is None and self._device.type == "cuda":
                self._cuda_stream = torch.cuda.Stream(self._device)

            while True:
                cmd = self._to_worker_queue.get()
                if cmd.startswith("seek:"):
                    # It's a seek command!  Totally untested so far...
                    seek_to_frame = int(cmd.split(":"))
                    assert self.cap is not None
                    self.cap.seek(frame_number=seek_to_frame)
                    self._vid_iter = iter(self.cap)
                    self._from_worker_queue.put("seek_ok")
                    continue
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
        self._thread = None
        if self._path_list is not None:
            print(
                f"MOTLoadVideoWithOrig delivered {self._frame_read_count} frames for file {str(self._path_list[self._current_path_index])}"
            )

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
        img = None
        if self.cap is not None:
            for _ in range(self._frame_step):
                img = next(self._vid_iter)
        else:
            if self._batch_size == 1:
                img = next(self._embedded_data_loader_iter)
            else:
                imgs = []
                for _ in range(self._batch_size):
                    imgs.append(next(self._embedded_data_loader_iter))
                assert imgs[0].ndim == 4
                if isinstance(imgs[0], np.ndarray):
                    img = np.concatenate(imgs, axis=0)
                else:
                    img = torch.cat(imgs, dim=0)
        assert img is not None
        if self.vw is None:
            self.vw = image_width(img)
            self.vh = image_height(img)
        return True, img

    def _is_cuda(self):
        return self._device.type == "cuda"

    def _get_next_batch(self):
        current_count = self._count.item()
        if current_count >= len(self) * self._batch_size or (
            self._max_frames and current_count >= self._max_frames
        ):
            raise StopIteration

        # BEGIN READ NEXT IMAGE
        while True:
            try:
                prof = getattr(self, "_profiler", None)
                rctx = prof.rf("dataloader.read_next_image") if getattr(prof, "enabled", False) else _contextlib.nullcontext()
                with rctx:
                    res, img0 = self._read_next_image()
                if not res or img0 is None:
                    print(f"Error loading frame: {self._count + self._start_frame_number}")
                    raise StopIteration()
                else:
                    break
            except StopIteration:
                if self.goto_next_video():
                    continue
                raise
        # END READ NEXT IMAGE

        default_stream = (
            torch.cuda.current_stream(img0.device) if (isinstance(img0, torch.Tensor) and self._is_cuda()) else None
        )
        cuda_stream = self._cuda_stream

        with cuda_stream_scope(cuda_stream), torch.no_grad():

            if default_stream is not None and cuda_stream is not None:
                cuda_stream.wait_stream(default_stream)

            if isinstance(img0, np.ndarray):
                img0 = torch.from_numpy(img0)

            if img0.ndim == 3:
                assert self._batch_size == 1
                img0 = img0.unsqueeze(0)
            assert img0.ndim == 4

            if self._preproc is not None:
                prof = getattr(self, "_profiler", None)
                pctx = prof.rf("dataloader.preproc") if getattr(prof, "enabled", False) else _contextlib.nullcontext()
                with pctx:
                    img0 = self._preproc(img0)

            if self._device.type != "cpu" and img0.device != self._device:
                img0 = img0.to(self._device, non_blocking=True)

            if (
                self._adjust_exposure is not None
                and self._adjust_exposure != 0
                and self._adjust_exposure != 1
            ):
                if isinstance(img0, StreamTensor):
                    img0 = img0.get()
                if not torch.is_floating_point(img0.dtype):
                    img0 = img0.to(torch.float, non_blocking=True)
                img0 = img0 * self._adjust_exposure

            # Does this need to be in imgs_info this way as an array?
            ids = torch.tensor(
                [int(self._next_frame_id + i) for i in range(len(img0))],
                dtype=torch.int64,
            )
            self._next_frame_id += len(ids)

            if self._data_pipeline is not None:
                if isinstance(img0, StreamTensor):
                    img0 = img0.get()

                if True:
                    orig_dtype = img0.dtype
                    if not torch.is_floating_point(img0):
                        img0 = img0.to(torch.float32)
                    img0 = make_channels_first(img0)
                    img0 *= (
                        torch.tensor([1.15, 1.15, 1.25], dtype=img0.dtype)
                        .to(device=img0.device, non_blocking=True)
                        .view(1, 3, 1, 1)
                    )
                    img0 = make_channels_last(img0)
                    if orig_dtype != img0.dtype:
                        img0 = img0.clamp(0.0, 255.0).to(orig_dtype)

                # show_image("img0", img0, wait=True)

                original_img0 = img0

                if torch.is_floating_point(img0) and img0.dtype != self._dtype:
                    img0 = img0.to(dtype=self._dtype, non_blocking=True)

                data_item = dict(
                    img=make_channels_last(img0),
                    img_info=dict(frame_id=ids[0]),
                    img_prefix=None,
                    img_id=ids,
                )
                data_item = self._data_pipeline(data_item)
                assert "img" not in data_item
                data_item["img"] = data_item.pop("inputs")
                img = data_item["img"]

                # Maybe get back the clipped image as the "original"
                if "clipped_image" in data_item:
                    assert False and "No longer supported"
                    clipped_image = data_item["clipped_image"]
                    if isinstance(clipped_image, list):
                        assert len(clipped_image) == 1
                        clipped_image = clipped_image[0]
                    if clipped_image is not None:
                        original_img0 = clipped_image["img"]
                        del data_item["clipped_image"]

                if isinstance(img, list):
                    assert len(img) == 1
                    img = img[0]
                    data = data_item
                else:
                    # mmcv2, trying not to use that weird DataContainer class,
                    # whatever the Hell that is supposed to be for
                    data = data_item
            else:
                original_img0 = img0

                if not self._original_image_only:
                    if self._dtype is not None and img0.dtype != self._dtype:
                        img0 = img0.to(self._dtype, non_blocking=True)
                    assert False  # Don't use this path anymore
                    img = self.make_letterbox_images(make_channels_first(img0))
                else:
                    if self._dtype is not None and self._dtype != original_img0.dtype:
                        original_img0 = original_img0.to(self._dtype, non_blocking=True)
                    img = original_img0

            if self.width_t is None:
                self.width_t = torch.tensor([image_width(img)], dtype=torch.int64)
                self.height_t = torch.tensor([image_height(img)], dtype=torch.int64)

            original_img0 = make_channels_first(original_img0)

            path = str(self._path_list[self._current_path_index]) if self._path_list else None
            imgs_info = [
                (self.height_t.repeat(len(ids)) if self._multi_width_img_info else self.height_t),
                (self.width_t.repeat(len(ids)) if self._multi_width_img_info else self.width_t),
                ids,
                self.video_id.repeat(len(ids)),
                [path if path is not None else self._game_id],
            ]

        self._count += self._batch_size

        # BEGIN _wrap_original_image
        # Do this last, after any cuda events are taken for the non-original image
        def _wrap_original_image(
            orig_img: torch.Tensor,
        ) -> Union[StreamCheckpoint, torch.Tensor]:
            assert isinstance(orig_img, torch.Tensor)
            if cuda_stream is not None:
                if (
                    not self._original_image_only
                    and self._device_for_original_image is not None
                    and self._device_for_original_image != orig_img.device
                ):
                    orig_img = orig_img.to(self._device_for_original_image, non_blocking=True)
                return StreamCheckpoint(
                    tensor=orig_img,
                )
            return orig_img

        # END _wrap_original_image

        if self._data_pipeline is not None:
            if cuda_stream is not None:
                if isinstance(data["img"], list):
                    # mmcv1 path
                    for i, img in enumerate(data["img"]):
                        img = StreamCheckpoint(
                            tensor=img,
                        )
                else:
                    # New mmcv2 path
                    assert isinstance(data["img"], torch.Tensor)
                    data["img"] = StreamCheckpoint(tensor=data["img"])
            prof = getattr(self, "_profiler", None)
            dctx = prof.rf("dataloader.data_pipeline") if getattr(prof, "enabled", False) else _contextlib.nullcontext()
            with dctx:
                if self._result_as_dict:
                    return dict(
                        original_imgs=_wrap_original_image(original_img0),
                        data=data,
                        imgs_info=imgs_info,
                        ids=ids,
                    )
                return _wrap_original_image(original_img0), data, None, imgs_info, ids
        if self._original_image_only:
            if not isinstance(original_img0, StreamTensor):
                original_img0 = _wrap_original_image(original_img0)
            if self._result_as_dict:
                return dict(
                    original_imgs=original_img0,
                    imgs_info=imgs_info,
                    ids=ids,
                )
            return dict(img=original_img0, imgs_info=imgs_info, frame_ids=ids)
        else:
            if cuda_stream is not None:
                img = StreamCheckpoint(
                    tensor=img,
                )
            if self._result_as_dict:
                return dict(
                    original_imgs=_wrap_original_image(original_img0),
                    img=img,
                    imgs_info=imgs_info,
                    ids=ids,
                )
            return _wrap_original_image(original_img0), img, None, imgs_info, ids

    def __next__(self):
        with self._seek_lock:
            self._timer.tic()
            prof = getattr(self, "_profiler", None)
            qctx = prof.rf("dataloader.dequeue") if getattr(prof, "enabled", False) else _contextlib.nullcontext()
            with qctx:
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
            self._frame_read_count += 1
            return results

    def tell(self) -> int:
        """
        Get which frame the dataset is on (which one it will load next)
        """
        return self._next_frame_id.clone()

    def seek(self, frame_number: int) -> None:
        """
        Seek to the frame number
        """
        with self._seek_lock:
            assert frame_number > 0  # 1-based
            if frame_number == self._next_frame_id:
                return
            self._to_worker_queue.put(f"seek:{frame_number}")
            while True:
                response = self._from_worker_queue.get()
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, str) and response == "seek_ok":
                    break
                    # otherwise it's probably a queued frame, so discard it
        return

    def __len__(self):
        if self._embedded_data_loader is not None:
            return len(self._embedded_data_loader)
        if self.vn is None:
            self.vn = self._video_info.frame_count
        return self.vn // self._batch_size  # number of frames

    # Optional: external wiring for profiler instance
    def set_profiler(self, profiler):
        self._profiler = profiler
