"""
Experiments in stitching
"""

import os
import cv2
import threading
import torch
import numpy as np
import time
from typing import Tuple
import multiprocessing

import queue
from typing import List

from hockeymom import core

from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer

from hmlib.stitching.remapper import create_remapper_config
from hmlib.stitching.blender import create_blender_config, create_stitcher
from hmlib.stitching.laplacian_blend import show_image

from hmlib.utils.image import (
    make_channels_last,
    make_channels_first,
    image_height,
    image_width,
)

# Some arbitrarily huge number of frames
_LARGE_NUMBER_OF_FRAMES = 1e128


class FrameRequest:
    def __init__(self, frame_id: int, want_alpha: bool = False):
        self.frame_id = frame_id
        self.want_alpha = want_alpha


class WorkerInfoReturned:
    def __init__(
        self,
        total_num_frames: int,
        last_requested_frame: int,
        process_id: int = os.getpid(),
    ):
        self.process_id = process_id
        self.total_num_frames = total_num_frames
        self.last_requested_frame = last_requested_frame


_VERBOSE = True


def INFO(*args, **kwargs):
    if not _VERBOSE:
        return
    print(*args, **kwargs)


def create_queue(mp: bool):
    if mp:
        return multiprocessing.Queue()
    else:
        return queue.Queue()


def safe_put_queue(queue, object):
    try:
        queue.put(object)
    except BrokenPipeError:
        # Ignore broken pipe error
        return
    return


##
#   _____ _   _  _        _     _           __          __         _
#  / ____| | (_)| |      | |   (_)          \ \        / /        | |
# | (___ | |_ _ | |_  ___| |__  _ _ __   __ _\ \  /\  / /___  _ __| | __ ___  _ __
#  \___ \| __| || __|/ __| '_ \| | '_ \ / _` |\ \/  \/ // _ \| '__| |/ // _ \| '__|
#  ____) | |_| || |_| (__| | | | | | | | (_| | \  /\  /| (_) | |  |   <|  __/| |
# |_____/ \__|_| \__|\___|_| |_|_|_| |_|\__, |  \/  \/  \___/|_|  |_|\_\\___||_|
#                                        __/ |
#                                       |___/
#
class StitchingWorker:
    def __init__(
        self,
        video_file_1: str,
        video_file_2: str,
        rank: int,
        batch_size: int = 1,
        pto_project_file: str = None,
        video_1_offset_frame: int = None,
        video_2_offset_frame: int = None,
        start_frame_number: int = 0,
        max_input_queue_size: int = 50,
        remap_thread_count: int = 10,
        blend_thread_count: int = 10,
        max_frames: int = None,
        frame_stride_count: int = 1,
        save_seams_and_masks: bool = True,
        remapping_device: torch.device = torch.device("cuda", 0),
        multiprocessingt_queue: bool = False,
        image_roi: List[int] = None,
        use_pytorch_remap: bool = True,
        blend_mode: str = "laplacian",
    ):
        assert max_input_queue_size > 0

        self._rank = rank
        self._batch_size = batch_size
        self._use_pytorch_remap = use_pytorch_remap

        self._remapping_device = remapping_device
        self._blend_mode = blend_mode

        self._start_frame_number = start_frame_number
        self._output_video = None
        self._video_1_offset_frame = video_1_offset_frame
        self._video_2_offset_frame = video_2_offset_frame
        self._video_file_1 = video_file_1
        self._video_file_2 = video_file_2
        self._pto_project_file = pto_project_file
        self._max_frames = (
            max_frames if max_frames is not None else _LARGE_NUMBER_OF_FRAMES
        )
        self._max_input_queue_size = min(max_input_queue_size, self._max_frames)
        self._remap_thread_count = remap_thread_count
        self._blend_thread_count = blend_thread_count
        # self._to_worker_queue = create_queue(mp=multiprocessingt_queue)
        # self._from_worker_queue = create_queue(mp=multiprocessingt_queue)
        # self._image_response_queue = create_queue(mp=multiprocessingt_queue)
        self._shutdown_barrier = None
        self._frame_stride_count = frame_stride_count
        self._open = False
        self._last_requested_frame = None
        # self._feeder_thread = None
        self._image_roi = image_roi
        self._forked = False
        self._closing = False
        self._save_seams_and_masks = save_seams_and_masks
        # self._in_queue = 0
        self._waiting_for_frame_id = None

        self._receive_timer = Timer()
        self._receive_count = 0
        self._cuda_stream = torch.cuda.Stream(device=self._remapping_device)

        self._next_frame_timer = Timer()

    def _rp_str(self):
        """Worker rank prefix string."""
        return "WR[" + str(self._rank) + "] "

    def _prepare_image(self, img):
        if not img.flags["C_CONTIGUOUS"]:
            img = img.ascontiguousarray(img)
        return img

    def _is_ready_to_exit(self):
        try:
            self._to_worker_queue.get_nowait()
            return True
        except queue.Empty:
            return False

    def start(self, fork: bool):
        # if fork:
        #     self._forked = True
        #     self._shutdown_barrier = multiprocessing.Barrier(2)
        #     if not os.fork():
        #         self._open_videos()
        #         self._from_worker_queue.put(
        #             (
        #                 "openned",
        #                 WorkerInfoReturned(
        #                     total_num_frames=self._total_num_frames,
        #                     last_requested_frame=self._last_requested_frame,
        #                 ),
        #             )
        #         )
        #         self._shutdown_barrier.wait()
        #         print(f"{self._rp_str()} Stitching worker shutting down")
        #     else:
        #         ack, worker_info = self._from_worker_queue.get()
        #         assert ack == "openned"
        #         self._last_requested_frame = worker_info.last_requested_frame
        #         self._total_num_frames = worker_info.total_num_frames
        #         self._open = True
        # else:
        #     self._open_videos()
        self._open_videos()

    def receive_image(self, expected_frame_id):
        stitched_frame = self._feed_next_frame(frame_id=expected_frame_id)
        # self._cuda_stream.synchronize()
        return stitched_frame

        # self._receive_timer.tic()
        # # INFO(f"rank {self._rank} ASK StitchingWorker.receive_image {frame_id}")
        # result = self._image_response_queue.get()
        # if isinstance(result, Exception):
        #     raise result
        # fid, image = result
        # # INFO(f"{self._rp_str()} GOT StitchingWorker.receive_image {fid}")
        # assert fid == expected_frame_id
        # self._receive_timer.toc()
        # self._receive_count += 1
        # return image

    def _initialize_from_initial_frame(self):
        if self._rank == 0:
            # Rank 0 will process the first frame normally
            return

        # TODO: Pass first frame through stitcher in order
        #       to init all blenders from the same image

        for _ in range(self._rank * self._batch_size):
            # print(f"rank {self._rank} pre-reading frame {i}")
            res1, _ = self._video1.read()
            res2, _ = self._video2.read()
            if not res1 or not res2:
                raise StopIteration()
        self._start_frame_number += self._rank

    def _open_videos(self):
        self._video1 = cv2.VideoCapture(self._video_file_1)
        self._video2 = cv2.VideoCapture(self._video_file_2)
        if self._start_frame_number or self._video_1_offset_frame:
            self._video1.set(
                cv2.CAP_PROP_POS_FRAMES,
                self._start_frame_number + self._video_1_offset_frame,
            )
        if self._start_frame_number or self._video_2_offset_frame:
            self._video2.set(
                cv2.CAP_PROP_POS_FRAMES,
                self._start_frame_number + self._video_2_offset_frame,
            )

        self._stitcher, self._xy_pos_1, self._xy_pos_2 = create_stitcher(
            dir_name=os.path.dirname(self._video_file_1),
            batch_size=self._batch_size,
            device=self._remapping_device,
        )
        self._stitcher.to(self._remapping_device)

        # self._stitcher = core.StitchingDataLoader(
        #     0,
        #     self._pto_project_file,
        #     os.path.splitext(self._pto_project_file)[0] + ".seam.png",
        #     os.path.splitext(self._pto_project_file)[0] + ".xor_mask.png",
        #     self._save_seams_and_masks,
        #     self._max_input_queue_size,
        #     (
        #         self._remap_thread_count
        #         if (not self._blend_mode or self._blend_mode == "multiblend")
        #         else 1
        #     ),
        #     (
        #         self._blend_thread_count
        #         if (not self._blend_mode or self._blend_mode == "multiblend")
        #         else 1
        #     ),
        # )

        # if self._use_pytorch_remap:
        #     self._remapper_config_1 = create_remapper_config(
        #         dir_name=os.path.dirname(self._video_file_1),
        #         image_index=0,
        #         basename="mapping_",
        #         source_hw=(
        #             int(self._video1.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        #             int(self._video1.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #         ),
        #         batch_size=self._batch_size,
        #         device=self._remapping_device,
        #     )

        #     self._remapper_config_2 = create_remapper_config(
        #         dir_name=os.path.dirname(self._video_file_2),
        #         image_index=1,
        #         basename="mapping_",
        #         source_hw=(
        #             int(self._video2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        #             int(self._video2.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #         ),
        #         batch_size=self._batch_size,
        #         device=self._remapping_device,
        #     )
        #     self._stitcher.configure_remapper(
        #         remapper_config=[
        #             self._remapper_config_1,
        #             self._remapper_config_2,
        #         ]
        #     )

        # if self._blend_mode != "multiblend":
        #     self._blender_config = create_blender_config(
        #         mode=self._blend_mode,
        #         dir_name=os.path.dirname(self._video_file_1),
        #         basename="mapping_",
        #         device=self._remapping_device,
        #         levels=4,
        #         lazy_init=True,
        #         interpolation="bilinear",
        #     )
        #     self._stitcher.configure_blender(blender_config=self._blender_config)

        self._initialize_from_initial_frame()

        # TODO: adjust max frames based on this
        self._total_num_frames = min(
            int(self._video1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self._video2.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        max_frames = min(
            self._max_frames, self._total_num_frames // self._frame_stride_count
        )
        self._end_frame = self._start_frame_number + (
            max_frames * self._frame_stride_count
        )
        assert self._end_frame >= 0

        # self._start_feeder_thread()
        self._open = True

    def close(self, in_process: bool = False):
        if self._forked and not in_process:
            safe_put_queue(self._to_worker_queue, None)
            safe_put_queue(self._to_worker_queue, None)
        else:
            self._stop_child_threads()
            self._video1.release()
            self._video2.release()
            if self._output_video is not None:
                self._output_video.release()
            self._open = False
        if self._shutdown_barrier is not None:
            self._shutdown_barrier.wait()

    def _read_frame_batch(self):
        frames_1 = []
        frames_2 = []
        for _ in range(self._batch_size):
            ret1, img1, ret2, img2 = self._read_next_frame_from_video()
            if not ret1 or not ret2:
                return False, None, False, None
            frames_1.append(
                torch.from_numpy(img1).to(self._remapping_device, non_blocking=True)
            )
            frames_2.append(
                torch.from_numpy(img2).to(self._remapping_device, non_blocking=True)
            )
        frames_1 = torch.stack(frames_1)
        frames_2 = torch.stack(frames_2)
        return True, frames_1, True, frames_2

    def _read_next_frame_from_video(self):
        ret1, img1 = self._video1.read()
        # Read the corresponding frame from the second video
        ret2, img2 = self._video2.read()
        return ret1, img1, ret2, img2

    def _feed_next_frame(self, frame_id: int) -> bool:
        #with torch.cuda.stream(self._cuda_stream):
        for _ in range(self._frame_stride_count):
            ret1, img1, ret2, img2 = self._read_frame_batch()
            if not ret1 or not ret2:
                return None

        # INFO(f"{self._rp_str()} Adding frame {frame_id} to stitch data loader")
        # For some reason, hold onto a ref to the images while we pushq
        # them down into the data loader, or else there will be a segfault eventually
        # assert self._max_input_queue_size >= 2
        # while self._in_queue >= self._max_input_queue_size:
        #     print("Too many images in the outgoing queue")
        #     time.sleep(0.01)
        # print(f"Adding frame {frame_id} images to stitcher")
        assert img1 is not None
        assert img2 is not None
        # self._in_queue += 1
        # print(f"rank {self._rank} feeding LR frame {frame_id}")

        sinfo_1 = core.StitchImageInfo()
        sinfo_1.image = make_channels_first(img1).to(torch.float, non_blocking=True)
        sinfo_1.xy_pos = self._xy_pos_1

        sinfo_2 = core.StitchImageInfo()
        sinfo_2.image = make_channels_first(img2).to(torch.float, non_blocking=True)
        sinfo_2.xy_pos = self._xy_pos_2

        # main_stream.synchronize()
        # torch.cuda.current_stream().synchronize()
        #self._cuda_stream.synchronize()
        #blended_stream_tensor = self._stitcher.forward(inputs=[sinfo_1, sinfo_2])
        return None

        # if self._use_pytorch_remap:
        #     if isinstance(img1, torch.Tensor):
        #         # Channels first
        #         image_1 = img1.permute(0, 3, 1, 2)
        #         image_2 = img2.permute(0, 3, 1, 2)
        #     else:
        #         image_1 = torch.from_numpy(
        #             np.expand_dims(img1.transpose(2, 0, 1), axis=0)
        #         )
        #         image_2 = torch.from_numpy(
        #             np.expand_dims(img2.transpose(2, 0, 1), axis=0)
        #         )
        #     self._stitcher.add_torch_frame(
        #         frame_id=frame_id,
        #         image_1=image_1,
        #         image_2=image_2,
        #     )
        # else:
        #     core.add_to_stitching_data_loader(
        #         self._stitcher,
        #         frame_id,
        #         self._prepare_image(img1),
        #         self._prepare_image(img2),
        #     )
        # return True
        return blended_stream_tensor / 255.0

    # def _image_getter_worker(self):
    #     pull_timer = Timer()
    #     count = 0
    #     for frame_id in range(
    #         self._start_frame_number, self._end_frame, self._frame_stride_count
    #     ):
    #         if self._is_ready_to_exit():
    #             break
    #         pull_timer.tic()
    #         self._waiting_for_frame_id = frame_id
    #         # if not self._blend_mode or self._blend_mode == "multiblend":
    #         #     stitched_frame = self._stitcher.get_stitched_frame(frame_id)
    #         # else:
    #         #     stitched_frame = self._stitcher.get_stitched_pytorch_frame(frame_id)
    #         # show_image("stitched_frameXX", stitched_frame, wait=True)
    #         # if stitched_frame is None:
    #         #     break
    #         # if isinstance(stitched_frame, torch.Tensor):
    #         #     stitched_frame = stitched_frame.numpy()
    #         # print(stitched_frame.shape)

    #         stitched_frame = self._feed_next_frame(frame_id=frame_id)

    #         assert self._in_queue > 0
    #         self._in_queue -= 1
    #         pull_timer.toc()

    #         # if count % 20 == 0:
    #         #     logger.info(
    #         #         "Pulling stitch frame from stitcher: {} ({:.2f} fps)".format(
    #         #             frame_id,
    #         #             1.0 / max(1e-5, pull_timer.average_time),
    #         #         )
    #         #     )
    #         #     pull_timer = Timer()
    #         while self._image_response_queue.qsize() > 25:
    #             time.sleep(0.1)
    #         self._image_response_queue.put((frame_id, stitched_frame))
    #         count += 1
    #     self._image_response_queue.put(StopIteration())

    # def _frame_feeder_worker(
    #     self,
    #     max_frames: int,
    # ):
    #     for frame_id in range(
    #         self._start_frame_number, self._end_frame, self._frame_stride_count
    #     ):
    #         if self._is_ready_to_exit():
    #             break
    #         # if not self._feed_next_frame(frame_id=frame_id):
    #         #     core.close_stitching_data_loader(self._stitcher, frame_id)
    #         #     break
    #         else:
    #             pass
    #     INFO(f"{self._rp_str()} Feeder thread exiting")

    def _start_feeder_thread(self):
        # self._feeder_thread = threading.Thread(
        #     target=self._frame_feeder_worker,
        #     args=(self._max_frames,),
        # )
        # self._feeder_thread.start()

        # self._image_getter_thread = threading.Thread(
        #     name="_image_getter_worker",
        #     target=self._image_getter_worker,
        #     args=(),
        # )
        # self._image_getter_thread.start()
        pass

    def _stop_child_threads(self):
        # if self._feeder_thread is not None:
        #     safe_put_queue(self._to_worker_queue, None)
        # if self._image_getter_thread is not None:
        #     safe_put_queue(self._to_worker_queue, None)
        # if self._feeder_thread is not None:
        #     self._feeder_thread.join()
        #     self._feeder_thread = None
        # if self._image_getter_thread is not None:
        #     if self._waiting_for_frame_id is not None:
        #         core.close_stitching_data_loader(
        #             self._stitcher, self._waiting_for_frame_id
        #         )
        #     self._image_getter_thread.join()
        #     self._image_getter_thread = None
        pass

    def __len__(self):
        return self._total_num_frames


class StitchingWorkersIterator:
    def __init__(
        self,
        stitching_workers: List[StitchingWorker],
        start_frame_number: int = 0,
        max_frames: int = _LARGE_NUMBER_OF_FRAMES,
    ):
        assert stitching_workers
        self._stitching_workers = stitching_workers
        self._worker_index = 0
        self._max_frames = max_frames
        self._current_frame = start_frame_number
        # self._last_frame = (start_frame_number +  + self._max_frames * len(self._stitching_workers)

    def __next__(self):
        # if self._current_frame >= self._last_frame:
        # raise StopIteration()
        stitching_worker = self._stitching_workers[self._worker_index]
        image = stitching_worker.receive_image(self._current_frame)
        if image is None:
            raise StopIteration()
        self._worker_index = (self._worker_index + 1) % len(self._stitching_workers)
        self._current_frame += 1
        return image
