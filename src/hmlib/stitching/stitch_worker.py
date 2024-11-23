"""
Experiments in stitching
"""

import multiprocessing
import os
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch

from hmlib.log import logger
from hmlib.stitching.blender import create_blender_config, create_stitcher
from hmlib.stitching.remapper import create_remapper_config
from hmlib.tracking_utils.timer import Timer
from hmlib.ui.show import show_image
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
)
from hockeymom import core

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
        assert frame_stride_count > 0
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

    # def _is_ready_to_exit(self):
    #     try:
    #         self._to_worker_queue.get_nowait()
    #         return True
    #     except queue.Empty:
    #         return False

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
        stitched_frame = self._stitch_next_frame(frame_id=expected_frame_id)
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

        self._open = True

    def close(self, in_process: bool = False):
        if self._forked and not in_process:
            safe_put_queue(self._to_worker_queue, None)
            safe_put_queue(self._to_worker_queue, None)
        else:
            self._video1.release()
            self._video2.release()
            if self._output_video is not None:
                self._output_video.release()
            self._open = False
        if self._shutdown_barrier is not None:
            self._shutdown_barrier.wait()

    def _read_frame_batch(self, discard: bool = False):
        frames_1 = []
        frames_2 = []
        for _ in range(self._batch_size):
            ret1, img1, ret2, img2 = self._read_next_frame_from_video()
            if not ret1 or not ret2:
                return False, None, False, None
            if not discard:
                frames_1.append(
                    torch.from_numpy(img1).to(self._remapping_device, non_blocking=True)
                )
                frames_2.append(
                    torch.from_numpy(img2).to(self._remapping_device, non_blocking=True)
                )
        if not discard:
            frames_1 = torch.stack(frames_1)
            frames_2 = torch.stack(frames_2)
        else:
            frames_1 = None
            frames_2 = None
        return True, frames_1, True, frames_2

    def _read_next_frame_from_video(self):
        ret1, img1 = self._video1.read()
        # Read the corresponding frame from the second video
        ret2, img2 = self._video2.read()
        return ret1, img1, ret2, img2

    def _stitch_next_frame(self, frame_id: int) -> bool:
        # with torch.cuda.stream(self._cuda_stream):
        nodiscard = self._frame_stride_count - 1
        for i in range(self._frame_stride_count):
            ret1, img1, ret2, img2 = self._read_frame_batch(discard=(i == nodiscard))
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

        return self._stitcher.forward(inputs=[sinfo_1, sinfo_2])

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
