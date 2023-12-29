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

from hmlib.stitching.remapper import (
    AsyncRemapperWorker,
    PairCallback,
    RemappedPair,
    ImageRemapper,
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
        device: str = None,
        multiprocessingt_queue: bool = False,
        image_roi: List[int] = None,
        use_pytorch_remap: bool = True,
    ):
        assert max_input_queue_size > 0

        self._rank = rank
        self._batch_size = batch_size
        self._use_pytorch_remap = use_pytorch_remap
        self._device = device
        self._is_cuda = self._device and self._device.startswith("cuda")
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
        self._to_worker_queue = create_queue(mp=multiprocessingt_queue)
        self._from_worker_queue = create_queue(mp=multiprocessingt_queue)
        self._image_response_queue = create_queue(mp=multiprocessingt_queue)
        self._shutdown_barrier = None
        self._frame_stride_count = frame_stride_count
        self._open = False
        self._last_requested_frame = None
        self._feeder_thread = None
        self._image_roi = image_roi
        self._forked = False
        self._closing = False
        self._save_seams_and_masks = save_seams_and_masks
        self._in_queue = 0
        self._waiting_for_frame_id = None

        self._remapper_1 = None
        self._remapper_2 = None

        self._receive_timer = Timer()
        self._receive_count = 0

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
        if fork:
            self._forked = True
            self._shutdown_barrier = multiprocessing.Barrier(2)
            if not os.fork():
                self._open_videos()
                self._from_worker_queue.put(
                    (
                        "openned",
                        WorkerInfoReturned(
                            total_num_frames=self._total_num_frames,
                            last_requested_frame=self._last_requested_frame,
                        ),
                    )
                )
                self._shutdown_barrier.wait()
                print(f"{self._rp_str()} Stitching worker shutting down")
            else:
                ack, worker_info = self._from_worker_queue.get()
                assert ack == "openned"
                self._last_requested_frame = worker_info.last_requested_frame
                self._total_num_frames = worker_info.total_num_frames
                self._open = True
        else:
            self._open_videos()

    def receive_image(self, expected_frame_id):
        self._receive_timer.tic()
        # INFO(f"rank {self._rank} ASK StitchingWorker.receive_image {frame_id}")
        result = self._image_response_queue.get()
        if isinstance(result, Exception):
            raise result
        fid, image = result
        # INFO(f"{self._rp_str()} GOT StitchingWorker.receive_image {fid}")
        assert fid == expected_frame_id
        self._receive_timer.toc()
        if self._receive_count and (self._receive_count % 20) == 0:
            logger.info(
                "Received frame frame {} from stitching worker {} ({:.2f} fps)".format(
                    fid,
                    self._rank,
                    1.0 / max(1e-5, self._receive_timer.average_time),
                )
            )
            self._receive_timer = Timer()
        self._receive_count += 1
        return image

    def _initialize_from_initial_frame(self):
        if self._rank == 0:
            # Rank 0 will process the first frame normally
            return

        for _ in range(self._rank):
            # print(f"rank {self._rank} pre-reading frame {i}")
            res1, _ = self._video1.read()
            res2, _ = self._video2.read()
            if not res1 or not res2:
                raise StopIteration()
        self._start_frame_number += self._rank

    def _open_videos(self):
        if self._is_cuda:
            first_frame_1 = 0
            first_frame_2 = 0
            if self._start_frame_number or self._video_1_offset_frame:
                first_frame_1 = self._start_frame_number + self._video_1_offset_frame
            if self._start_frame_number or self._video_2_offset_frame:
                first_frame_2 = (self._start_frame_number + self._video_2_offset_frame,)
            self._video1 = cv2.cudacodec.createVideoReader(self._video_file_1)
            self._video2 = cv2.cudacodec.createVideoReader(self._video_file_2)
            self._video1.setVideoReaderProps(
                cv2.cudacodec.VideoReaderProps_PROP_DECODED_FRAME_IDX, first_frame_1
            )
            self._video2.setVideoReaderProps(
                cv2.cudacodec.VideoReaderProps_PROP_DECODED_FRAME_IDX, first_frame_2
            )
        else:
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

        self._stitcher = core.StitchingDataLoader(
            0,
            self._pto_project_file,
            os.path.splitext(self._pto_project_file)[0] + ".seam.png",
            os.path.splitext(self._pto_project_file)[0] + ".xor_mask.png",
            self._save_seams_and_masks,
            self._max_input_queue_size,
            self._remap_thread_count,
            self._blend_thread_count,
        )

        self._initialize_from_initial_frame()

        # TODO: adjust max frames based on this
        if self._is_cuda:
            ok, f1 = self._video1.get(cv2.CAP_PROP_FRAME_COUNT)
            assert ok
            ok, f2 = self._video2.get(cv2.CAP_PROP_FRAME_COUNT)
            assert ok
            self._total_num_frames = int(min(f1, f2))
        else:
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

        if self._use_pytorch_remap:
            project_dir = os.path.dirname(self._pto_project_file)
            self._pair_callback = PairCallback(callback=self._deliver_remapped_pair)
            remapper_1 = ImageRemapper(
                dir_name=project_dir,
                basename="mapping_0000",
                device=self._device,
                source_hw=[
                    int(self._video1.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(self._video1.get(cv2.CAP_PROP_FRAME_WIDTH)),
                ],
                interpolation=None,
            )
            self._remapper_1 = AsyncRemapperWorker(
                image_remapper=remapper_1,
                pair_callback=self._pair_callback.aggregate_callback_1,
            )
            remapper_2 = ImageRemapper(
                dir_name=project_dir,
                basename="mapping_0001",
                device=self._device,
                source_hw=[
                    int(self._video2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(self._video2.get(cv2.CAP_PROP_FRAME_WIDTH)),
                ],
                interpolation=None,
            )
            self._remapper_2 = AsyncRemapperWorker(
                image_remapper=remapper_2,
                pair_callback=self._pair_callback.aggregate_callback_2,
            )
            self._remapper_1.start(batch_size=self._batch_size)
            self._remapper_2.start(batch_size=self._batch_size)

        self._start_feeder_thread()
        self._open = True

    def _deliver_remapped_pair(self, remapped_pair: RemappedPair):
        # TODO: handle exception and stuff
        assert isinstance(remapped_pair, RemappedPair)
        assert remapped_pair.image_1.shape[0] == 1
        assert remapped_pair.image_2.shape[0] == 1
        remapped_pair.image_1 = (
            remapped_pair.image_1.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )
        remapped_pair.image_2 = (
            remapped_pair.image_2.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )
        self._stitcher.add_remapped_frame(
            remapped_pair.frame_id,
            remapped_pair.image_1[0],
            self._remapper_1.xy_pos,
            remapped_pair.image_2[0],
            self._remapper_2.xy_pos,
        )

    def close(self, in_process: bool = False):
        if self._forked and not in_process:
            safe_put_queue(self._to_worker_queue, None)
            safe_put_queue(self._to_worker_queue, None)
        else:
            if self._use_pytorch_remap:
                self._remapper_1.stop()
                self._remapper_2.stop()
                self._remapper_1 = None
                self._remapper_2 = None
            self._stop_child_threads()
            self._video1.release()
            self._video2.release()
            if self._output_video is not None:
                self._output_video.release()
            self._open = False
        if self._shutdown_barrier is not None:
            self._shutdown_barrier.wait()

    def _read_next_frame_from_video(self):
        if self._is_cuda:
            ret1, img1 = self._video1.nextFrame()
            # Read the corresponding frame from the second video
            ret2, img2 = self._video2.nextFrame()
        else:
            ret1, img1 = self._video1.read()
            # Read the corresponding frame from the second video
            ret2, img2 = self._video2.read()
        return ret1, img1, ret2, img2

    def _on_remapped_frame(self, frame_id_and_remapped_image):
        pass

    def _feed_next_frame(self, frame_id: int) -> bool:
        for _ in range(self._frame_stride_count):
            ret1, img1, ret2, img2 = self._read_next_frame_from_video()
            if not ret1 or not ret2:
                return False

        # INFO(f"{self._rp_str()} Adding frame {frame_id} to stitch data loader")
        # For some reason, hold onto a ref to the images while we push
        # them down into the data loader, or else there will be a segfault eventually
        assert self._max_input_queue_size >= 2
        while self._in_queue >= self._max_input_queue_size:
            # print("Too many images in the outgoing queue")
            time.sleep(0.01)
        # print(f"Adding frame {frame_id} images to stitcher")
        assert img1 is not None
        assert img2 is not None
        self._in_queue += 1
        # print(f"rank {self._rank} feeding LR frame {frame_id}")

        if self._use_pytorch_remap:
            self._remapper_1.send(
                frame_id=frame_id,
                source_tensor=np.expand_dims(img1.transpose(2, 0, 1), axis=0),
            )
            self._remapper_2.send(
                frame_id=frame_id,
                source_tensor=np.expand_dims(img2.transpose(2, 0, 1), axis=0),
            )
        else:
            core.add_to_stitching_data_loader(
                self._stitcher,
                frame_id,
                self._prepare_image(img1),
                self._prepare_image(img2),
            )
        return True

    def _image_getter_worker(self):
        pull_timer = Timer()
        count = 0
        for frame_id in range(
            self._start_frame_number, self._end_frame, self._frame_stride_count
        ):
            if self._is_ready_to_exit():
                break
            pull_timer.tic()
            self._waiting_for_frame_id = frame_id
            stitched_frame = core.get_stitched_frame_from_data_loader(
                self._stitcher, frame_id
            )
            if stitched_frame is None:
                break
            assert self._in_queue > 0
            self._in_queue -= 1
            pull_timer.toc()

            if count % 20 == 0:
                logger.info(
                    "Pulling stitch frame from stitcher: {} ({:.2f} fps)".format(
                        frame_id,
                        1.0 / max(1e-5, pull_timer.average_time),
                    )
                )
                pull_timer = Timer()
            while self._image_response_queue.qsize() > 25:
                time.sleep(0.1)
            self._image_response_queue.put((frame_id, stitched_frame))
            count += 1
        self._image_response_queue.put(StopIteration())

    def _frame_feeder_worker(
        self,
        max_frames: int,
    ):
        for frame_id in range(
            self._start_frame_number, self._end_frame, self._frame_stride_count
        ):
            if self._is_ready_to_exit():
                break
            if not self._feed_next_frame(frame_id=frame_id):
                core.close_stitching_data_loader(self._stitcher, frame_id)
                break
            else:
                pass
        INFO(f"{self._rp_str()} Feeder thread exiting")

    def _start_feeder_thread(self):
        self._feeder_thread = threading.Thread(
            target=self._frame_feeder_worker,
            args=(self._max_frames,),
        )
        self._feeder_thread.start()

        self._image_getter_thread = threading.Thread(
            name="_image_getter_worker",
            target=self._image_getter_worker,
            args=(),
        )
        self._image_getter_thread.start()

    def _stop_child_threads(self):
        if self._feeder_thread is not None:
            safe_put_queue(self._to_worker_queue, None)
        if self._image_getter_thread is not None:
            safe_put_queue(self._to_worker_queue, None)
        if self._feeder_thread is not None:
            self._feeder_thread.join()
            self._feeder_thread = None
        if self._image_getter_thread is not None:
            if self._waiting_for_frame_id is not None:
                core.close_stitching_data_loader(
                    self._stitcher, self._waiting_for_frame_id
                )
            self._image_getter_thread.join()
            self._image_getter_thread = None

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
