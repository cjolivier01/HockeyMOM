"""
Experiments in stitching
"""
import os
import cv2
import threading
import time
import multiprocessing
import numpy as np
from typing import List

from pathlib import Path

from hockeymom import core

from lib.tracking_utils import visualization as vis
from lib.ffmpeg import extract_frame_image
from lib.stitch_synchronize import (
    configure_video_stitching,
    find_sitched_roi,
)


def _get_dir_name(path):
    if os.path.isdir(path):
        return path
    return Path(path).parent


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


def distribute_items_detailed(total_item_count, worker_count):
    base_items_per_worker = total_item_count // worker_count
    remainder = total_item_count % worker_count

    distribution = []
    for i in range(worker_count):
        if i < remainder:
            distribution.append(base_items_per_worker + 1)
        else:
            distribution.append(base_items_per_worker)

    return distribution


_VERBOSE = True


def INFO(*args, **kwargs):
    if not _VERBOSE:
        return
    print(*args, **kwargs)


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
        pto_project_file: str = None,
        video_1_offset_frame: int = None,
        video_2_offset_frame: int = None,
        start_frame_number: int = 0,
        max_input_queue_size: int = 50,
        remap_thread_count: int = 10,
        blend_thread_count: int = 10,
        max_frames: int = None,
        frame_stride_count: int = 1,
    ):
        assert max_input_queue_size > 0
        self._rank = rank
        self._start_frame_number = start_frame_number
        self._output_video = None
        self._video_1_offset_frame = video_1_offset_frame
        self._video_2_offset_frame = video_2_offset_frame
        self._video_file_1 = video_file_1
        self._video_file_2 = video_file_2
        self._pto_project_file = pto_project_file
        self._max_input_queue_size = min(max_input_queue_size, max_frames)
        self._remap_thread_count = remap_thread_count
        self._blend_thread_count = blend_thread_count
        self._max_frames = max_frames
        self._to_worker_queue = multiprocessing.Queue()
        self._from_worker_queue = multiprocessing.Queue()
        self._image_request_queue = multiprocessing.Queue()
        self._image_response_queue = multiprocessing.Queue()
        self._shutdown_barrier = None
        self._frame_stride_count = frame_stride_count
        self._open = False
        self._last_requested_frame = None
        self._feeder_thread = None
        self._image_roi = None
        self._forked = False
        self._closing = False

    def rp_str(self):
        """Worker rank prefix string."""
        return "WR[" + str(self._rank) + "] "

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
                print("Stitching worker shutting down")
            else:
                ack, worker_info = self._from_worker_queue.get()
                assert ack == "openned"
                self._last_requested_frame = worker_info.last_requested_frame
                self._total_num_frames = worker_info.total_num_frames
                self._open = True
        else:
            self._open_videos()

    def get(self):
        return self._from_worker_queue.get()

    def request_image(self, frame_id: int):
        # INFO(f"StitchingWorker.request_image {frame_id}")
        self._image_request_queue.put(frame_id)

    def receive_image(self, frame_id):
        # INFO(f"ASK StitchingWorker.receive_image {frame_id}")
        result = self._image_response_queue.get()
        if isinstance(result, Exception):
            raise result
        fid, image = result
        # INFO(f"GOT StitchingWorker.receive_image {fid}")
        assert fid == frame_id
        return image

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
        # TODO: adjust max frames based on this
        self._total_num_frames = min(
            int(self._video1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self._video2.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._stitcher = core.StitchingDataLoader(
            0,
            self._pto_project_file,
            self._max_input_queue_size,
            self._remap_thread_count,
            self._blend_thread_count,
        )
        self._start_feeder_thread()
        self._prime_frame_request_queue()
        self._open = True

    def close(self, in_process: bool = False):
        if self._forked and not in_process:
            self._to_worker_queue.put(None)
            self._image_request_queue.put(None)
        else:
            self._stop_child_threads()
            self._video1.release()
            self._video2.release()
            if self._output_video is not None:
                self._output_video.release()
            self._open = False
        if self._shutdown_barrier is not None:
            self._shutdown_barrier.wait()

    def _feed_next_frame(
        self,
    ) -> bool:
        frame_request = self._to_worker_queue.get()
        if frame_request is None:
            raise StopIteration()
        frame_id = frame_request.frame_id
        frame_id = int(frame_id)
        ret1, img1 = self._video1.read()
        if not ret1:
            return False
        # Read the corresponding frame from the second video
        ret2, img2 = self._video2.read()
        if not ret2:
            return False
        # INFO(f"Adding frame {frame_id} to stitch data loader")
        core.add_to_stitching_data_loader(self._stitcher, frame_id, img1, img2)
        return True

    def _get_next_frame(self, frame_id: int):
        # INFO(f"Asking for frame {frame_id} from stitch data loader")
        stitched_frame = core.get_stitched_frame_from_data_loader(
            self._stitcher, frame_id
        )
        # INFO(f"Got frame {frame_id} from stitch data loader")
        if stitched_frame is None:
            raise StopIteration()
        return stitched_frame

    def _image_getter_worker(self):
        frame_count = 0
        while frame_count < self._max_frames:
            frame_id = self._image_request_queue.get()
            if frame_id is None:
                break
            stitched_image = self._get_next_frame(frame_id=frame_id)
            self._image_response_queue.put((frame_id, stitched_image))
            frame_count += 1
        self._image_response_queue.put(StopIteration())

    def _frame_feeder_worker(
        self,
        max_frames: int,
    ):
        frame_count = 0
        while not max_frames or frame_count < max_frames:
            if not self._feed_next_frame():
                break
            else:
                self._from_worker_queue.put("ok")
            frame_count += 1
        INFO("Feeder thread exiting")
        self._from_worker_queue.put(StopIteration())

    def _prime_frame_request_queue(self):
        for i in range(self._max_input_queue_size):
            req_frame = self._start_frame_number + (i * self._frame_stride_count)
            if (
                not self._max_frames
                or req_frame < self._start_frame_number + self._max_frames
            ):
                self._to_worker_queue.put(
                    FrameRequest(frame_id=req_frame, want_alpha=(i == 0))
                )
                self._last_requested_frame = req_frame
        INFO(f"self._last_requested_frame={self._last_requested_frame}")

    def _start_feeder_thread(self):
        self._feeder_thread = threading.Thread(
            target=self._frame_feeder_worker,
            args=(self._max_frames,),
        )
        self._feeder_thread.start()

        self._image_getter_thread = threading.Thread(
            target=self._image_getter_worker,
            args=(),
        )
        self._image_getter_thread.start()

        # self._last_requested_frame = self._start_frame_number
        # self.request_next_frame()

    def _stop_child_threads(self):
        if self._feeder_thread is not None:
            self._to_worker_queue.put(None)
            self._feeder_thread.join()
            self._feeder_thread = None
        if self._image_getter_thread is not None:
            self._image_request_queue.put(None)
            self._image_getter_thread.join()
            self._image_getter_thread = None

    def request_next_frame(self):
        req_frame = self._last_requested_frame + self._frame_stride_count
        if (
            not self._max_frames
            or req_frame < self._start_frame_number + self._max_frames
        ):
            self._last_requested_frame += self._frame_stride_count
            # INFO(f"request_next_frame(): self._to_worker_queue( {self._last_requested_frame} )")
            self._to_worker_queue.put(
                FrameRequest(frame_id=self._last_requested_frame, want_alpha=False)
            )
        else:
            # INFO("request next frame {req_frame} would be too many")
            pass


##
#   _____ _   _  _        _     _____        _                     _
#  / ____| | (_)| |      | |   |  __ \      | |                   | |
# | (___ | |_ _ | |_  ___| |__ | |  | | __ _| |_  __ _  ___   ___ | |_
#  \___ \| __| || __|/ __| '_ \| |  | |/ _` | __|/ _` |/ __| / _ \| __|
#  ____) | |_| || |_| (__| | | | |__| | (_| | |_| (_| |\__ \|  __/| |_
# |_____/ \__|_| \__|\___|_| |_|_____/ \__,_|\__|\__,_||___/ \___| \__|
#
#
class StitchDataset:
    def __init__(
        self,
        video_file_1: str,
        video_file_2: str,
        pto_project_file: str = None,
        video_1_offset_frame: int = None,
        video_2_offset_frame: int = None,
        output_stitched_video_file: str = None,
        start_frame_number: int = 0,
        max_input_queue_size: int = 50,
        remap_thread_count: int = 10,
        blend_thread_count: int = 10,
        max_frames: int = None,
        auto_configure: bool = True,
        num_workers: int = 1,
    ):
        assert max_input_queue_size > 0
        self._start_frame_number = start_frame_number
        self._output_stitched_video_file = output_stitched_video_file
        self._output_video = None
        self._video_1_offset_frame = video_1_offset_frame
        self._video_2_offset_frame = video_2_offset_frame
        self._video_file_1 = video_file_1
        self._video_file_2 = video_file_2
        self._pto_project_file = pto_project_file
        self._max_input_queue_size = max_input_queue_size
        self._remap_thread_count = remap_thread_count
        self._blend_thread_count = blend_thread_count
        self._max_frames = max_frames
        self._to_coordinator_queue = multiprocessing.Queue()
        self._from_coordinator_queue = multiprocessing.Queue()
        self._from_coordinator_queue = multiprocessing.Queue()
        self._current_frame = start_frame_number
        self._next_requested_frame = start_frame_number
        self._image_roi = None
        self._fps = None
        self._auto_configure = auto_configure
        self._num_workers = num_workers
        self._stitching_workers = {}
        # Temporary until we get the middle-man
        self._current_worker = 0
        self._current_get_next_frame_worker = 0
        self._ordering_queue = core.SortedRGBImageQueue()
        self._coordinator_thread = None

    def __delete__(self):
        for worker in self._stitching_workers.values():
            worker.close()

    def stitching_worker(self, worker_number: int):
        return self._stitching_workers[worker_number]

    def create_stitching_worker(
        self,
        rank: int,
        start_frame_number: int,
        frame_stride_count: int,
        max_frames: int,
    ):
        stitching_worker = StitchingWorker(
            rank=rank,
            video_file_1=self._video_file_1,
            video_file_2=self._video_file_2,
            pto_project_file=self._pto_project_file,
            video_1_offset_frame=self._video_1_offset_frame,
            video_2_offset_frame=self._video_2_offset_frame,
            start_frame_number=start_frame_number,
            max_input_queue_size=self._max_input_queue_size,
            remap_thread_count=self._remap_thread_count,
            blend_thread_count=self._blend_thread_count,
            max_frames=max_frames,
            frame_stride_count=frame_stride_count,
        )
        return stitching_worker

    def configure_stitching(self):
        if not self._pto_project_file:
            dir_name = _get_dir_name(self._video_file_1)
            self._pto_project_file, lfo, rfo = configure_video_stitching(
                dir_name, self._video_file_1, self._video_file_2
            )
            self._video_1_offset_frame = lfo
            self._video_2_offset_frame = rfo

    def initialize(self):
        if self._auto_configure:
            self.configure_stitching()

    @property
    def fps(self):
        if self._fps is None:
            video1 = cv2.VideoCapture(self._video_file_1)
            if not video1.isOpened():
                raise AssertionError(f"Could not open video file: {self._video_file_1}")
            self._fps = video1.get(cv2.CAP_PROP_FPS)
            video1.release()
        return self._fps

    def close(self):
        for stitching_worker in self._stitching_workers.values():
            stitching_worker.close()
        self._stop_coordinator_thread()
        self._stitching_workers.clear()

    def _maybe_write_output(self, output_img):
        if self._output_stitched_video_file:
            if self._output_video is None:
                fps = self.fps
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                final_video_size = (output_img.shape[1], output_img.shape[0])
                self._output_video = cv2.VideoWriter(
                    filename=self._output_stitched_video_file,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=final_video_size,
                    isColor=True,
                )
                assert self._output_video.isOpened()
                self._output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)

            self._output_video.write(output_img)

    def _prepare_next_frame(self, frame_id: int):
        # INFO(f"_prepare_next_frame( {frame_id} )")
        stitching_worker = self._stitching_workers[self._current_worker]
        stitching_worker.request_image(frame_id=frame_id)
        stitched_frame = stitching_worker.receive_image(frame_id=frame_id)
        self._current_worker = (self._current_worker + 1) % len(self._stitching_workers)
        if self._image_roi is None:
            self._image_roi = find_sitched_roi(stitched_frame)

        copy_data = True
        # INFO(f"Localling enqueing frame {frame_id}")
        stitched_frame = stitched_frame.copy()
        self._ordering_queue.enqueue(frame_id, stitched_frame, copy_data)

    def _start_coordinator_thread(self):
        assert self._coordinator_thread is None
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_thread_worker,
            args=(),
        )
        self._coordinator_thread.start()
        for i in range(min(self._max_input_queue_size, self._max_frames)):
            # INFO(f"putting _to_coordinator_queue.put({self._next_requested_frame})")
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._next_requested_frame += 1

    def _stop_coordinator_thread(self):
        if self._coordinator_thread is not None:
            self._to_coordinator_queue.put("stop")
            self._coordinator_thread.join()
            self._coordinator_thread = None

    def _coordinator_thread_worker(self):
        try:
            frame_count = 0
            while frame_count < self._max_frames:
                command = self._to_coordinator_queue.get()
                if isinstance(command, str) and command == "stop":
                    break
                frame_id = int(command)
                self._prepare_next_frame(frame_id)
                self._from_coordinator_queue.put(("ok", frame_id))
                frame_count += 1
            self._from_coordinator_queue.put(StopIteration())
        except Exception as ex:
            self._from_coordinator_queue.put(ex)

    @staticmethod
    def prepare_frame_for_video(
        image: np.array, image_roi: np.array, show_image: bool = False
    ):
        if not image_roi:
            if image.shape[2] == 4:
                image = image[:, :, :3]
        else:
            image = image[image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3]
        if show_image:
            cv2.imshow("online_im", image)
            cv2.waitKey(1)
        return image

    def __iter__(self):
        if not self._stitching_workers:
            self.initialize()
            # Openend close to validate existance as well as get some stats, such as fps
            for worker_number in range(self._num_workers):
                max_for_worker = self._max_frames
                if max_for_worker is not None:
                    max_for_worker = distribute_items_detailed(
                        self._max_frames, self._num_workers
                    )[worker_number]  # TODO: call just once
                self._stitching_workers[worker_number] = self.create_stitching_worker(
                    rank=worker_number,
                    start_frame_number=self._start_frame_number + worker_number,
                    frame_stride_count=self._num_workers,
                    max_frames=max_for_worker,
                )
                self._stitching_workers[worker_number].start(fork=False)
            self._start_coordinator_thread()
        return self

    def get_next_frame(self):
        stitching_worker = self._stitching_workers[self._current_get_next_frame_worker]
        status = stitching_worker.get()
        if isinstance(status, Exception):
            raise status

        assert status == "ok"
        # INFO(f"Trying to locally dequeue frame id: {self._current_frame}")
        stitched_frame = self._ordering_queue.dequeue_key(self._current_frame)
        # INFO(f"Locally dequeued frame id: {self._current_frame}")
        self._current_get_next_frame_worker = (
            self._current_get_next_frame_worker + 1
        ) % self._num_workers

        if (
            not self._max_frames
            or self._next_requested_frame < self._start_frame_number + self._max_frames
        ):
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._next_requested_frame += 1
            stitching_worker.request_next_frame()
        else:
            # INFO(
            #     f"Next frame {self._next_requested_frame} would be above the max allowed frame, so not queueing"
            # )
            pass
        return stitched_frame

    def __next__(self):
        # INFO(f"BEGIN next() self._from_coordinator_queue.get() {self._current_frame}")
        status = self._from_coordinator_queue.get()
        # INFO(f"END next() self._from_coordinator_queue.get( {self._current_frame})")
        if isinstance(status, Exception):
            self.close()
            raise status
        else:
            status, frame_id = status
            assert status == "ok"
            assert frame_id == self._current_frame
        stitched_frame = self.get_next_frame()

        # Code doesn't handle strides channbels efficiently
        stitched_frame = self.prepare_frame_for_video(
            stitched_frame,
            image_roi=self._image_roi,
        )

        self._current_frame += 1
        self._maybe_write_output(stitched_frame)
        return stitched_frame

    def __len__(self):
        return self._total_num_frames
