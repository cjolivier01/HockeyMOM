"""
Experiments in stitching
"""
import os
import cv2
import threading
import argparse
import torch
import traceback
import numpy as np
from typing import List

from pathlib import Path

from hockeymom import core

from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.stitch_synchronize import (
    configure_video_stitching,
)

from hmlib.ffmpeg import BasicVideoInfo
from hmlib.video_out import VideoOutput, ImageProcData


def _get_dir_name(path):
    if os.path.isdir(path):
        return path
    return Path(path).parent


from hmlib.stitching.worker import (
    StitchingWorker,
    create_queue,
    safe_put_queue,
    INFO,
    _LARGE_NUMBER_OF_FRAMES,
)


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
        fork_workers: bool = False,
        image_roi: List[int] = None,
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
        self._max_frames = (
            max_frames if max_frames is not None else _LARGE_NUMBER_OF_FRAMES
        )
        self._to_coordinator_queue = create_queue(mp=False)
        self._from_coordinator_queue = create_queue(mp=False)
        self._current_frame = start_frame_number
        self._next_requested_frame = start_frame_number
        self._image_roi = image_roi
        self._fps = None
        self._bitrate = None
        self._auto_configure = auto_configure
        self._num_workers = num_workers
        self._stitching_workers = {}
        self._fork_workers = fork_workers
        # Temporary until we get the middle-man (StitchingWorkersIterator)
        self._current_worker = 0
        self._ordering_queue = core.SortedPyArrayUin8Queue()
        self._coordinator_thread = None

        self._next_frame_timer = Timer()
        self._next_frame_counter = 0

        self._next_timer = Timer()

        self._prepare_next_frame_timer = Timer()

        self._video_1_info = BasicVideoInfo(video_file_1)
        self._video_2_info = BasicVideoInfo(video_file_2)
        v1o = 0 if self._video_1_offset_frame is None else self._video_1_offset_frame
        v2o = 0 if self._video_2_offset_frame is None else self._video_2_offset_frame
        self._total_number_of_frames = int(
            min(
                self._video_1_info.frame_count - v1o,
                self._video_2_info.frame_count - v2o,
            )
        )

        self._video_output = None

    def __delete__(self):
        for worker in self._stitching_workers.values():
            worker.close()

    @property
    def lfo(self):
        assert self._video_1_offset_frame is not None
        return self._video_1_offset_frame

    @property
    def rfo(self):
        assert self._video_2_offset_frame is not None
        return self._video_2_offset_frame

    def stitching_worker(self, worker_number: int):
        return self._stitching_workers[worker_number]

    def create_stitching_worker(
        self,
        rank: int,
        start_frame_number: int,
        frame_stride_count: int,
        max_frames: int,
        max_input_queue_size: int,
    ):
        stitching_worker = StitchingWorker(
            rank=rank,
            video_file_1=self._video_file_1,
            video_file_2=self._video_file_2,
            pto_project_file=self._pto_project_file,
            video_1_offset_frame=self._video_1_offset_frame,
            video_2_offset_frame=self._video_2_offset_frame,
            start_frame_number=start_frame_number,
            max_input_queue_size=max_input_queue_size,
            remap_thread_count=self._remap_thread_count,
            blend_thread_count=self._blend_thread_count,
            max_frames=max_frames,
            frame_stride_count=frame_stride_count,
            multiprocessingt_queue=self._fork_workers,
        )
        return stitching_worker

    def configure_stitching(self):
        if not self._pto_project_file:
            dir_name = _get_dir_name(self._video_file_1)
            self._pto_project_file, lfo, rfo = configure_video_stitching(
                dir_name,
                self._video_file_1,
                self._video_file_2,
                left_frame_offset=self._video_1_offset_frame,
                right_frame_offset=self._video_2_offset_frame,
            )
            self._video_1_offset_frame = lfo
            self._video_2_offset_frame = rfo

    def initialize(self):
        if self._auto_configure:
            self.configure_stitching()

    def _load_video_props(self):
        video1 = cv2.VideoCapture(self._video_file_1)
        if not video1.isOpened():
            raise AssertionError(f"Could not open video file: {self._video_file_1}")
        self._fps = video1.get(cv2.CAP_PROP_FPS)
        self._bitrate = video1.get(cv2.CAP_PROP_BITRATE)
        video1.release()

    @property
    def fps(self):
        if self._fps is None:
            self._load_video_props()
        return self._fps

    @property
    def bitrate(self):
        if self._bitrate is None:
            self._load_video_props()
        return self._bitrate

    def close(self):
        for stitching_worker in self._stitching_workers.values():
            stitching_worker.close()
        self._stop_coordinator_thread()
        self._stitching_workers.clear()
        if self._video_output is not None:
            self._video_output.stop()
            self._video_output = None

    def _maybe_write_output(self, output_img):
        if self._output_stitched_video_file:
            if self._output_video is None:
                fps = self.fps
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                # Write lossless Huffyuv codec
                # fourcc = cv2.VideoWriter_fourcc(*"HFYU")
                final_video_size = (output_img.shape[1], output_img.shape[0])
                self._output_video = cv2.VideoWriter(
                    filename=self._output_stitched_video_file,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=final_video_size,
                    isColor=True,
                )
                assert self._output_video.isOpened()
                print(f"OUPUT VIDEO BITRATE: {self.bitrate}")
                self._output_video.set(cv2.CAP_PROP_BITRATE, self.bitrate)

            self._output_video.write(output_img)

    def _prepare_next_frame(self, frame_id: int):
        # INFO(f"_prepare_next_frame( {frame_id} )")
        self._prepare_next_frame_timer.tic()
        stitching_worker = self._stitching_workers[self._current_worker]
        stitched_frame = stitching_worker.receive_image(expected_frame_id=frame_id)
        self._current_worker = (self._current_worker + 1) % len(self._stitching_workers)
        # INFO(f"Locally enqueing frame {frame_id}")
        self._ordering_queue.enqueue(frame_id, stitched_frame)
        self._prepare_next_frame_timer.toc()

    def _start_coordinator_thread(self):
        assert self._coordinator_thread is None
        for i in range(min(self._max_input_queue_size, self._max_frames)):
            # INFO(f"putting _to_coordinator_queue.put({self._next_requested_frame})")
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._from_coordinator_queue.put(("ok", self._next_requested_frame))
            self._next_requested_frame += 1
        self._coordinator_thread = threading.Thread(
            name="StitchCoordinator",
            target=self._coordinator_thread_worker,
            args=(self._next_requested_frame,),
        )
        self._coordinator_thread.start()
        print("Cached request queue")

    def _stop_coordinator_thread(self):
        if self._coordinator_thread is not None:
            self._to_coordinator_queue.put("stop")
            self._coordinator_thread.join()
            self._coordinator_thread = None

    def _send_frame_to_video_out(self, frame_id: int, stitched_frame: torch.Tensor):
        if not self._output_stitched_video_file:
            return
        if self._video_output is None:
            args = argparse.Namespace()
            args.fixed_edge_rotation = False
            args.crop_output_image = False
            args.use_watermark = False
            args.show_image = False
            args.plot_frame_number = False
            self._video_output_size = torch.tensor(
                (stitched_frame.shape[1], stitched_frame.shape[0]), dtype=torch.int32
            )
            self._video_output_box = torch.tensor(
                (0, 0, self._video_output_size[0] - 1, self._video_output_size[1] - 1),
                dtype=torch.float32,
            )
            self._video_output = VideoOutput(
                args=args,
                output_video_path=self._output_stitched_video_file,
                output_frame_width=self._video_output_size[0],
                output_frame_height=self._video_output_size[1],
                fps=self.fps,
            )
        image_proc_data = ImageProcData(
            frame_id=frame_id,
            img=stitched_frame,
            current_box=self._video_output_box.clone(),
        )

        self._video_output.append(image_proc_data)

    def _coordinator_thread_worker(self, next_requested_frame, *args, **kwargs):
        try:
            frame_count = 0
            while frame_count < self._max_frames:
                command = self._to_coordinator_queue.get()
                if isinstance(command, str) and command == "stop":
                    break
                frame_id = int(command)
                self._prepare_next_frame(frame_id)
                self._from_coordinator_queue.put(("ok", next_requested_frame))
                frame_count += 1
                next_requested_frame += 1
            self._from_coordinator_queue.put(StopIteration())
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                print(ex)
                traceback.print_exc()
            safe_put_queue(self._from_coordinator_queue, ex)

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
                    )[
                        worker_number
                    ]  # TODO: call just once
                self._stitching_workers[worker_number] = self.create_stitching_worker(
                    rank=worker_number,
                    start_frame_number=self._start_frame_number,
                    frame_stride_count=self._num_workers,
                    max_frames=max_for_worker,
                    max_input_queue_size=int(
                        self._max_input_queue_size / self._num_workers + 1
                    ),
                )
                self._stitching_workers[worker_number].start(fork=self._fork_workers)
            self._start_coordinator_thread()
        return self

    def get_next_frame(self):
        self._next_frame_timer.tic()

        stitched_frame = self._ordering_queue.dequeue_key(self._current_frame)

        # INFO(f"Locally dequeued frame id: {self._current_frame}")
        if (
            not self._max_frames
            or self._next_requested_frame < self._start_frame_number + self._max_frames
        ):
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._next_requested_frame += 1
        else:
            INFO(
                f"Next frame {self._next_requested_frame} would be above the max allowed frame, so not queueing"
            )
            pass

        self._next_frame_timer.toc()
        self._next_frame_counter += 1
        if self._next_frame_counter % 20 == 0:
            logger.info(
                "Stitch on-demand next-frame delivery speed: {:.2f} fps".format(
                    1.0 / max(1e-5, self._next_frame_timer.average_time)
                )
            )
            self._next_frame_timer = Timer()

        return stitched_frame

    def __next__(self):
        # INFO(f"\nBEGIN next() self._from_coordinator_queue.get() {self._current_frame}")
        # print(f"self._from_coordinator_queue size: {self._from_coordinator_queue.qsize()}")
        status = self._from_coordinator_queue.get()
        self._next_timer.tic()
        # INFO(f"END next() self._from_coordinator_queue.get( {self._current_frame})\n")
        if isinstance(status, Exception):
            self.close()
            raise status
        else:
            status, frame_id = status
            assert status == "ok"
            # print(f"self._from_coordinator_queue.get() = {frame_id}, self._current_frame = {self._current_frame} ")
            assert frame_id == self._current_frame

        # self._next_timer.tic()
        stitched_frame = self.get_next_frame()

        # Code doesn't handle strides channbels efficiently
        stitched_frame = self.prepare_frame_for_video(
            stitched_frame,
            image_roi=self._image_roi,
        )

        self._send_frame_to_video_out(frame_id=frame_id, stitched_frame=stitched_frame)

        self._current_frame += 1
        self._maybe_write_output(stitched_frame)
        self._next_timer.toc()
        return stitched_frame

    def __len__(self):
        return self._total_number_of_frames
