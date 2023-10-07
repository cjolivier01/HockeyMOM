"""
Experiments in stitching
"""
import os
import cv2
import threading
import multiprocessing

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
        frame_step_count: int = 2,
    ):
        assert max_input_queue_size > 0
        assert num_workers > 0
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
        # self._to_worker_queue = multiprocessing.Queue()
        # self._from_worker_queue = multiprocessing.Queue()
        self._open = False
        self._current_frame = start_frame_number
        self._last_requested_frame = None
        self._feeder_thread = None
        self._image_roi = None
        self._fps = None
        self._auto_configure = auto_configure
        self._frame_step_count = frame_step_count
        self._num_workers = num_workers
        self._worker_number = 0
        if self._num_workers > 1:
            self._ordering_queue = core.SortedRGBImageQueue()
        else:
            self._ordering_queue = None
        self._from_worker_queues = [
            multiprocessing.Queue() for _ in range(self._num_workers)
        ]
        self._to_worker_queues = [
            multiprocessing.Queue() for _ in range(self._num_workers)
        ]

    def configure_stitching(self):
        if not self._pto_project_file:
            dir_name = _get_dir_name(self._video_file_1)
            self._pto_project_file, lfo, rfo = configure_video_stitching(
                dir_name, self._video_file_1, self._video_file_2
            )
            self._video_1_offset_frame = lfo
            self._video_2_offset_frame = rfo

    def _fork_worker(self, worker_number: int):
        if os.fork():
            return
        assert not self._open
        self._worker_number = worker_number
        self._start_frame_number += worker_number
        self._frame_step_count *= self._num_workers
        self._open()
        self.frame_feeder_worker(
            max_frames=self._max_frames / self._num_workers, worker_number=worker_number
        )

    def _open_videos(self):
        if self._auto_configure:
            self.configure_stitching()

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
        self._total_num_frames = min(
            int(self._video1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self._video2.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._fps = self._video1.get(cv2.CAP_PROP_FPS)
        self._open = True

    def _create_dataloader(self):
        self._stitcher = core.StitchingDataLoader(
            0,
            self._pto_project_file,
            self._max_input_queue_size,
            self._remap_thread_count,
            self._blend_thread_count,
        )

    @property
    def fps(self):
        return self._fps

    def close(self):
        self._video1.release()
        self._video2.release()
        if self._output_video is not None:
            self._output_video.release()
        self._open = False

    def _maybe_write_output(self, output_img):
        if self._output_stitched_video_file:
            if self._output_video is None:
                fps = self._video1.get(cv2.CAP_PROP_FPS)
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

    def _feed_next_frame(
        self,
        worker_number: int,
    ) -> bool:
        frame_request = self._to_worker_queues[worker_number].get()
        if frame_request is None:
            raise StopIteration
        frame_id = frame_request.frame_id
        frame_id = int(frame_id)
        ret1, img1 = self._video1.read()
        if not ret1:
            return False
        # Read the corresponding frame from the second video
        ret2, img2 = self._video2.read()
        if not ret2:
            return False

        core.add_to_stitching_data_loader(self._stitcher, frame_id, img1, img2)

        # Maybe skip over some frames
        if self._frame_step_count > 1:
            # it is much faster to simply read the frames and discard them
            # than to try to set the frame explicitly
            for _ in range(self._frame_step_count - 1):
                self._video1.read()
                self._video2.read()

        return True

    def get_next_frame(self, frame_id: int):
        stitched_frame = core.get_stitched_frame_from_data_loader(
            self._stitcher, frame_id
        )
        if stitched_frame is None:
            raise StopIteration
        if self._image_roi is None:
            self._image_roi = find_sitched_roi(stitched_frame)

        stitched_frame = self.prepare_frame_for_video(
            stitched_frame,
            image_roi=self._image_roi,
        )
        return stitched_frame

    def frame_feeder_worker(
        self,
        max_frames: int,
        worker_number: int,
    ):
        frame_count = 0
        while not max_frames or frame_count < max_frames:
            if not self._feed_next_frame(worker_number=worker_number):
                break
            else:
                self._from_worker_queues[worker_number].put("ok")
            frame_count += 1
        print("Feeder thread exiting")
        self._from_worker_queues[worker_number].put(StopIteration())

    def _start_feeder_thread(self):
        self._feeder_thread = threading.Thread(
            target=self.frame_feeder_worker,
            args=(self._max_frames, self._worker_number),
        )
        self._feeder_thread.start()
        for i in range(self._max_input_queue_size):
            req_frame = self._current_frame + i
            if (
                not self._max_frames
                or req_frame < self._start_frame_number + self._max_frames
            ):
                self._to_worker_queues[self._worker_number].put(
                    FrameRequest(frame_id=req_frame, want_alpha=(i == 0))
                )
                self._last_requested_frame = req_frame

    def _stop_feeder_thread(self):
        if self._feeder_thread is not None:
            self._to_worker_queues[self._worker_number].put(None)
            self._feeder_thread.join()
            self._feeder_thread = None

    def prepare_frame_for_video(self, image, image_roi):
        if not image_roi:
            if image.shape[2] == 4:
                image = image[:, :, :3]
        else:
            image = image[image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3]
        # cv2.imshow("online_im", image)
        # cv2.waitKey(1)
        return image

    def __iter__(self):
        if not self._open:
            self._open_videos()
            self._create_dataloader()
            self._start_feeder_thread()
        return self

    def __next__(self):
        status = self._from_worker_queues[0].get()
        if isinstance(status, Exception):
            raise status
        else:
            assert status == "ok"
        stitched_frame = self.get_next_frame(self._current_frame)
        self._current_frame += 1
        self._last_requested_frame += 1
        self._to_worker_queues[0].put(
            FrameRequest(frame_id=self._last_requested_frame, want_alpha=False)
        )
        self._maybe_write_output(stitched_frame)
        return stitched_frame

    def __len__(self):
        return self._total_num_frames
