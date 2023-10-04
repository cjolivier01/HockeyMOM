"""
Experiments in stitching
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import threading
import multiprocessing

from typing import List

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import tifffile

from hockeymom import core

from lib.tracking_utils import visualization as vis
from lib.ffmpeg import extract_frame_image
from lib.stitch_synchronize import synchronize_by_audio


def get_tiff_tag_value(tiff_tag):
    if len(tiff_tag.value) == 1:
        return tiff_tag.value
    assert len(tiff_tag.value) == 2
    numerator, denominator = tiff_tag.value
    return float(numerator) / denominator


def get_image_geo_position(tiff_image_file: str):
    xpos, ypos = 0, 0
    with tifffile.TiffFile(tiff_image_file) as tif:
        tags = tif.pages[0].tags
        # Access the TIFFTAG_XPOSITION
        x_position = get_tiff_tag_value(tags.get("XPosition"))
        y_position = get_tiff_tag_value(tags.get("YPosition"))
        x_resolution = get_tiff_tag_value(tags.get("XResolution"))
        y_resolution = get_tiff_tag_value(tags.get("YResolution"))
        xpos = int(x_position * x_resolution + 0.5)
        ypos = int(y_position * y_resolution + 0.5)
        print(f"x={xpos}, y={ypos}")
    return xpos, ypos


def extract_frames(
    dir_name: str,
    video_left: str,
    left_frame_number: int,
    video_right: str,
    right_frame_number: int = 10,
):
    file_name_without_extension, _ = os.path.splitext(video_left)
    left_output_image_file = os.path.join(
        dir_name, file_name_without_extension + ".png"
    )

    file_name_without_extension, _ = os.path.splitext(video_right)
    right_output_image_file = os.path.join(
        dir_name, file_name_without_extension + ".png"
    )

    extract_frame_image(
        os.path.join(dir_name, video_left),
        frame_number=left_frame_number,
        dest_image=left_output_image_file,
    )
    extract_frame_image(
        os.path.join(dir_name, video_right),
        frame_number=right_frame_number,
        dest_image=right_output_image_file,
    )

    return left_output_image_file, right_output_image_file


def build_stitching_project(
    project_file_path: str,
    image_files=List[str],
    skip_if_exists: bool = True,
    test_blend: bool = True,
    fov: int = 108,
):
    pto_path = Path(project_file_path)
    dir_name = pto_path.parent

    if skip_if_exists and os.path.exists(project_file_path):
        print(f"Project file already exists (skipping project creatio9n): {project_file_path}")
        return True

    assert len(image_files) == 2
    left_image_file = image_files[0]
    right_image_file = image_files[1]

    curr_dir = os.getcwd()
    try:
        os.chdir(dir_name)
        cmd = [
            "pto_gen",
            "-o",
            project_file_path,
            "-f",
            str(fov),
            left_image_file,
            right_image_file,
        ]
        cmd_str = " ".join(cmd)
        os.system(cmd_str)
        cmd = ["cpfind", "--linearmatch", project_file_path, "-o", project_file_path]
        os.system(" ".join(cmd))
        cmd = [
            "autooptimiser",
            "-a",
            "-m",
            "-l",
            "-s",
            "-o",
            project_file_path,
            project_file_path,
        ]
        os.system(" ".join(cmd))
        if test_blend:
            cmd = [
                "nona",
                "-m",
                "TIFF_m",
                "-o",
                "nona_" + project_file_path,
                project_file_path,
            ]
            os.system(" ".join(cmd))
            cmd = [
                "enblend",
                "-o",
                os.path.join(dir_name, "panorama.tif"),
                os.path.join(dir_name, "my_project*.tif"),
            ]
            os.system(" ".join(cmd))
    finally:
        os.chdir(curr_dir)
    return True


def configure_video_stitching(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    project_file_name: str = "my_project.pto",
    base_frame_offset: int = 800,
    audio_sync_seconds: int = 15,
):
    lfo, rfo = synchronize_by_audio(
        file0_path=os.path.join(dir_name, video_left),
        file1_path=os.path.join(dir_name, video_right),
        seconds=audio_sync_seconds,
    )

    left_image_file, right_image_file = extract_frames(
        dir_name,
        video_left,
        base_frame_offset + lfo,
        video_right,
        base_frame_offset + rfo,
    )

    # PTO Project File
    pto_project_file = os.path.join(dir_name, project_file_name)

    build_stitching_project(
        pto_project_file, image_files=[left_image_file, right_image_file]
    )

    return pto_project_file, lfo, rfo


def get_dir_name(path):
    if os.path.isdir(path):
        return path
    return Path(path).parent


def find_roi(image):
    w = image.shape[1]
    h = image.shape[0]

    minus_w = int(w / 18)
    minus_h = int(h / 15)
    roi = [
        minus_w,
        int(minus_h * 1.5),
        image.shape[1] - minus_w,
        image.shape[0] - minus_h,
    ]
    return roi


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
        self._to_worker_queue = multiprocessing.Queue()
        self._from_worker_queue = multiprocessing.Queue()
        self._open = False
        self._current_frame = start_frame_number
        self._last_requested_frame = None
        self._feeder_thread = None
        self._image_roi = None
        self._fps = None
        self._auto_configure = auto_configure

    def configure_stitching(self):
        if not self._pto_project_file:
            dir_name = get_dir_name(self._video_file_1)
            self._pto_project_file, lfo, rfo = configure_video_stitching(
                dir_name, self._video_file_1, self._video_file_2
            )
            self._video_1_offset_frame = lfo
            self._video_2_offset_frame = rfo

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
        self._stitcher = core.StitchingDataLoader(
            0,
            self._pto_project_file,
            self._max_input_queue_size,
            self._remap_thread_count,
            self._blend_thread_count,
        )
        self._fps = self._video1.get(cv2.CAP_PROP_FPS)
        self._start_feeder_thread()
        self._open = True

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
    ) -> bool:
        frame_request = self._to_worker_queue.get()
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
        return True

    def get_next_frame(self, frame_id: int):
        stitched_frame = core.get_stitched_frame_from_data_loader(
            self._stitcher, frame_id
        )
        if stitched_frame is None:
            raise StopIteration
        if self._image_roi is None:
            self._image_roi = find_roi(stitched_frame)

        stitched_frame = self.prepare_frame_for_video(
            stitched_frame,
            image_roi=self._image_roi,
        )
        return stitched_frame

    def frame_feeder_worker(
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
        print("Feeder thread exiting")
        self._from_worker_queue.put(StopIteration())

    def _start_feeder_thread(self):
        self._feeder_thread = threading.Thread(
            target=self.frame_feeder_worker,
            args=(self._max_frames,),
        )
        self._feeder_thread.start()
        for i in range(self._max_input_queue_size):
            req_frame = self._current_frame + i
            if (
                not self._max_frames
                or req_frame < self._start_frame_number + self._max_frames
            ):
                self._to_worker_queue.put(
                    FrameRequest(frame_id=req_frame, want_alpha=(i == 0))
                )
                self._last_requested_frame = req_frame

    def _stop_feeder_thread(self):
        if self._feeder_thread is not None:
            self._to_worker_queue.put(None)
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
        return self

    def __next__(self):
        status = self._from_worker_queue.get()
        if isinstance(status, Exception):
            raise status
        else:
            assert status == "ok"
        stitched_frame = self.get_next_frame(self._current_frame)
        self._current_frame += 1
        self._last_requested_frame += 1
        self._to_worker_queue.put(
            FrameRequest(frame_id=self._last_requested_frame, want_alpha=False)
        )
        self._maybe_write_output(stitched_frame)
        return stitched_frame

    def __len__(self):
        return self._total_num_frames
