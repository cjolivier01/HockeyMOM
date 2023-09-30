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

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import tifffile

from lib.ffmpeg import copy_audio
from lib.ui.mousing import draw_box_with_mouse
from lib.tracking_utils.log import logger
from lib.datasets.dataset.stitching import StitchDataset, build_stitching_project, find_roi

from hockeymom import core


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

        self.duration = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.duration = 0.0



def stitch_videos():
    vid_dir = os.path.join(os.environ["HOME"], "Videos")

    # PTO Project File
    pto_project_file = f"{vid_dir}/my_project.pto"

    build_stitching_project(pto_project_file)

    # start_frame_number = 2000
    start_frame_number = 0

    max_frames = 100

    output_stitched_video_file = "./stitched_output.avi"

    data_loader = StitchDataset(
        video_file_1=f"{vid_dir}/left.mp4",
        video_file_2=f"{vid_dir}/right.mp4",
        pto_project_file=pto_project_file,
        video_1_offset_frame=217,
        video_2_offset_frame=0,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
    )

    frame_count = 0
    start = None
    for i, stitched_image in enumerate(data_loader):
        print(f"Read frame {i}")
        if i == 0:
            find_roi(stitched_image)
        frame_count += 1
        if i == 1:
            start = time.time()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )


def main():
    stitch_videos()


if __name__ == "__main__":
    main()
    print("Done.")
