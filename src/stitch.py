"""
Experiments in stitching
"""
import os
import time
import argparse
import yaml
import numpy as np
from typing import Tuple
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from pathlib import Path
from hmlib.opts import opts
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.ui.mousing import draw_box_with_mouse
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.config import get_clip_box, get_game_config
from hmlib.stitching.remapper import ImageRemapper

from hmlib.stitch_synchronize import (
    configure_video_stitching,
)

from hmlib.datasets.dataset.stitching import (
    StitchDataset,
)

from hockeymom import core

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--lfo",
        "--left_frame_offset",
        default=None,
        type=float,
        help="Left frame offset",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of stitching workers"
    )
    parser.add_argument(
        "--project-file",
        "--project_file",
        default="autooptimiser_out.pto",
        type=str,
        help="Use project file as input to stitcher",
    )
    parser.add_argument(
        "--rfo",
        "--right_frame_offset",
        default=None,
        type=float,
        help="Right frame offset",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=str,
        help="Video directory to find 'left.mp4' and 'right.mp4'",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        type=str,
        help="Game ID",
    )
    return parser


def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    lfo: int = None,
    rfo: int = None,
    game_id: str = None,
    project_file_name: str = "my_project.pto",
    start_frame_number: int = 0,
    max_frames: int = None,
    output_stitched_video_file: str = os.path.join(".", "stitched_output.avi"),
):
    left_vid = BasicVideoInfo(os.path.join(dir_name, video_left))
    right_vid = BasicVideoInfo(os.path.join(dir_name, video_right))
    total_frames = min(left_vid.frame_count, right_vid.frame_count)
    print(f"Total possible stitched video frames: {total_frames}")

    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left,
        video_right,
        project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
    )

    # nona = core.HmNona(pto_project_file)
    # control_points = nona.get_control_points()
    # print(control_points)

    data_loader = StitchDataset(
        video_file_1=os.path.join(dir_name, video_left),
        video_file_2=os.path.join(dir_name, video_right),
        pto_project_file=pto_project_file,
        video_1_offset_frame=lfo,
        video_2_offset_frame=rfo,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
        num_workers=1,
        # remap_thread_count=10,
        # blend_thread_count=10,
        remap_thread_count=1,
        blend_thread_count=1,
        fork_workers=False,
        image_roi=get_clip_box(game_id=game_id, root_dir=ROOT_DIR),
    )

    frame_count = 0
    start = None

    dataset_timer = Timer()
    for i, _ in enumerate(data_loader):
        if i > 1:
            dataset_timer.toc()
        if i % 20 == 0:
            logger.info(
                "Dataset frame {} ({:.2f} fps)".format(
                    i, 1.0 / max(1e-5, dataset_timer.average_time)
                )
            )
            dataset_timer = Timer()

        frame_count += 1
        if i == 1:
            # draw_box_with_mouse(stitched_image, destroy_all_windows_after=True)
            start = time.time()
        dataset_timer.tic()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )
    return lfo, rfo


def main(args):
    video_left = "left.mp4"
    video_right = "right.mp4"

    args.lfo = 15
    args.rfo = 0
    lfo, rfo = stitch_videos(
        args.video_dir,
        video_left,
        video_right,
        lfo=args.lfo,
        rfo=args.rfo,
        project_file_name=args.project_file,
        game_id=args.game_id,
    )


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
    print("Done.")
