"""
Experiments in stitching
"""
import os
import time
import argparse
import numpy as np
from typing import Tuple, List
import cv2

import torch
import torch.nn.functional as F

from hmlib.stitch_synchronize import get_image_geo_position

import hockeymom.core as core
from hmlib.tracking_utils.timer import Timer

from hmlib.stitching.remapper import (
    ImageRemapper,
    read_frame_batch,
    pad_tensor_to_size,
    pad_tensor_to_size_batched,
)

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("Image Remapper")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images",
    )
    parser.add_argument(
        "--project-file",
        "--project_file",
        default="autooptimiser_out.pto",
        type=str,
        help="Use project file as input to stitcher",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=str,
        help="Video directory to find 'left.mp4' and 'right.mp4'",
    )
    parser.add_argument(
        "--lfo",
        "--left_frame_offset",
        default=None,
        type=float,
        help="Left frame offset",
    )
    parser.add_argument(
        "--rfo",
        "--right_frame_offset",
        default=None,
        type=float,
        help="Right frame offset",
    )
    return parser


class BlendImageInfo:
    def __init__(self, width: int, height: int, xpos: int, ypos: int):
        self.width = width
        self.height = height
        self.xpos = xpos
        self.ypos = ypos


class ImageBlender:
    def __init__(self, images: List[BlendImageInfo], seam: torch.Tensor):
        pass


def make_blender_compatible_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        assert tensor.dim() == 3
        if tensor.size(0) == 3 or tensor.size(0) == 4:
            # Need to make channels-last
            tensor = tensor.permute(1, 2, 0)
        return tensor.contiguous().cpu().numpy()
    if tensor.shape[0] == 3 or tensor.shape[0] == 4:
        tensor = tensor.transpose(1, 2, 0)
    return np.ascontiguousarray(tensor)


def blend_video(
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    basename_1: str,
    basename_2: str,
    interpolation: str = None,
    lfo: float = 0,
    rfo: float = 0,
    show: bool = False,
):
    cap_1 = cv2.VideoCapture(os.path.join(dir_name, video_file_1))
    if not cap_1 or not cap_1.isOpened():
        raise AssertionError(
            f"Could not open video file: {os.path.join(dir_name, video_file_1)}"
        )
    else:
        if lfo:
            cap_1.set(cv2.CAP_PROP_POS_FRAMES, lfo)

    cap_2 = cv2.VideoCapture(os.path.join(dir_name, video_file_2))
    if not cap_2 or not cap_2.isOpened():
        raise AssertionError(
            f"Could not open video file: {os.path.join(dir_name, video_file_2)}"
        )
    else:
        if rfo:
            cap_2.set(cv2.CAP_PROP_POS_FRAMES, rfo)

    device = "cuda"
    batch_size = 2

    source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
    source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)

    remapper_1 = ImageRemapper(
        dir_name=dir_name,
        basename=basename_1,
        source_hw=source_tensor_1.shape[-2:],
        channels=source_tensor_1.shape[1],
        interpolation=interpolation,
    )
    remapper_1.init(batch_size=batch_size)
    remapper_1.to(device=device)

    remapper_2 = ImageRemapper(
        dir_name=dir_name,
        basename=basename_2,
        source_hw=source_tensor_2.shape[-2:],
        channels=source_tensor_2.shape[1],
        interpolation=interpolation,
    )
    remapper_2.init(batch_size=batch_size)
    remapper_2.to(device=device)

    seam_filename = os.path.join(dir_name, "seam_file.png")
    xor_filename = os.path.join(dir_name, "xor_file.png")
    blender = core.EnBlender(
        args=[
            f"--save-seams",
            seam_filename,
            f"--save-xor",
            xor_filename,
        ]
    )

    timer = Timer()
    frame_count = 0
    while True:
        destination_tensor_1 = remapper_1.forward(source_image=source_tensor_1)
        destination_tensor_1 = destination_tensor_1.contiguous().cpu()
        destination_tensor_2 = remapper_2.forward(source_image=source_tensor_2)
        destination_tensor_2 = destination_tensor_2.contiguous().cpu()

        if frame_count == 0:
            blended = blender.blend_images(
                left_image=make_blender_compatible_tensor(destination_tensor_1[0]),
                left_xy_pos=[remapper_1.xpos, remapper_1.ypos],
                right_image=make_blender_compatible_tensor(destination_tensor_2[0]),
                right_xy_pos=[remapper_2.xpos, remapper_2.ypos],
            )
            if show:
                cv2.imshow("blended", blended)
                cv2.waitKey(1)

        frame_count += 1
        if frame_count != 1:
            timer.toc()

        if frame_count % 20 == 0:
            print(
                "Remapping: {:.2f} fps".format(
                    batch_size * 1.0 / max(1e-5, timer.average_time)
                )
            )
            if frame_count % 50 == 0:
                timer = Timer()

        # if show:
        #     for i in range(len(destination_tensor_1)):
        #         cv2.imshow(
        #             "destination_tensor_1",
        #             destination_tensor_1[i].permute(1, 2, 0).numpy(),
        #         )
        #         #cv2.waitKey(1)
        #         cv2.imshow(
        #             "destination_tensor_2",
        #             destination_tensor_2[i].permute(1, 2, 0).numpy(),
        #         )
        #         cv2.waitKey(1)

        source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
        source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)
        timer.tic()


def main(args):
    with torch.no_grad():
        blend_video(
            "left.mp4",
            "right.mp4",
            args.video_dir,
            "mapping_0000",
            "mapping_0001",
            lfo=args.lfo,
            rfo=args.rfo,
            interpolation="",
            show=args.show,
        )


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
    print("Done.")
