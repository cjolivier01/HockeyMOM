from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import math
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
from lib.ffmpeg import copy_audio

logger.setLevel(logging.INFO)

ORIGINAL_IMAGE_SIZE = (1088, 608)
ORIGINAL_PROCESS_IMAGE_SIZE = (1920, 1080)


def next_power_of_2(n):
    if n <= 0:
        raise ValueError("Input should be a positive integer")

    # Check if n is already a power of two
    if (n & (n - 1)) == 0:
        return n

    # Find the next power of 2
    power = math.ceil(math.log2(n))
    return 2**power


def demo(opt):
    result_root = opt.output_root if opt.output_root != "" else "."
    mkdir_if_missing(result_root)

    logger.info("Starting tracking...")

    opt.img_size = (4096, 1024)

    # opt.img_size = (4096 * 3 // 2, 1024 * 3 // 2)

    # opt.img_size = (4096 // 2, 1024 // 2)

    # opt.img_size = (4096 // 2, 1024 // 2)

    # opt.img_size = (4096 // 4, 1024 // 4)

    # opt.img_size = (4096, 1800)
    # opt.img_size = (1088, 608)
    # opt.img_size = (1088*2, 608*2)
    # opt.img_size = ORIGINAL_IMAGE_SIZE

    # my_process_image_size = (4096//2, 1024//2)

    opt.process_img_size = opt.img_size
    # opt.process_img_size = my_process_image_size
    # opt.process_img_size = ORIGINAL_PROCESS_IMAGE_SIZE

    input_video_files = opt.input_video.split(",")

    if len(input_video_files) == 2:
        video_1_offset_frame = None
        video_2_offset_frame = None

        video_1_offset_frame = 13
        video_2_offset_frame = 0

        dataloader = datasets.LoadAutoStitchedVideoWithOrig(
            path_video_1=input_video_files[0],
            path_video_2=input_video_files[1],
            video_1_offset_frame=video_1_offset_frame,
            video_2_offset_frame=video_2_offset_frame,
            img_size=opt.img_size,
            process_img_size=opt.process_img_size,
        )
    else:
        assert len(input_video_files) == 1
        dataloader = datasets.LoadVideoWithOrig(
            path=input_video_files[0],
            img_size=opt.img_size,
            process_img_size=opt.process_img_size,
        )
    result_filename = os.path.join(result_root, "results.txt")

    frame_dir = None if opt.output_format == "text" else osp.join(result_root, "frame")
    eval_seq(
        opt,
        dataloader,
        "mot",
        result_filename,
        save_dir=frame_dir,
        show_image=False,
        use_cuda=opt.gpus != [-1],
    )

    output_video_path = osp.join(result_root, "tracking_output.mov")

    if opt.output_format == "video":
        cmd_str = "ffmpeg -f image2 -i {}/%05d.png -b:v 5000k -c:v mpeg4 {}".format(
            osp.join(result_root, "frame"), output_video_path
        )
        # trunk-ignore(bandit/B605)
        os.system(cmd_str)

    if opt.output_format in ["video", "live_video"]:
        output_video_with_audio_path = osp.join(
            result_root, "tracking_output-with-audio.mov"
        )
        copy_audio(
            input_video_files[0], output_video_path, output_video_with_audio_path
        )


if __name__ == "__main__":
    opt = opts().init()
    opt.output_format = "live_video"
    demo(opt)
