"""
Experiments in stitching
"""
import os
import time
from pathlib import Path
from hmlib.opts import opts
from hmlib.ffmpeg import copy_audio
from hmlib.ui.mousing import draw_box_with_mouse
from hmlib.tracking_utils.log import logger

# from lib.tiff import print_geotiff_info
from hmlib.stitch_synchronize import (
    # synchronize_by_audio,
    # build_stitching_project,
    # extract_frames,
    configure_video_stitching,
)

from hmlib.datasets.dataset.stitching import (
    StitchDataset,
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=int,
        help="Video directory to find 'left.mp4' and 'right.mp4'",
    )
    return parser


def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    lfo: int = None,
    rfo: int = None,
    project_file_name: str = "my_project.pto",
    start_frame_number:int = 0,
    max_frames: int = None,
    output_stitched_video_file: str = "./stitched_output.avi",
):
    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left,
        video_right,
        project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
    )

    data_loader = StitchDataset(
        video_file_1=f"{dir_name}/{video_left}",
        video_file_2=f"{dir_name}/{video_right}",
        pto_project_file=pto_project_file,
        video_1_offset_frame=lfo,
        video_2_offset_frame=rfo,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
        num_workers=1,
    )

    frame_count = 0
    start = None
    for i, stitched_image in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Read frame {start_frame_number + i}")
        frame_count += 1
        if i == 1:
            # draw_box_with_mouse(stitched_image, destroy_all_windows_after=True)
            start = time.time()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )
    return lfo, rfo


def main(args):
    if args.video_dir is None:
        args.video_dir = os.path.join(os.environ["HOME"], "Videos", "stockton")
    video_left = "left.mp4"
    video_right = "right.mp4"
    # lfo = 13
    # rfo = 0
    lfo, rfo = stitch_videos(
        args.video_dir,
        video_left,
        video_right,
        lfo=lfo,
        rfo=rfo,
    )


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
    print("Done.")
