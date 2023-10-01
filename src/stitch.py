"""
Experiments in stitching
"""
import os
import time

from pathlib import Path

from lib.opts import opts
from lib.ffmpeg import copy_audio
from lib.ui.mousing import draw_box_with_mouse
from lib.tracking_utils.log import logger
from lib.stitch_synchronize import synchronize_by_audio
from lib.datasets.dataset.stitching import (
    StitchDataset,
    build_stitching_project,
    extract_frames,
)


def stitch_videos(dir_name: str):
    vid_dir = os.path.join(os.environ["HOME"], "Videos")

    # PTO Project File
    pto_project_file = os.path.join(
        os.environ["HOME"], "Videos", "sabercats-parts", "my_project.pto"
    )

    build_stitching_project(pto_project_file)

    # start_frame_number = 2000
    start_frame_number = 0

    max_frames = 10

    output_stitched_video_file = "./stitched_output.avi"

    data_loader = StitchDataset(
        video_file_1=f"{vid_dir}/left-1.mp4",
        video_file_2=f"{vid_dir}/right-1.mp4",
        pto_project_file=pto_project_file,
        video_1_offset_frame=91,
        video_2_offset_frame=0,
        # remap_thread_count=1,
        # blend_thread_count=1,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
    )

    frame_count = 0
    start = None
    for i, stitched_image in enumerate(data_loader):
        print(f"Read frame {i}")
        # if i == 0:
        #     find_roi(stitched_image)
        frame_count += 1
        if i == 1:
            start = time.time()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )


def main():
    dir_name = os.path.join(os.environ["HOME"], "Videos", "sabercats-parts")
    video_left = "left-1-small.avi"
    video_right = "right-1-small.avi"

    left_frame_offset = synchronize_by_audio(
        file0_path=os.path.join(dir_name, video_left),
        file1_path=os.path.join(dir_name, video_right),
        seconds=15,
    )

    if left_frame_offset < 0:
        extract_frames(dir_name, video_left, -left_frame_offset, video_right, 0)
    else:
        extract_frames(dir_name, video_left, 0, video_right, left_frame_number)

    stitch_videos(dir_name)


if __name__ == "__main__":
    main()
    print("Done.")
