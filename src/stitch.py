"""
Experiments in stitching
"""
import os
import time

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

from lib.ffmpeg import copy_audio
from lib.ui.mousing import draw_box_with_mouse
from lib.tracking_utils.log import logger
from lib.datasets.dataset.stitching import (
    StitchDataset,
    build_stitching_project,
)


def stitch_videos():
    vid_dir = os.path.join(os.environ["HOME"], "Videos")

    # PTO Project File
    pto_project_file = os.path.join(os.environ["HOME"], "Videos", "my_project.pto")

    build_stitching_project(pto_project_file)

    # start_frame_number = 2000
    start_frame_number = 0

    max_frames = 10

    output_stitched_video_file = "./stitched_output.avi"

    data_loader = StitchDataset(
        video_file_1=f"{vid_dir}/left.mp4",
        video_file_2=f"{vid_dir}/right.mp4",
        pto_project_file=pto_project_file,
        video_1_offset_frame=217,
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
    stitch_videos()


if __name__ == "__main__":
    main()
    print("Done.")
