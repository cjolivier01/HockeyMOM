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


def setup_stitching_project(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    project_file_name: str = "my_project.pto",
):
    lfo, rfo = synchronize_by_audio(
        file0_path=os.path.join(dir_name, video_left),
        file1_path=os.path.join(dir_name, video_right),
        seconds=15,
    )

    base_frame_offset = 800

    left_image_file, right_image_file = extract_frames(
        dir_name,
        video_left,
        base_frame_offset + lfo,
        video_right,
        base_frame_offset + rfo,
    )

    # HACK
    # left_image_file = os.path.join(dir_name, "left-1-small.png")
    # right_image_file = os.path.join(dir_name, "right-1-small.png")

    # PTO Project File
    pto_project_file = os.path.join(dir_name, project_file_name)

    build_stitching_project(
        pto_project_file, image_files=[left_image_file, right_image_file]
    )
    return pto_project_file, lfo, rfo


def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    project_file_name: str = "my_project.pto",
):
    # lfo, rfo = synchronize_by_audio(
    #     file0_path=os.path.join(dir_name, video_left),
    #     file1_path=os.path.join(dir_name, video_right),
    #     seconds=15,
    # )

    # base_frame_offset = 800

    # left_image_file, right_image_file = extract_frames(
    #     dir_name,
    #     video_left,
    #     base_frame_offset + lfo,
    #     video_right,
    #     base_frame_offset + rfo,
    # )

    # # HACK
    # # left_image_file = os.path.join(dir_name, "left-1-small.png")
    # # right_image_file = os.path.join(dir_name, "right-1-small.png")

    # # PTO Project File
    # pto_project_file = os.path.join(dir_name, project_file_name)

    # build_stitching_project(
    #     pto_project_file, image_files=[left_image_file, right_image_file]
    # )

    pto_project_file, lfo, rfo = setup_stitching_project(
        dir_name, video_left, video_right, project_file_name
    )

    # start_frame_number = 2000
    start_frame_number = 0

    max_frames = 100

    output_stitched_video_file = "./stitched_output.avi"

    data_loader = StitchDataset(
        video_file_1=f"{dir_name}/{video_left}",
        video_file_2=f"{dir_name}/{video_right}",
        pto_project_file=pto_project_file,
        video_1_offset_frame=lfo,
        video_2_offset_frame=rfo,
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
    video_left = "left-1.mp4"
    video_right = "right-1.mp4"
    # video_left = "left-1-small.avi"
    # video_right = "right-1-small.avi"

    #lfo, rfo = stitch_videos(dir_name, video_left, video_right)

    # lfo, rfo = 0, 91
    # if lfo < 0:
    #     copy_audio(
    #         video_left, output_video_path, output_video_with_audio_path
    #     )



if __name__ == "__main__":
    main()
    print("Done.")
