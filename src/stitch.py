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

def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    lfo: int = None,
    rfo: int = None,
    project_file_name: str = "my_project.pto",
):
    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left,
        video_right,
        project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
    )

    start_frame_number = 200
    # start_frame_number = 0

    max_frames = 300

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
        num_workers=1,
    )

    frame_count = 0
    start = None
    for i, stitched_image in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Read frame {start_frame_number + i}")
        frame_count += 1
        if i == 1:
            #draw_box_with_mouse(stitched_image, destroy_all_windows_after=True)
            start = time.time()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )
    return lfo, rfo


def main():
    # stitch_images(
    #     "/mnt/data/Videos/vacaville/my_project0000.tif",
    #     "/mnt/data/Videos/vacaville/my_project0001.tif",
    # )

    #dir_name = os.path.join(os.environ["HOME"], "Videos", "sabercats-parts")
    dir_name = os.path.join(os.environ["HOME"], "Videos", "stockton")
    video_left = "left.mp4"
    video_right = "right.mp4"
    # video_left = "left-1-small.avi"
    # video_right = "right-1-small.avi"
    # lfo = 0
    # rfo = 92
    lfo = 13
    rfo = 0
    lfo, rfo = stitch_videos(
        dir_name,
        video_left,
        video_right,
        lfo=lfo,
        rfo=rfo,
    )

    # lfo, rfo = 0, 91
    # if lfo < 0:
    #     copy_audio(
    #         video_left, output_video_path, output_video_with_audio_path
    #     )
    pass


if __name__ == "__main__":
    main()
    print("Done.")
