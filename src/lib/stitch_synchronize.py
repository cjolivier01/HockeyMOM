import os
import moviepy.editor as mp
import numpy as np

import os

# import sys
# import torch
# import torch.nn as nn
import numpy as np

# import time
# import cv2
# import threading
# import multiprocessing

from typing import List

from pathlib import Path
import tifffile

from hockeymom import core

from lib.tracking_utils import visualization as vis
from lib.ffmpeg import extract_frame_image


def synchronize_by_audio(
    file0_path: str, file1_path: str, seconds: int = 15, create_new_clip: bool = False
):
    # Load the videos
    print("Openning videos...")
    full_video0 = mp.VideoFileClip(file0_path)
    full_video1 = mp.VideoFileClip(file1_path)

    video0 = full_video0.subclip(0, seconds)
    video1 = full_video1.subclip(0, seconds)

    video_1_frame_count = video0.fps * video0.duration
    video_2_frame_count = video1.fps * video0.duration

    # Load audio from the videos
    print("Loading audio...")
    audio1 = video0.audio.to_soundarray()
    audio2 = video1.audio.to_soundarray()

    audio_items_per_frame_1 = audio1.shape[0] / video_1_frame_count
    audio_items_per_frame_2 = audio2.shape[0] / video_2_frame_count

    # Calculate the cross-correlation of audio1 and audio2
    print("Calculating cross-correlation...")
    correlation = np.correlate(audio1[:, 0], audio2[:, 0], mode="full")
    lag = np.argmax(correlation) - len(audio1) + 1

    # Calculate the time offset in seconds
    fps = video0.fps
    frame_offset = lag / audio_items_per_frame_1
    time_offset = frame_offset / fps

    print(f"Left frame offset: {frame_offset}")
    print(f"Time offset: {time_offset} seconds")

    # Synchronize video1 with video0
    if create_new_clip:
        print("Creating new subclip...")
        if frame_offset:
            if frame_offset < 0:
                synchronized_video = full_video1.subclip(
                    max(0, -time_offset), full_video1.duration
                )
                new_file_name = add_suffix_to_filename(file1_path, "sync")
            else:
                synchronized_video = full_video0.subclip(
                    max(0, -time_offset), full_video0.duration
                )
                new_file_name = add_suffix_to_filename(file0_path, "sync")

            # Write the synchronized video to a file
            print("Writing synchronized file...")
            synchronized_video.write_videofile(new_file_name, codec="libx264")
            synchronized_video.close()

    # Close the videos
    video0.close()
    video1.close()
    full_video0.close()
    full_video1.close()

    # Adjust to the starting frame number in each video (i.e. frame_offset might be a negative number)
    left_frame_offset = int(frame_offset if frame_offset > 0 else 0)
    right_frame_offset = int(-frame_offset if frame_offset < 0 else 0)

    return left_frame_offset, right_frame_offset


def add_suffix_to_filename(filename, suffix):
    base_name, extension = os.path.splitext(filename)
    new_filename = f"{base_name}_{suffix}{extension}"
    return new_filename


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
    right_frame_number: int,
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
        print(
            f"Project file already exists (skipping project creatio9n): {project_file_path}"
        )
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
                os.path.join(dir_name, f"{pto_path.name}*.tif"),
            ]
            os.system(" ".join(cmd))
    finally:
        os.chdir(curr_dir)
    return True


def _add_suffix(file_path: str, suffix: str):
    root, ext = os.path.splitext(file_path)
    new_filepath = f"{root}{suffix}{ext}"
    return new_filepath


def configure_video_stitching(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    project_file_name: str = "my_project.pto",
    left_frame_offset: int = None,
    right_frame_offset: int = None,
    base_frame_offset: int = 800,
    audio_sync_seconds: int = 15,
):
    if left_frame_offset is None or right_frame_offset is None:
        print("Syncronizing...")
        left_frame_offset, right_frame_offset = synchronize_by_audio(
            file0_path=os.path.join(dir_name, video_left),
            file1_path=os.path.join(dir_name, video_right),
            seconds=audio_sync_seconds,
        )

    # PTO Project File
    pto_project_file = os.path.join(dir_name, project_file_name)
    if not os.path.exists(pto_project_file):
        # Try to use a frame during the second video (if it exists)
        extract_path_1 = _add_suffix(video_left, "-1")
        extract_path_2 = _add_suffix(video_right, "-1")
        if not os.path.exists(extract_path_1) or not os.path.exists(extract_path_2):
            extract_path_1 = video_left
            extract_path_2 = video_right
        left_image_file, right_image_file = extract_frames(
            dir_name,
            video_left,
            base_frame_offset + left_frame_offset,
            video_right,
            base_frame_offset + right_frame_offset,
        )

        build_stitching_project(
            pto_project_file, image_files=[left_image_file, right_image_file]
        )

    return pto_project_file, left_frame_offset, right_frame_offset


def find_sitched_roi(image):
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


if __name__ == "__main__":
    # video_number = 0
    # Currently, expects files to be named like
    # "left-0.mp4", "right-0.mp4" and in /home/Videos directory
    synchronize_by_audio(
        file0_path=f"{os.environ['HOME']}/Videos/sabercats-parts/left-1.mp4",
        file1_path=f"{os.environ['HOME']}/Videos/sabercats-parts/right-1.mp4",
        # file0_path=f"{os.environ['HOME']}/Videos/left-{video_number}.mp4",
        # file1_path=f"{os.environ['HOME']}/Videos/right-{video_number}.mp4",
    )
