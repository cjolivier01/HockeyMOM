import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import moviepy.editor as mp
import numpy as np
import scipy
import tifffile
import torch
import torch.nn.functional as F

from hmlib.config import (
    get_game_config_private,
    get_game_dir,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.ffmpeg import extract_frame_image
from hmlib.hm_opts import hm_opts
from hmlib.stitching.control_points import calculate_control_points
from hmlib.stitching.hugin import configure_control_points, load_pto_file, save_pto_file
from hmlib.utils.path import add_suffix_to_filename

MULTIBLEND_BIN = os.path.join(os.environ["HOME"], "src", "multiblend", "src", "multiblend")


def synchronize_by_audio(
    file0_path: str,
    file1_path: str,
    seconds: int = 15,
    create_new_clip: bool = False,
    device: torch.device = None,
):
    # Load the videos
    print("Openning videos...")
    full_video0 = mp.VideoFileClip(file0_path)
    full_video1 = mp.VideoFileClip(file1_path)

    seconds = min(seconds, min(full_video0.duration - 0.5, full_video1.duration - 0.5))

    video0 = full_video0.subclip(0, seconds)
    video1 = full_video1.subclip(0, seconds)

    video_1_frame_count = video0.fps * video0.duration
    # video_2_frame_count = video1.fps * video0.duration

    # Load audio from the videos
    print("Loading audio...")
    audio1 = video0.audio.to_soundarray()
    audio2 = video1.audio.to_soundarray()

    audio_items_per_frame_1 = audio1.shape[0] / video_1_frame_count
    # audio_items_per_frame_2 = audio2.shape[0] / video_2_frame_count

    # Calculate the cross-correlation of audio1 and audio2
    print("Calculating cross-correlation...")
    if device is None:
        # correlation = np.correlate(audio1[:, 0], audio2[:, 0], mode="full")
        correlation = scipy.signal.correlate(audio1[:, 0], audio2[:, 0], mode="full")
        lag = np.argmax(correlation) - len(audio1) + 1
    else:
        audio1 = torch.from_numpy(audio1[:, 0]).unsqueeze(0).unsqueeze(0).to(device)
        audio2 = torch.from_numpy(audio2[:, 0]).unsqueeze(0).unsqueeze(0).to(device)

        # Compute correlation using convolution
        # The 'groups' argument ensures a separate convolution for each batch
        correlation = F.conv1d(audio1, audio2.flip(-1), padding=audio2.size(-1) - 1, groups=1)

        # Remove added dimensions to get the final 1D correlation tensor
        correlation = correlation.squeeze()
        lag, idx = torch.argmax(correlation) - len(audio1) + 1

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
                synchronized_video = full_video1.subclip(max(0, -time_offset), full_video1.duration)
                new_file_name = add_suffix_to_filename(file1_path, "sync")
            else:
                synchronized_video = full_video0.subclip(max(0, -time_offset), full_video0.duration)
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
    left_frame_offset = frame_offset if frame_offset > 0 else 0
    right_frame_offset = -frame_offset if frame_offset < 0 else 0

    return left_frame_offset, right_frame_offset


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
    return xpos, ypos


def extract_frames(
    dir_name: str,
    video_left: str,
    left_frame_number: int,
    video_right: str,
    right_frame_number: int,
):
    file_name_without_extension, _ = os.path.splitext(video_left)
    left_output_image_file = os.path.join(dir_name, file_name_without_extension + ".png")

    file_name_without_extension, _ = os.path.splitext(video_right)
    right_output_image_file = os.path.join(dir_name, file_name_without_extension + ".png")

    if not os.path.exists(left_output_image_file):
        extract_frame_image(
            os.path.join(dir_name, video_left),
            frame_number=left_frame_number,
            dest_image=left_output_image_file,
        )
    if not os.path.exists(right_output_image_file):
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
    force: bool = False,
):
    # TODO: need to fix this function
    # assert project_file_path.endswith("my_project.pto")
    pto_path = Path(project_file_path)
    dir_name = pto_path.parent
    hm_project = os.path.join(dir_name, "hm_project.pto")
    autooptimiser_out = os.path.join(dir_name, "autooptimiser_out.pto")
    skip_if_exists = False

    if skip_if_exists and (os.path.exists(autooptimiser_out) or os.path.exists(project_file_path)):
        print(f"Project file already exists (skipping project creation): {autooptimiser_out}")
        return True

    assert len(image_files) == 2
    left_image_file = image_files[0]
    right_image_file = image_files[1]

    curr_dir = os.getcwd()
    try:
        os.chdir(dir_name)
        cmd = [
            "pto_gen",
            "-p",
            "1",
            "-o",
            hm_project,
            "-f",
            str(fov),
            left_image_file,
            right_image_file,
        ]
        cmd_str = " ".join(cmd)
        os.system(cmd_str)

        if True:
            configure_control_points(
                output_directory=dir_name,
                project_file_path=hm_project,
                image0=left_image_file,
                image1=right_image_file,
                force=True,
            )
        else:
            cmd = ["cpfind", "--linearmatch", hm_project, "-o", project_file_path]
            os.system(" ".join(cmd))

        # autooptimiser (RANSAC?)
        cmd = [
            "autooptimiser",
            "-a",
            "-m",
            "-l",
            "-s",
            "-o",
            autooptimiser_out,
            hm_project,
        ]
        os.system(" ".join(cmd))

        # Output mapping files
        cmd = [
            "nona",
            "-m",
            "TIFF_m",
            "-c",
            "-o",
            "mapping_",
            autooptimiser_out,
        ]
        os.system(" ".join(cmd))

        if test_blend:
            cmd = [
                "nona",
                "-m",
                "TIFF_m",
                "-o",
                project_file_path,
                autooptimiser_out,
            ]
            os.system(" ".join(cmd))
            if os.path.exists(MULTIBLEND_BIN):
                cmd = [
                    MULTIBLEND_BIN,
                    "-o",
                    os.path.join(dir_name, "panorama.tif"),
                    os.path.join(dir_name, autooptimiser_out + "*.tif"),
                ]
            else:
                print(f"Could not find blender for sample panorama creation: {MULTIBLEND_BIN}")
            os.system(" ".join(cmd))
    finally:
        os.chdir(curr_dir)
    return True


def load_or_calculate_control_points(
    game_id: str,
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    force: bool = False,
    device: Optional[torch.device] = None,
    save: bool = True,
) -> Dict[str, torch.Tensor]:
    config = get_game_config_private(game_id=game_id)
    control_points = get_nested_value(config, "game.stitching.control_points") if not force else {}
    if force or not control_points:
        # Calculate them...
        control_points = calculate_control_points(image=image0, image1=image1, device=device)
        assert "m_kpts0" in control_points and "m_kpts1" in control_points
        # Remove stuff we don't want
        control_points.pop("kpts0")
        control_points.pop("kpts1")

        if save:
            config = set_nested_value(config, "game.stitching.control_points", control_points)
            save_private_config(game_id=game_id, data=config)


def configure_synchronization(
    game_id: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    audio_sync_seconds: float = 15.0,
    force: bool = False,
) -> Dict[str, float]:
    config = get_game_config_private(game_id=game_id)
    # set_nested_value(config, "game.stitching.frame_offsets", {"left": 1.0, "right": 0.0})
    frame_offsets = (
        get_nested_value(config, "game.stitching.frame_offsets", None) if not force else dict()
    )
    if (
        force
        or not frame_offsets
        or frame_offsets.get("left") is None
        or frame_offsets.get("right") is None
    ):
        # Calculate by audio
        game_dir = get_game_dir(game_id=game_id)
        lfo, rfo = synchronize_by_audio(
            file0_path=os.path.join(game_dir, video_left),
            file1_path=os.path.join(game_dir, video_right),
            seconds=audio_sync_seconds,
        )
        if frame_offsets is None:
            frame_offsets = {}
        frame_offsets["left"] = float(lfo)
        frame_offsets["right"] = float(rfo)
        set_nested_value(config, "game.stitching.frame_offsets", frame_offsets)
        save_private_config(game_id=game_id, data=config)
    else:
        print(
            f"Preconfigured: left frame offset: {frame_offsets['left']}, right frame offset: {frame_offsets['right']}"
        )
    # Not get form the config
    frame_offsets = get_nested_value(config, "game.stitching.frame_offsets")
    return frame_offsets


def configure_video_stitching(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    project_file_name: str = "my_project.pto",
    left_frame_offset: int = None,
    right_frame_offset: int = None,
    base_frame_offset: int = 100,
    audio_sync_seconds: int = 15,
    force: bool = False,
):
    if left_frame_offset is None or right_frame_offset is None:
        if True:
            frame_offsets = configure_synchronization(
                game_id=dir_name.split("/")[-1],
                video_left=video_left,
                video_right=video_right,
                audio_sync_seconds=audio_sync_seconds,
                force=force,
            )
            left_frame_offset = float(frame_offsets["left"])
            right_frame_offset = float(frame_offsets["right"])
        else:
            left_frame_offset, right_frame_offset = synchronize_by_audio(
                file0_path=os.path.join(dir_name, video_left),
                file1_path=os.path.join(dir_name, video_right),
                seconds=audio_sync_seconds,
            )

    # PTO Project File

    force = True

    game_id = dir_name.split("/")[-1]

    pto_project_file = os.path.join(dir_name, project_file_name)
    if force or not os.path.exists(pto_project_file):
        left_image_file, right_image_file = extract_frames(
            dir_name,
            video_left,
            base_frame_offset + left_frame_offset,
            video_right,
            base_frame_offset + right_frame_offset,
        )

        build_stitching_project(
            project_file_path=pto_project_file,
            image_files=[left_image_file, right_image_file],
            force=force,
            skip_if_exists=not force,
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
    # Currently, expects files to be named like
    # "left-0.mp4", "right-0.mp4" and in /home/Videos directory
    opts = hm_opts()
    args = opts.parse()
    synchronize_by_audio(
        file0_path=f"{os.environ['HOME']}/Videos/{args.game_id}/left.mp4",
        file1_path=f"{os.environ['HOME']}/Videos/{args.game_id}/right.mp4",
    )
