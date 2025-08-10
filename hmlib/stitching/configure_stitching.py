import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import hockeymom
from hmlib.config import get_game_config_private, get_game_dir, get_nested_value, save_private_config, set_nested_value
from hmlib.hm_opts import hm_opts
from hmlib.stitching.control_points import calculate_control_points
from hmlib.stitching.hugin import configure_control_points, load_pto_file, save_pto_file
from hmlib.utils.audio import load_audio_as_tensor
from hmlib.utils.path import add_suffix_to_filename
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import extract_frame_image

from .synchronize import configure_synchronization


def get_multiblend_bin() -> str:
    parent_dir: str = os.path.join(os.path.join(os.path.dirname(hockeymom.__file__)), "..")
    multiblend_app_name: str = "multiblend"
    local_multiblend: str = os.path.join(parent_dir, multiblend_app_name)
    if os.path.exists(local_multiblend):
        return os.path.realpath(local_multiblend)
    return multiblend_app_name


def get_tiff_tag_value(tiff_tag):
    if len(tiff_tag.value) == 1:
        return tiff_tag.value
    assert len(tiff_tag.value) == 2
    numerator, denominator = tiff_tag.value
    return float(numerator) / denominator


def is_older_than(file1: str, file2: str):
    try:
        mtime1 = os.path.getmtime(file1)
        mtime2 = os.path.getmtime(file2)
        return mtime2 < mtime1
    except OSError:
        return None


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


def get_extracted_frame_image_file(video_name: str, dir_name: Optional[str] = None):
    if dir_name:
        video_name = os.path.join(dir_name, video_name)
    file_name_without_extension, _ = os.path.splitext(video_name)
    return file_name_without_extension + ".png"


def extract_frames(
    video_left: str,
    left_frame_number: int,
    video_right: str,
    right_frame_number: int,
):
    # Absolute paths
    assert "/" in video_left
    assert "/" in video_right
    left_output_image_file = get_extracted_frame_image_file(video_left)

    right_output_image_file = get_extracted_frame_image_file(video_right)

    if not os.path.exists(left_output_image_file):
        extract_frame_image(
            video_left,
            frame_number=left_frame_number,
            dest_image=left_output_image_file,
        )
    if not os.path.exists(right_output_image_file):
        extract_frame_image(
            video_right,
            frame_number=right_frame_number,
            dest_image=right_output_image_file,
        )

    return left_output_image_file, right_output_image_file


def build_stitching_project(
    project_file_path: str,
    image_files: List[str],
    max_control_points: int,
    skip_if_exists: bool = True,
    test_blend: bool = True,
    fov: int = 108,
    scale: Optional[float] = None,
    force: bool = False,
):
    pto_path = Path(project_file_path)
    dir_name = pto_path.parent
    hm_project = project_file_path
    autooptimiser_out = os.path.join(dir_name, "autooptimiser_out.pto")
    assert autooptimiser_out != hm_project
    if skip_if_exists and (
        os.path.exists(project_file_path)
        and os.path.exists(autooptimiser_out)
        and not is_older_than(project_file_path, autooptimiser_out)
    ):
        print(f"Project file already exists (skipping project creation): {autooptimiser_out}")
        return True
    assert len(image_files) == 2
    left_image_file = image_files[0]
    right_image_file = image_files[1]

    curr_dir = os.getcwd()
    os.chdir(dir_name)
    try:
        use_hugin = False
        if not os.path.exists(hm_project) or force:
            cmd = [
                "pto_gen",
                "-p",
                "0",
                "-o",
                hm_project,
                "-f",
                str(fov),
                left_image_file,
                right_image_file,
            ]
            cmd_str = " ".join(cmd)
            os.system(cmd_str)
        else:
            use_hugin = True
        if True:
            configure_control_points(
                output_directory=str(dir_name),
                project_file_path=hm_project,
                image0=left_image_file,
                image1=right_image_file,
                max_control_points=max_control_points,
                force=True,
                use_hugin=use_hugin,
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
        if scale and scale != 1.0:
            cmd += [
                "-x",
                str(scale),
            ]
        os.system(" ".join(cmd))

        # Output mapping files
        cmd = [
            "nona",
            "-m",
            "TIFF_m",
            "-z",
            "NONE",
            "--bigtiff",
            "-c",
            "-o",
            "mapping_",
            autooptimiser_out,
        ]
        os.system(" ".join(cmd))
        seam_file: str = os.path.join(dir_name, "seam_file.png")
        cmd = [
            "enblend",
            f"--save-masks={seam_file}",
            "-o",
            os.path.join(dir_name, "panorama.tif"),
            os.path.join(dir_name, "mapping_????.tif"),
        ]
        os.system(" ".join(cmd))
        # See if it came out with a reasonable seam file
        distribution: Dict[int, float] = get_pixel_value_percentages(seam_file)
        # Really, it should be way above this number for a good seam, but so far
        # the "broken case" is much below this number (like 0.5%).
        kMinAllowableSeamPercent: float = 10.0
        if any(pct < kMinAllowableSeamPercent for pct in distribution.values()):
            print(f"Warning: seam file {seam_file} has low seam values, indicating a bad seam.")
            for val, pct in sorted(distribution.items()):
                print(f"Seam value {val:3d}: {pct:5.2f}%")
            # Delete the seam file so that it doesn't get used accidentally
            os.remove(seam_file)
            # If the seam is bad, try using multiblend instead
            cmd = [
                get_multiblend_bin(),
                f"--save-seams={seam_file}",
                "-o",
                os.path.join(dir_name, "panorama.tif"),
                os.path.join(dir_name, "mapping_????.tif"),
            ]
            os.system(" ".join(cmd))
            # Check again (should be ok now unless the stitch is really bad)
            distribution: Dict[int, float] = get_pixel_value_percentages(seam_file)
            if any(pct < kMinAllowableSeamPercent for pct in distribution.values()):
                print(f"Warning: seam file {seam_file} has low seam values, indicating a bad seam.")
                for val, pct in sorted(distribution.items()):
                    print(f"Seam value {val:3d}: {pct:5.2f}%")
                return False

    finally:
        os.chdir(curr_dir)
    return True


def get_pixel_value_percentages(image_path: str) -> Dict[int, float]:
    """
    Opens a grayscale image and computes the percentage of each pixel value.

    Args:
        image_path (str): Path to the input PNG.

    Returns:
        Dict[int, float]: Mapping from pixel value (0–255) to percentage of image,
                          as a float in [0.0, 100.0].
    """
    # Open image and ensure it's in 8-bit grayscale mode
    with Image.open(image_path) as img:
        gray: Image.Image = img.convert("L")
        arr: np.ndarray = np.array(gray, dtype=np.uint8)

    # Total number of pixels
    total: int = arr.size

    # Count occurrences of each value
    counts: np.ndarray = np.bincount(arr.flatten(), minlength=256)

    # Build percentage dict, omitting values with zero count
    percentages: Dict[int, float] = {value: (count / total) * 100.0 for value, count in enumerate(counts) if count > 0}

    return percentages


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


def configure_video_stitching(
    dir_name: str,
    video_left: str,
    video_right: str,
    max_control_points: int,
    project_file_name: str = "hm_project.pto",
    left_frame_offset: int = None,
    right_frame_offset: int = None,
    base_frame_offset: int = 0,
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
    pto_project_file: str = os.path.join(dir_name, project_file_name)
    autooptimiser_out: str = os.path.join(dir_name, "autooptimiser_out.pto")
    if (
        force
        or not os.path.exists(pto_project_file)
        or not os.path.exists(autooptimiser_out)
        or (os.path.exists(pto_project_file) and is_older_than(pto_project_file, autooptimiser_out))
    ):
        left_image_file, right_image_file = extract_frames(
            video_left,
            base_frame_offset + left_frame_offset,
            video_right,
            base_frame_offset + right_frame_offset,
        )

        build_stitching_project(
            project_file_path=pto_project_file,
            image_files=[left_image_file, right_image_file],
            max_control_points=max_control_points,
            force=force,
            skip_if_exists=not force,
        )

    return pto_project_file, left_frame_offset, right_frame_offset
