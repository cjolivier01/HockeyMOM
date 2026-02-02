"""High-level helpers for configuring two-camera stitching projects.

This module wraps Hugin PTO generation, control-point creation, seam
estimation and per-game synchronization into reusable functions.

@see @ref hmlib.stitching.control_points.calculate_control_points "calculate_control_points"
@see @ref hmlib.stitching.hugin.configure_control_points "configure_control_points"
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tifffile
import torch

from hmlib.config import (
    get_game_config_private,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.stitching.control_points import calculate_control_points
from hmlib.stitching.hugin import configure_control_points
from hmlib.video.video_stream import extract_frame_image

from .synchronize import configure_synchronization


def _resolve_local_binary(executable: str) -> Optional[str]:
    """Return a package-local binary path if available.

    Prefers `<hmlib_root>/bin/<executable>` both for Bazel runfiles
    and for installed wheels.
    """
    base_dir = Path(__file__).resolve().parent.parent
    bin_path = base_dir / "bin" / executable
    if bin_path.is_file() and os.access(bin_path, os.X_OK):
        return str(bin_path)
    return None


def _get_color_adjustment_adders(
    game_id: str,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Return per-side RGB adders from game config, if present.

    Expected config structure in private game config (e.g. $HOME/Videos/<game_id>/config.yaml):

    game:
      stitching:
        color_adjustment:
          left:
            r: 45
            g: 35
            b: 56
          right:
            r: 48
            g: 35
            b: 49
    """
    cfg = get_game_config_private(game_id=game_id)
    node = get_nested_value(cfg, "game.stitching.color_adjustment")
    if not isinstance(node, dict):
        return None, None

    def _side(name: str) -> Optional[List[float]]:
        side = node.get(name)
        if not isinstance(side, dict):
            return None
        try:
            r = float(side.get("r"))
            g = float(side.get("g"))
            b = float(side.get("b"))
            return [r, g, b]
        except Exception:
            return None

    return _side("left"), _side("right")


def _apply_color_adders_to_image_file(image_path: str, adders: Optional[List[float]]) -> None:
    """Apply per-channel RGB adders to a PNG file in-place.

    - Operates on uint8 images, clamping to [0, 255].
    - No-op if adders is None or the file is missing.
    """
    if not adders:
        return
    if not os.path.exists(image_path):
        return
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        # Ensure we have at least 3 channels; treat input as BGR
        if img.ndim == 2:
            # Grayscale; treat all channels the same by broadcasting R/G/B adders
            arr = img.astype(np.float32)
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = img.astype(np.float32)
            # Convert BGR -> RGB for intuitive R/G/B indexing
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        # Apply adders to RGB channels
        arr[..., 0] = np.clip(arr[..., 0] + adders[0], 0.0, 255.0)
        arr[..., 1] = np.clip(arr[..., 1] + adders[1], 0.0, 255.0)
        arr[..., 2] = np.clip(arr[..., 2] + adders[2], 0.0, 255.0)

        # Convert back to BGR before saving so downstream OpenCV users see expected ordering
        arr_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, arr_bgr)
    except Exception:
        # Do not fail stitching configuration if adjustment fails
        pass


def get_multiblend_bin() -> str:
    """Return the path to the `multiblend` binary, preferring a workspace-local build."""
    resolved = _resolve_local_binary("multiblend")
    if resolved is not None:
        return resolved
    return "multiblend"


def get_enblend_bin() -> str:
    """Return the path to the `enblend` binary, preferring a workspace-local build."""
    resolved = _resolve_local_binary("enblend")
    if resolved is not None:
        return resolved
    return "enblend"


def get_tiff_tag_value(tiff_tag):
    """Decode a TIFF rational tag into a Python scalar."""
    if len(tiff_tag.value) == 1:
        return tiff_tag.value
    assert len(tiff_tag.value) == 2
    numerator, denominator = tiff_tag.value
    return float(numerator) / denominator


def is_older_than(file1: str, file2: str):
    """Return True if `file2` is older than `file1`, or None if missing."""
    try:
        mtime1 = os.path.getmtime(file1)
        mtime2 = os.path.getmtime(file2)
        return mtime2 < mtime1
    except OSError:
        return None


def get_image_geo_position(tiff_image_file: str):
    """Return integer pixel position (x, y) from a mapping TIFF file."""
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
    """Return the PNG path used when extracting a frame from a video file."""
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
    """Extract one frame from each side video to PNGs on disk.

    @param video_left: Absolute path to left video.
    @param left_frame_number: Frame index to extract from left video.
    @param video_right: Absolute path to right video.
    @param right_frame_number: Frame index to extract from right video.
    @return: Tuple of paths ``(left_png, right_png)``.
    """
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
    """Create or update a Hugin PTO project and seam masks for two images.

    @param project_file_path: Output PTO project path.
    @param image_files: List of two input image paths (left, right).
    @param max_control_points: Maximum number of control points to use.
    @param skip_if_exists: If True, reuse existing project when up-to-date.
    @param test_blend: Whether to create/test seam masks using `enblend`.
    @param fov: Horizontal field-of-view in degrees.
    @param scale: Optional scale factor passed to `autooptimiser`.
    @param force: If True, always rebuild, ignoring mtimes.
    @return: True on success, False if seam quality tests fail.
    """
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
            run_result = os.system(cmd_str)
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
            get_enblend_bin(),
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
    arr: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return {}

    # Total number of pixels
    total: int = arr.size

    # Count occurrences of each value
    counts: np.ndarray = np.bincount(arr.flatten(), minlength=256)

    # Build percentage dict, omitting values with zero count
    percentages: Dict[int, float] = {
        value: (count / total) * 100.0 for value, count in enumerate(counts) if count > 0
    }

    return percentages


def load_or_calculate_control_points(
    game_id: str,
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    force: bool = False,
    device: Optional[torch.device] = None,
    save: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load game-specific control points or compute them with LightGlue.

    @param game_id: Game identifier used to resolve private config.
    @param image0: First image (path or tensor).
    @param image1: Second image (path or tensor).
    @param force: If True, ignore cached control points and recompute.
    @param device: Optional device for LightGlue/SuperPoint.
    @param max_control_points: Maximum number of points to keep.
    @param output_directory: Optional directory for debug visualizations.
    @param save: If True, persist control points into game config.
    @return: Dict with at least ``m_kpts0`` and ``m_kpts1`` tensors.
    """
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
    """Configure a two-camera stitching project from game videos.

    Uses audio-based synchronization, frame extraction, optional per-side
    color adjustment and Hugin PTO generation to produce mapping TIFFs.

    @param dir_name: Game directory containing videos and config.
    @param video_left: Left-side video filename or path.
    @param video_right: Right-side video filename or path.
    @param max_control_points: Max number of control points to search.
    @param project_file_name: PTO filename inside ``dir_name``.
    @param left_frame_offset: Manually specified left frame offset (or None).
    @param right_frame_offset: Manually specified right frame offset (or None).
    @param base_frame_offset: Global offset added to both sides.
    @param audio_sync_seconds: Seconds of audio used for synchronization.
    @param force: If True, recompute PTO and seam even if up-to-date.
    @return: Tuple ``(pto_project_file, left_frame_offset, right_frame_offset)``.
    """
    if left_frame_offset is None or right_frame_offset is None:
        frame_offsets = configure_synchronization(
            game_id=dir_name.split("/")[-1],
            video_left=video_left,
            video_right=video_right,
            audio_sync_seconds=audio_sync_seconds,
            force=force,
        )
        left_frame_offset = float(frame_offsets["left"])
        right_frame_offset = float(frame_offsets["right"])

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

        # Apply optional per-side color adjustments to extracted PNGs
        # before they are used by Hugin / PTO configuration.
        game_id = dir_name.split("/")[-1]
        left_adders, right_adders = _get_color_adjustment_adders(game_id=game_id)
        _apply_color_adders_to_image_file(left_image_file, left_adders)
        _apply_color_adders_to_image_file(right_image_file, right_adders)

        build_stitching_project(
            project_file_path=pto_project_file,
            image_files=[left_image_file, right_image_file],
            max_control_points=max_control_points,
            force=force,
            skip_if_exists=not force,
        )

    return pto_project_file, left_frame_offset, right_frame_offset
