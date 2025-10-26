import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from hmlib.config import (
    get_game_config_private,
    get_game_dir,
    get_nested_value,
    prepend_root_dir,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.models.loader import get_model_config
from hmlib.segm.ice_rink import find_ice_rink_masks
from hmlib.ui.show import show_image
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.image import image_height, image_width
from hmlib.utils.video import load_first_video_frame

# GoPro pattern is GXzzxxxx.mp4, where zz is chapter number and zzzz is video
# number
GOPRO_FILE_PATTERN: str = r"^G[A-Z][0-9]{6}\.(MP4|mp4)$"
INSTA360_FILE_PATTERN: str = r"^VID_[0-9]{8}_[0-9]{6}_[0-9]{3}\.(MP4|mp4)$"
LEFT_PART_FILE_PATTERN: str = r"left-[0-9]\.mp4$"
RIGHT_PART_FILE_PATTERN: str = r"right-[0-9]\.mp4$"
LEFT_FILE_PATTERN: str = r"left.mp4"
RIGHT_FILE_PATTERN: str = r"right.mp4"


VideoChapter = Dict[Union[int, str], Any]
VideosDictKey = Union[int, str]
VideosDict = Dict[VideosDictKey, VideoChapter]


def gopro_get_video_and_chapter(filename: Path) -> Tuple[int, int]:
    """
    Get video and chapter number

    Returns: (video number, chapter number)
    """
    name = Path(filename).stem
    assert name[0] == "G"
    assert name[1] in {"H", "X"}
    return int(name[4:8]), int(name[2:4])


def insta360_get_video_and_chapter(filename: Path) -> Tuple[int, int]:
    """Return (video identifier, chapter number) for Insta360 clips."""
    name = Path(filename).stem
    tokens = name.split("_")
    assert tokens[0] == "VID"
    assert len(tokens) >= 4
    date_token, video_token, chapter_token = tokens[1], tokens[2], tokens[3]
    video_id = int(f"{date_token}{video_token}")
    return video_id, int(chapter_token)


def get_lr_part_number(filename: str) -> int:
    """
    Get video and chapter number

    Returns: (video number, chapter number)
    """
    name = Path(filename).stem
    tokens = name.split("-")
    return int(tokens[-1])


def find_matching_files(re_pattern: str, directory: str) -> List[str]:
    # Regex to match the file format 'GLXXXXXX.mp4'
    pattern = re.compile(re_pattern)

    # List to store the names of matching files
    matching_files: List[str] = []

    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        if pattern.match(filename):
            matching_files.append(os.path.join(directory, filename))

    return sorted(matching_files)


def prune_chapters(videos: VideosDict) -> Tuple[VideosDict, VideosDict]:
    """
    Prune out videos that don't have matching chapters.
    Returns Tuple:
        0: Videos/chapters with all matching chapters
        1: Videos/chapters that were extracted
    """
    matching: VideosDict = {}
    unmatching: VideosDict = {}
    all_chapters: Set[int] = set()
    for video, chapters in videos.items():
        # Not pruning anything yet
        pass
    return videos, {}


def get_available_videos(dir_name: str, prune: bool = False) -> VideosDict:
    """
    Get available videos in the given directory

    :return: # Video # / left|right -> Chapter # -> filename
    """
    gopro_files: List[str] = find_matching_files(re_pattern=GOPRO_FILE_PATTERN, directory=dir_name)
    # Video # / left|right -> Chapter # -> filename
    videos_dict: VideosDict = OrderedDict()
    for file in gopro_files:
        video, chapter = gopro_get_video_and_chapter(filename=file)
        if video not in videos_dict:
            videos_dict[video] = {}
        videos_dict[video][chapter] = file

    insta360_files: List[str] = find_matching_files(re_pattern=INSTA360_FILE_PATTERN, directory=dir_name)
    for file in insta360_files:
        video, chapter = insta360_get_video_and_chapter(filename=file)
        if video not in videos_dict:
            videos_dict[video] = {}
        videos_dict[video][chapter] = file

    # Plain left file
    files: List[str] = find_matching_files(re_pattern=LEFT_FILE_PATTERN, directory=dir_name)
    if files:
        assert len(files) == 1
        videos_dict["left"] = {}
        videos_dict["left"][1] = files[0]
    else:
        files = find_matching_files(re_pattern=LEFT_PART_FILE_PATTERN, directory=dir_name)
        if files:
            videos_dict["left"] = {}
            for file in sorted(files):
                videos_dict["left"][get_lr_part_number(file)] = file

    # Plain right file
    files: List[str] = find_matching_files(re_pattern=RIGHT_FILE_PATTERN, directory=dir_name)
    if files:
        assert len(files) == 1
        videos_dict["right"] = {}
        videos_dict["right"][1] = files[0]
    else:
        files = find_matching_files(re_pattern=RIGHT_PART_FILE_PATTERN, directory=dir_name)
        if files:
            videos_dict["right"] = {}
            for file in sorted(files):
                videos_dict["right"][get_lr_part_number(file)] = file
    if prune:
        videos_dict, discarded_videos = prune_chapters(videos=videos_dict)
        if discarded_videos:
            print(f"Discarding videos: {discarded_videos}")
    return videos_dict


def detect_video_rink_masks(
    game_id: str,
    videos_dict: VideosDict,
    device: torch.device = None,
) -> VideosDict:
    if device is None:
        gpu_allocator = GpuAllocator(gpus=None)
        device: torch.device = (
            torch.device("cuda", gpu_allocator.allocate_fast())
            if not gpu_allocator.is_single_lowmem_gpu(low_threshold_mb=1024 * 10)
            else torch.device("cpu")
        )

    keys = []
    images: List[torch.Tensor] = []
    for key, value in videos_dict.items():
        keys.append(key)
        min_chapter_key = min(value, key=value.get)
        frame = load_first_video_frame(value[min_chapter_key])
        images.append(frame)
        value["first_frame"] = load_first_video_frame(value[min_chapter_key])

    config_file, checkpoint = get_model_config(game_id=game_id, model_name="ice_rink_segm")

    frame_ir_masks = find_ice_rink_masks(
        image=images,
        config_file=prepend_root_dir(config_file),
        checkpoint=prepend_root_dir(checkpoint),
        device=device,
    )
    assert len(frame_ir_masks) == len(keys)
    for i, key in enumerate(keys):
        videos_dict[key]["rink_profile"] = frame_ir_masks[i]

    return videos_dict


def get_orientation_dict(rink_mask: torch.Tensor) -> Dict[str, float]:
    assert rink_mask.dtype == torch.bool
    assert rink_mask.ndim == 2
    width = image_width(rink_mask)
    height = image_height(rink_mask)

    float_mask = rink_mask.to(torch.float)
    divisor = 8
    left_sum = float_mask[:, : int(width // divisor)].sum().item()
    right_sum = float_mask[:, int(width - width // divisor) :].sum().item()
    top_sum = float_mask[: int(height // divisor), :].sum().item()
    bottom_sum = float_mask[int(height // divisor) :, :].sum().item()
    return {
        "left": left_sum,
        "right": right_sum,
        "top": top_sum,
        "bottom": bottom_sum,
    }


def get_orientation(rink_mask: torch.Tensor) -> str:
    assert rink_mask.dtype == torch.bool
    assert rink_mask.ndim == 2
    width = image_width(rink_mask)

    float_mask = rink_mask.to(torch.float)
    divisor = 8
    left_sum = float_mask[:, : int(width // divisor)].sum().item()
    right_sum = float_mask[:, int(width - width // divisor) :].sum().item()

    # show_image("mask", float_mask * 255)

    if left_sum > right_sum:
        # Most ice on the left of image, so this is the right side
        return "right"
    if right_sum > left_sum:
        # Most ice on the right of image, so this is the left side
        return "left"

    # For right end-zone, most ice will be at the bottom and left
    # height = image_height(rink_mask)
    # top_sum = float_mask[: int(height // divisor), :].sum()
    # bottom_sum = float_mask[int(height // divisor) :, :].sum()

    return "UNKNOWN"


def get_game_videos_analysis(
    game_id: str, videos_dict: Optional[VideosDict] = None, device: torch.device = None
) -> VideosDict:
    if videos_dict is None:
        videos_dict = get_available_videos(dir_name=get_game_dir(game_id=game_id))
    videos_dict = detect_video_rink_masks(game_id=game_id, videos_dict=videos_dict, device=device)

    new_dict: VideosDict = {}

    video_ssm: Dict[VideosDictKey, Dict[str, float]] = {}

    for key, value in videos_dict.items():
        mask = value["rink_profile"]["combined_mask"]

        sum_map: Dict[str, float] = get_orientation_dict(mask)
        video_ssm[key] = sum_map

        orientation = get_orientation(mask)
        if isinstance(key, str):
            if key.startswith("left"):
                assert orientation == "left"
            elif key.startswith("right"):
                assert orientation == "right"
        if orientation in videos_dict:
            if isinstance(key, str):
                assert key.startswith(orientation)
        else:
            new_dict[orientation] = value
        print(f"{key} orientation: {orientation}")
    videos_dict.update(new_dict)
    return videos_dict


def extract_chapters_file_list(chapter_map: VideoChapter) -> List[str]:
    numeric_keys = sorted(key for key in chapter_map.keys() if isinstance(key, (int, float)))
    if not numeric_keys:
        return []

    start = int(numeric_keys[0])
    expected = list(range(start, start + len(numeric_keys)))
    actual = [int(key) for key in numeric_keys]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"Non-consecutive chapter numbers: present={actual}, missing={missing}")

    return [chapter_map[index] for index in expected]


def configure_game_videos(
    game_id: str,
    force: bool = False,
    write_results: bool = True,
    device: torch.device = None,
) -> Dict[str, List[str]]:
    dir_name = get_game_dir(game_id=game_id)
    videos_dict = get_available_videos(dir_name=dir_name)
    if not force and ("left" in videos_dict and "right" in videos_dict):
        return {
            "left": extract_chapters_file_list(videos_dict["left"]),
            "right": extract_chapters_file_list(videos_dict["right"]),
        }
    private_config = get_game_config_private(game_id=game_id)
    if not force:
        # See if we have it already
        left_list = get_nested_value(private_config, "game.videos.left")
        right_list = get_nested_value(private_config, "game.videos.right")
        if left_list and right_list:
            left_list = [os.path.join(dir_name, p) for p in left_list]
            right_list = [os.path.join(dir_name, p) for p in right_list]
            return {
                "left": left_list,
                "right": right_list,
            }
    videos_dict = get_game_videos_analysis(game_id=game_id, device=device)
    left_list = extract_chapters_file_list(videos_dict["left"])
    right_list = extract_chapters_file_list(videos_dict["right"])
    # Make sure they both have the same chapters
    if write_results:
        assert videos_dict["left"].keys() == videos_dict["right"].keys()
        set_nested_value(private_config, "game.videos.left", [Path(p).name for p in left_list])
        set_nested_value(private_config, "game.videos.right", [Path(p).name for p in right_list])
        save_private_config(game_id=game_id, data=private_config)
    return {
        "left": left_list,
        "right": right_list,
    }


def _main(args: argparse.Namespace):
    game_id = args.game_id
    assert game_id
    results = configure_game_videos(game_id=game_id)
    # videos_dict = get_game_videos_analysis(game_id=game_id)
    print(results)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Video Orientation Analysis")
    parser.add_argument("--force", action="store_true", help="Force all recalculations")
    return parser


def main():
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()

    _main(args)


if __name__ == "__main__":
    main()
    print("Done.")
