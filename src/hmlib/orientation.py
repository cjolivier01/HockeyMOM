import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import torch

from hmlib.config import get_game_config_private, get_game_dir, save_private_config
from hmlib.hm_opts import hm_opts
from hmlib.utils.video import load_first_video_frame

# GoPro pattern is GXzzxxxx.mp4, where zz is chapter number and zzzz is video
# number
GOPRO_FILE_PATTERN: str = r"^G[A-Z][0-9]{6}\.mp4$"
LEFT_PART_FILE_PATTERN: str = r"left-[0-9]\.mp4$"
RIGHT_PART_FILE_PATTERN: str = r"right-[0-9]\.mp4$"
LEFT_FILE_PATTERN: str = r"left.mp4"
RIGHT_FILE_PATTERN: str = r"right.mp4"


VideosDict = Dict[Union[int, str], List[Dict[Union[int, str], Any]]]

def gopro_get_video_and_chapter(filename: Path) -> Tuple[int, int]:
    """
    Get video and chapter number

    Returns: (video number, chapter number)
    """
    name = filename.stem
    assert name[0] == "G"
    assert name[1] in {"H", "X"}
    return int(name[4:8]), int(name[2:4])


def get_lr_part_number(filename: Path) -> int:
    """
    Get video and chapter number

    Returns: (video number, chapter number)
    """
    name = filename.stem
    tokens = name.split("-")
    return int(tokens[-1])


def find_matching_files(pattern: str, directory: str) -> List[Path]:
    # Regex to match the file format 'GLXXXXXX.mp4'
    pattern = re.compile(pattern)

    # List to store the names of matching files
    matching_files: List[Path] = []

    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        if pattern.match(filename):
            matching_files.append(directory / Path(filename))

    return sorted(matching_files)


def get_available_videos(dir_name: str) -> VideosDict:
    """
    Get available videos in the given directory

    :return: # Video # / left|right -> Chapter # -> Path
    """
    gopro_files: List[Path] = find_matching_files(pattern=GOPRO_FILE_PATTERN, directory=dir_name)
    # Video # / left|right -> Chapter # -> Path
    videos_dict: Dict[Union[int, str], List[Dict[int, Path]]] = OrderedDict()
    for file in gopro_files:
        video, chapter = gopro_get_video_and_chapter(filename=file)
        if video not in videos_dict:
            videos_dict[video] = {}
        videos_dict[video][chapter] = file

    # Plain left file
    files: List[Path] = find_matching_files(pattern=LEFT_FILE_PATTERN, directory=dir_name)
    if files:
        assert len(files) == 1
        videos_dict["left"] = {}
        videos_dict["left"][1] = files[0]
    else:
        files = find_matching_files(pattern=LEFT_PART_FILE_PATTERN, directory=dir_name)
        if files:
            videos_dict["left"] = {}
            for file in files:
                videos_dict["left"][get_lr_part_number(file)] = file

    # Plain right file
    files: List[Path] = find_matching_files(pattern=LEFT_FILE_PATTERN, directory=dir_name)
    if files:
        assert len(files) == 1
        videos_dict["right"] = {}
        videos_dict["right"][1] = files[0]
    else:
        files = find_matching_files(pattern=RIGHT_PART_FILE_PATTERN, directory=dir_name)
        if files:
            videos_dict["right"] = {}
            for file in files:
                videos_dict["right"][get_lr_part_number(file)] = file
    return videos_dict


def load_first_video_frames(videos_dict: VideosDict) -> VideosDict:
    keys = []
    # files = []
    for key, value in videos_dict.items():
        keys.append(key)
        # files.append(load_first_frame(value[1]))
        value["first_frame"] = load_first_video_frame(value[1])
    return videos_dict


def main(args: argparse.Namespace):
    game_id = args.game_id
    assert game_id
    dir_name = get_game_dir(game_id=game_id)
    videos_dict = get_available_videos(dir_name=dir_name)
    videos_dict = load_first_video_frames(videos_dict)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Video Orientation Analysis")
    parser.add_argument("--force", action="store_true", help="Force all recalculations")
    return parser


if __name__ == "__main__":
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()

    args.game_id = "pdp"

    main(args)
    print("Done.")
