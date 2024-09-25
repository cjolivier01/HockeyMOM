import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

from hmlib import load
from hmlib.config import get_game_config_private, get_game_dir, save_private_config
from hmlib.hm_opts import hm_opts

# GoPro pattern is GXzzxxxx.mp4, where zz is chapter number and zzzz is video
# number
GOPRO_FILE_PATTERN: str = r"^G[A-Z][0-9]{6}\.mp4$"
LEFT_PART_FILE_PATTERN: str = r"left-[0-9]\.mp4$"
RIGHT_PART_FILE_PATTERN: str = r"right-[0-9]\.mp4$"
LEFT_FILE_PATTERN: str = r"left.mp4"
RIGHT_FILE_PATTERN: str = r"right.mp4"


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


def get_available_videos(dir_name: str) -> Dict[Union[int, str], List[Dict[int, Path]]]:
    """
    Get available videos in the given directory

    :return: # Video # / left|right -> Chapter # -> Path
    """
    gopro_files: List[Path] = find_matching_files(pattern=GOPRO_FILE_PATTERN, directory=dir_name)
    # Video # / left|right -> Chapter # -> Path
    video_dict: Dict[Union[int, str], List[Dict[int, Path]]] = OrderedDict()
    for file in gopro_files:
        video, chapter = gopro_get_video_and_chapter(filename=file)
        if video not in video_dict:
            video_dict[video] = {}
        video_dict[video][chapter] = file

    # Plain left file
    files: List[Path] = find_matching_files(pattern=LEFT_FILE_PATTERN, directory=dir_name)
    if files:
        assert len(files) == 1
        video_dict["left"] = {}
        video_dict["left"][1] = files[0]
    else:
        files = find_matching_files(pattern=LEFT_PART_FILE_PATTERN, directory=dir_name)
        if files:
            video_dict["left"] = {}
            for file in files:
                video_dict["left"][get_lr_part_number(file)] = file

    # Plain right file
    files: List[Path] = find_matching_files(pattern=LEFT_FILE_PATTERN, directory=dir_name)
    if files:
        assert len(files) == 1
        video_dict["right"] = {}
        video_dict["right"][1] = files[0]
    else:
        files = find_matching_files(pattern=RIGHT_PART_FILE_PATTERN, directory=dir_name)
        if files:
            video_dict["right"] = {}
            for file in files:
                video_dict["right"][get_lr_part_number(file)] = file
    return video_dict


def main(args: argparse.Namespace):
    game_id = args.game_id
    assert game_id
    dir_name = get_game_dir(game_id=game_id)
    video_dict = get_available_videos(dir_name=dir_name)
    print(video_dict)


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
