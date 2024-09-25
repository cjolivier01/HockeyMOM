import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
    name = filename.name
    assert name[0] == 'G'
    assert name[1] in {'H', 'X'}
    return int(name[4:8]), int(name[2:4])


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


def get_gopro_videos(directory: str) -> List[str]:
    pass


def main(args: argparse.Namespace):
    game_id = args.game_id
    assert game_id
    dir_name = get_game_dir(game_id=game_id)
    gopro_files: List[Path] = find_matching_files(pattern=GOPRO_FILE_PATTERN, directory=dir_name)
    # Video # / left|right -> Chapter # -> Path
    video_dict: Dict[Union[int, str], List[Dict[int, Path]]] = OrderedDict()
    for file in gopro_files:
        video, chapter = gopro_get_video_and_chapter(filename=file)
        if video not in video_dict:
            video_dict[video] = dict()
        video_dict[video][chapter] = file
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
