import os
import re
from pathlib import Path
from typing import List

from hmlib.config import get_game_config_private, get_game_dir, save_private_config
from hmlib.hm_opts import hm_opts

GOPRO_FILE_PATTERN: str = r'^GL[A-Z][0-9]{6}\.mp4$'
LEFT_PART_FILE_PATTERN: str = r'left-[0-9]\.mp4$'
RIGHT_PART_FILE_PATTERN: str = r'right-[0-9]\.mp4$'
LEFT_FILE_PATTERN: str = r"left.mp4"
RIGHT_FILE_PATTERN: str = r"right.mp4"

def find_matching_files(pattern: str, directory: str) -> List[Path]:
    # Regex to match the file format 'GLXXXXXX.mp4'
    pattern = re.compile(r'^GL[A-Z][0-9]{6}\.mp4$')

    # List to store the names of matching files
    matching_files = []

    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        if pattern.match(filename):
            matching_files.append(filename)

    return matching_files


def get_gopro_videos(directory: str) -> List[str]:
    pass

def main():
    pass


if __name__ == "__main__":
    parser = hm_opts.parser()
    main()
