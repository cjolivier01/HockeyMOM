import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from hmlib.config import get_game_config_private, get_game_dir, save_private_config
from hmlib.hm_opts import hm_opts
from hmlib.models.loader import get_model_config
from hmlib.segm.ice_rink import find_ice_rink_masks
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.image import image_height, image_width
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


def detect_video_rink_masks(
    game_id: str,
    videos_dict: VideosDict,
    device: torch.device = None,
) -> VideosDict:
    if device is None:
        gpu_allocator = GpuAllocator(gpus=args.gpus)
        device: torch.device = (
            torch.device("cuda", gpu_allocator.allocate_fast())
            if not gpu_allocator.is_single_lowmem_gpu(low_threshold_mb=1024 * 10)
            else torch.device("cpu")
        )

    keys = []
    images: List[torch.Tensor] = []
    for key, value in videos_dict.items():
        keys.append(key)
        # files.append(load_first_frame(value[1]))
        frame = load_first_video_frame(value[1])
        images.append(frame)
        # value["first_frame"] = load_first_video_frame(value[1])

    config_file, checkpoint = get_model_config(game_id=game_id, model_name="ice_rink_segm")

    frame_ir_masks = find_ice_rink_masks(
        image=images,
        config_file=config_file,
        checkpoint=checkpoint,
        device=device,
    )
    assert len(frame_ir_masks) == len(keys)
    for i, key in enumerate(keys):
        videos_dict[key]["rink_profile"] = frame_ir_masks[i]

    return videos_dict


def get_orientation(rink_mask: torch.Tensor) -> str:
    assert rink_mask.dtype == torch.bool
    assert rink_mask.ndim == 2
    width = image_width(rink_mask)

    float_mask = rink_mask.to(torch.float)
    divisor = 4
    left_sum = float_mask[:, : int(width // divisor)].sum()
    right_sum = float_mask[:, int(width // divisor) :].sum()
    if left_sum > right_sum:
        # Most ice on the left of image, so this is the right side
        return "right"
    elif right_sum < left_sum:
        return "left"

    # For right end-zone, most ice will be at the bottom and left
    # height = image_height(rink_mask)
    # top_sum = float_mask[: int(height // divisor), :].sum()
    # bottom_sum = float_mask[int(height // divisor) :, :].sum()

    return "UNKNOWN"


def main(args: argparse.Namespace):
    game_id = args.game_id
    assert game_id
    dir_name = get_game_dir(game_id=game_id)
    videos_dict = get_available_videos(dir_name=dir_name)
    videos_dict = detect_video_rink_masks(game_id=game_id, videos_dict=videos_dict)

    for key, value in videos_dict.items():
        mask = value["rink_profile"]["combined_mask"]
        orientation = get_orientation(torch.from_numpy(mask))
        if key.startswith("left"):
            assert orientation == "left"
        elif key.startswith("right"):
            assert orientation == "right"
        if orientation in videos_dict:
            assert orientation == key
        else:
            videos_dict[orientation] = value
        print(f"{key} orientation: {orientation}")


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
