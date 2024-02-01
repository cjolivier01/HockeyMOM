import numpy as np
from typing import List
import torch

# Functions
from ._hockeymom import (
    _hello_world,
    _enblend,
    _nona_process_images,
    _add_to_stitching_data_loader,
)

# Classes
from ._hockeymom import (
    HMPostprocessConfig,
    ImagePostProcessor,
    HmNona,
    EnBlender,
    StitchingDataLoader,
    RemapperConfig,
    BlenderConfig,
    ImageRemapper,
    SortedRGBImageQueue,
    SortedPyArrayUin8Queue,
    FFmpegVideoWriter,
)

__all__ = [
    "hello_world",
    "enblend",
    "HmNona",
    "StitchingDataLoader",
    "RemapperConfig",
    "ImageRemapper",
    "BlenderConfig"
    "nona_process_images",
    "EnBlender",
    "FFmpegVideoWriter",
]


def hello_world():
    _hello_world()


def enblend(output_file: str, input_files: List[str]) -> int:
    return _enblend(output_file, input_files)


def nona_process_images(nona: HmNona, image_left: np.array, image_right: np.array):
    return _nona_process_images(nona, image_left, image_right)


def close_stitching_data_loader(
    data_loader: StitchingDataLoader,
    frame_id: int,
) -> int:
    return _add_to_stitching_data_loader(data_loader, frame_id, None, None)


def add_to_stitching_data_loader(
    data_loader: StitchingDataLoader,
    frame_id: int,
    image_left: np.array,
    image_right: np.array,
) -> int:
    return _add_to_stitching_data_loader(data_loader, frame_id, image_left, image_right)

