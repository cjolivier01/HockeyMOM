import numpy as np
from typing import List
from ._hockeymom import _hello_world, _enblend, _emblend_images

# from ._hockeymom import  _enblend


def hello_world():
    _hello_world()


def enblend(output_file: str, input_files: List[str]) -> int:
    return _enblend(output_file, input_files)


def emblend_images(
    image_left: np.array,
    image_right: np.array,
    xy_pos_1: List[int] = [0, 0],
    xy_pos_2: List[int] = [0, 0],
) -> np.array:
    return _emblend_images(image_left, xy_pos_1, image_right, xy_pos_2)
