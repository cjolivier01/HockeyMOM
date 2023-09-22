import numpy as np
from typing import List
from ._hockeymom import  _hello_world, _enblend, _emblend_images
#from ._hockeymom import  _enblend

def hello_world():
    _hello_world()

def enblend(output_file: str, input_files: List[str]) -> int:
    return _enblend(output_file, input_files)

def emblend_images(image_left: np.array, image_right: np.array) -> np.array:
    return _emblend_images(image_left, image_right)
