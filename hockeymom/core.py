from typing import List
from ._hockeymom import  _hello_world, _enblend

def hello_world():
    _hello_world()

def enblend(output_file: str, input_files: List[str]) -> int:
    return _enblend(output_file, input_files)
