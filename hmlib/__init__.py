import importlib
import os
import sys

import numpy as np

from .hm_transforms import HmLoadImageFromWebcam

# Get the absolute path of this file's parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add it to sys.path if not already present
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



# Define a dictionary that mimics the structure of np.sctypes
# You'll need to populate this with the specific dtypes your code expects
# This is a simplified example; a full implementation would be more extensive
_sctypes_mock = {
    "int": [np.int8, np.int16, np.int32, np.int64],
    "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
    "float": [np.float16, np.float32, np.float64],
    "complex": [np.complex64, np.complex128],
    "others": [np.bool_, np.object_, np.str_, np.bytes_],
}

# Monkey-patch np.sctypes
np.sctypes = _sctypes_mock


__all__ = [
    "HmLoadImageFromWebcam",
    "LazyImport",
]


class LazyImport:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return getattr(self.module, name)


# from .hm_transforms import *
# import hmlib.hm_transforms
# import hmlib.models.end_to_end
