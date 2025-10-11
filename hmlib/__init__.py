import importlib

import numpy as np

# Define a dictionary that mimics the structure of np.sctypes
# You'll need to populate this with the specific dtypes your code expects
# This is a simplified example; a full implementation would be more extensive
# Define the sctypes dictionary, based on NumPy 1.x behavior
# WARNING: This is an incomplete snapshot and may not fully replicate all behavior.
# It is meant to unblock basic operations, not to be a perfect substitute.
sctypes_patch = {
    "int": [np.int8, np.int16, np.int32, np.int64, np.intc, np.intp, np.byte, np.short, np.long, np.longlong],
    "uint": [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.uintc,
        np.uintp,
        np.ubyte,
        np.ushort,
        np.ulong,
        np.ulonglong,
    ],
    "float": [np.float16, np.float32, np.float64, np.half, np.single, np.double, np.longdouble],
    "complex": [np.complex64, np.complex128, np.complex256, np.csingle, np.cdouble, np.clongdouble],
    "others": [np.bool_, np.object_, np.str_, np.unicode_, np.void, np.datetime64, np.timedelta64],
}

# Monkey-patch np.sctypes
np.sctypes = _sctypes_mock

from .hm_transforms import HmLoadImageFromWebcam

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
