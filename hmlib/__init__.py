"""High-level HockeyMOM Python package.

This package exposes reusable pipelines, CLI helpers and utility code on top of
the core :mod:`hockeymom` native extension.

@see @ref hmlib.hm_opts "hmlib.hm_opts" for shared CLI options.
@see @ref hmlib.hm_transforms "hmlib.hm_transforms" for common data transforms.
@see @ref hmlib.utils.progress_bar.ProgressBar "ProgressBar" for CLI progress UI.
"""

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
    """Lazy module proxy that imports on first attribute access.

    This keeps startup overhead low for CLIs by delaying heavy imports until
    they are actually needed.

    @param module_name: Fully-qualified module path to import.
    @see @ref hmlib.builder.HM "HM registry" for dynamic pipeline registration.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return getattr(self.module, name)


from .constants import *
