"""Shared `hockeymom` native-extension symbols for Python callers."""

from __future__ import annotations

from hockeymom.core import (
    BlenderConfig,
    CudaStitchPanoF32,
    CudaStitchPanoNF32,
    CudaStitchPanoNU8,
    CudaStitchPanoU8,
    HmByteTrackConfig,
    HmTracker,
    HmTrackerPredictionMode,
    ImageBlender,
    ImageBlenderMode,
    ImageRemapper,
    RemapImageInfo,
    WHDims,
)

try:
    from hockeymom.core import EnBlender
except ImportError:
    EnBlender = None

HOCKEYMOM_AVAILABLE = True
HOCKEYMOM_IMPORT_ERROR = None


def hockeymom_error_message() -> str:
    return ""


def require_hockeymom(feature: str) -> None:
    del feature


__all__ = [
    "BlenderConfig",
    "CudaStitchPanoF32",
    "CudaStitchPanoNF32",
    "CudaStitchPanoNU8",
    "CudaStitchPanoU8",
    "EnBlender",
    "HmByteTrackConfig",
    "HmTracker",
    "HmTrackerPredictionMode",
    "HOCKEYMOM_AVAILABLE",
    "HOCKEYMOM_IMPORT_ERROR",
    "ImageBlender",
    "ImageBlenderMode",
    "ImageRemapper",
    "RemapImageInfo",
    "WHDims",
    "hockeymom_error_message",
    "require_hockeymom",
]
