"""Compatibility layer for optional `hockeymom` native-extension imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional

_core = None
HOCKEYMOM_IMPORT_ERROR: Optional[BaseException] = None

try:
    _core = import_module("hockeymom.core")
except Exception as exc:  # pragma: no cover - exercised in ROCm/non-native envs
    HOCKEYMOM_IMPORT_ERROR = exc


HOCKEYMOM_AVAILABLE = _core is not None


def hockeymom_error_message() -> str:
    if HOCKEYMOM_IMPORT_ERROR is None:
        return "unknown import error"
    return f"{type(HOCKEYMOM_IMPORT_ERROR).__name__}: {HOCKEYMOM_IMPORT_ERROR}"


def require_hockeymom(feature: str) -> None:
    if HOCKEYMOM_AVAILABLE:
        return
    raise RuntimeError(
        f"{feature} requires the hockeymom native extension, but it is unavailable "
        f"({hockeymom_error_message()})."
    )


@dataclass
class WHDims:
    width: int = 0
    height: int = 0


class BlenderConfig:
    def __init__(self) -> None:
        self.mode: str = ""
        self.levels: int = 0
        self.seam: Any = None
        self.xor_map: Any = None
        self.lazy_init: bool = False
        self.interpolation: str = "bilinear"
        self.device: str = "cpu"


class RemapImageInfo:
    pass


class HmByteTrackConfig:
    def __init__(self) -> None:
        self.init_track_thr = 0.7
        self.obj_score_thrs_low = 0.1
        self.obj_score_thrs_high = 0.6
        self.match_iou_thrs_high = 0.1
        self.match_iou_thrs_low = 0.5
        self.match_iou_thrs_tentative = 0.3
        self.track_buffer_size = 30
        self.num_frames_to_keep_lost_tracks = 30
        self.weight_iou_with_det_scores = True
        self.num_tentatives = 3
        self.return_user_ids = False
        self.return_track_age = False
        self.prediction_mode = "BoundingBox"


class HmTrackerPredictionMode:
    BoundingBox = "BoundingBox"


class ImageBlenderMode:
    Laplacian = "laplacian"
    HardSeam = "hard-seam"


class _UnavailableNativeType:
    pass


if HOCKEYMOM_AVAILABLE:
    WHDims = _core.WHDims
    BlenderConfig = _core.BlenderConfig
    RemapImageInfo = _core.RemapImageInfo
    ImageBlender = _core.ImageBlender
    ImageBlenderMode = _core.ImageBlenderMode
    EnBlender = getattr(_core, "EnBlender", None)
    ImageRemapper = getattr(_core, "ImageRemapper", None)
    CudaStitchPanoF32 = _core.CudaStitchPanoF32
    CudaStitchPanoNF32 = _core.CudaStitchPanoNF32
    CudaStitchPanoNU8 = _core.CudaStitchPanoNU8
    CudaStitchPanoU8 = _core.CudaStitchPanoU8
    HmByteTrackConfig = _core.HmByteTrackConfig
    HmTrackerPredictionMode = _core.HmTrackerPredictionMode
else:
    ImageBlender = None
    EnBlender = None
    ImageRemapper = None
    CudaStitchPanoF32 = _UnavailableNativeType
    CudaStitchPanoNF32 = _UnavailableNativeType
    CudaStitchPanoNU8 = _UnavailableNativeType
    CudaStitchPanoU8 = _UnavailableNativeType


__all__ = [
    "BlenderConfig",
    "CudaStitchPanoF32",
    "CudaStitchPanoNF32",
    "CudaStitchPanoNU8",
    "CudaStitchPanoU8",
    "EnBlender",
    "HmByteTrackConfig",
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
