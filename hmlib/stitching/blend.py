"""Seam blend selection shared by the stitching entry points.

The vocabulary matches HockeyMONStream's ``ParseBlendMode`` so a game config
written by either application means the same thing in both, with ``multiblend``
as the one HockeyMON-only addition. Blending is a render-time choice: it must
not reach :class:`hmlib.stitching.settings.StitchingSettings`, whose manifest is
calibration provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Canonical names, in the order an operator picks between them.
BLEND_MODES = ("laplacian", "alpha", "gpu-hard-seam", "multiblend")
# Modes the CUDA panorama stitchers can render. `multiblend` is the CPU
# enblend/multiblend path and has no GPU implementation.
GPU_BLEND_MODES = ("laplacian", "alpha", "gpu-hard-seam")
_BLEND_MODE_ALIASES = {
    "hard": "gpu-hard-seam",
    "hard-seam": "gpu-hard-seam",
}
DEFAULT_BLEND_MODE = "laplacian"
DEFAULT_BLEND_LEVELS = 11
# Mirrors hm-cupano's BlendSettings::kDefaultFeatherFraction / kMaxFeatherFraction.
DEFAULT_FEATHER_FRACTION = 0.05
MAX_FEATHER_FRACTION = 1.0


def normalize_blend_mode(value: Any) -> str:
    """Return the canonical blend mode name or raise.

    Accepts the same spellings as HockeyMONStream: case-insensitive, and
    underscores interchangeable with hyphens.
    """
    normalized = str(value).strip().lower().replace("_", "-")
    normalized = _BLEND_MODE_ALIASES.get(normalized, normalized)
    if normalized not in BLEND_MODES:
        choices = ", ".join(BLEND_MODES)
        raise ValueError(f"Unsupported stitching blend mode {value!r}; choose one of: {choices}")
    return normalized


def normalize_feather_fraction(value: Any) -> float:
    """Return a validated alpha crossfade width as a fraction of the narrowest camera.

    Messages name the offending value, so an unresolved ``GLOBAL.*`` reference from
    a config root that predates the key is diagnosable from the error alone.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"stitching.blend_feather_fraction must be a number; got {value!r}")
    try:
        fraction = float(value)
    except ValueError as exc:
        raise ValueError(
            f"stitching.blend_feather_fraction must be a number; got {value!r}"
        ) from exc
    if not math.isfinite(fraction):
        raise ValueError(f"stitching.blend_feather_fraction must be finite; got {value!r}")
    if not 0.0 <= fraction <= MAX_FEATHER_FRACTION:
        raise ValueError(
            f"stitching.blend_feather_fraction must be in [0, {MAX_FEATHER_FRACTION:g}]; "
            f"got {value!r}"
        )
    return fraction


@dataclass(frozen=True)
class BlendSettings:
    """Resolved seam blend choice for one stitcher construction."""

    mode: str = DEFAULT_BLEND_MODE
    feather_fraction: float = DEFAULT_FEATHER_FRACTION
    max_levels: int = DEFAULT_BLEND_LEVELS

    @property
    def levels(self) -> int:
        """Pyramid level count in the historical encoding, where 0 means hard seam.

        hm-cupano interprets a bare level count exactly this way, and alpha mode
        carries its width in ``feather_fraction`` instead, so 0 is right there too.
        """
        return self.max_levels if self.mode == "laplacian" else 0

    def require_gpu_mode(self) -> "BlendSettings":
        """Raise when the CUDA panorama stitchers cannot render this mode."""
        if self.mode not in GPU_BLEND_MODES:
            choices = ", ".join(GPU_BLEND_MODES)
            raise ValueError(
                f"Stitching blend mode {self.mode!r} has no GPU implementation; "
                f"choose one of: {choices}, or stitch with --python-blender"
            )
        return self


def resolve_blend_settings(
    stitch_config: Optional[Mapping[str, Any]] = None,
    *,
    blend_mode: Any = None,
    blend_feather_fraction: Any = None,
    max_blend_levels: Any = None,
) -> BlendSettings:
    """Resolve the effective blend choice from a ``stitching`` mapping plus overrides.

    Keyword overrides win over the mapping; ``None`` means unspecified.
    """
    config: Mapping[str, Any] = stitch_config or {}
    mode = blend_mode if blend_mode is not None else config.get("blend_mode")
    mode = normalize_blend_mode(mode) if mode is not None else DEFAULT_BLEND_MODE
    fraction = (
        blend_feather_fraction
        if blend_feather_fraction is not None
        else config.get("blend_feather_fraction")
    )
    fraction = (
        normalize_feather_fraction(fraction) if fraction is not None else DEFAULT_FEATHER_FRACTION
    )
    levels = max_blend_levels if max_blend_levels is not None else config.get("max_blend_levels")
    if levels is None:
        levels = DEFAULT_BLEND_LEVELS
    if isinstance(levels, bool) or not isinstance(levels, (int, float, str)):
        raise ValueError("stitching.max_blend_levels must be an integer")
    try:
        levels = int(levels)
    except ValueError as exc:
        raise ValueError("stitching.max_blend_levels must be an integer") from exc
    # A non-positive count is the legacy "use the default" spelling on the CLI,
    # not a request for a hard seam; `blend_mode` decides that.
    if levels <= 0:
        levels = DEFAULT_BLEND_LEVELS
    return BlendSettings(mode=mode, feather_fraction=fraction, max_levels=levels)


__all__ = [
    "BLEND_MODES",
    "DEFAULT_BLEND_LEVELS",
    "DEFAULT_BLEND_MODE",
    "DEFAULT_FEATHER_FRACTION",
    "GPU_BLEND_MODES",
    "MAX_FEATHER_FRACTION",
    "BlendSettings",
    "normalize_blend_mode",
    "normalize_feather_fraction",
    "resolve_blend_settings",
]
