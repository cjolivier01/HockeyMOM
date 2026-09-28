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
# Modes the Python blender can render. Alpha is a hm-cupano kernel with no
# Python equivalent. `multiblend` is in neither set: it names the calibration-time
# enblend/multiblend binaries, and create_blender_config returns a seamless
# config for it that the blender then dereferences, so no video path can run it.
PYTHON_BLEND_MODES = ("laplacian", "gpu-hard-seam")
_BLEND_MODE_ALIASES = {
    "hard": "gpu-hard-seam",
    "hard-seam": "gpu-hard-seam",
}
DEFAULT_BLEND_MODE = "laplacian"
DEFAULT_BLEND_LEVELS = 11
# Mirrors hm-cupano's BlendSettings::kDefaultFeatherFraction / kMaxFeatherFraction.
DEFAULT_FEATHER_FRACTION = 0.05
MAX_FEATHER_FRACTION = 1.0


def config_blend_mode(value: Any) -> Optional[str]:
    """Normalize a configured mode, or None when the config specifies none.

    Absent, an explicit null, and a blank string all mean "inherit", matching
    what the surrounding config machinery means by an empty scalar. Anything
    else is returned folded but unvalidated, so a caller can still show or
    report a mode this build does not know.
    """
    if value is None:
        return None
    folded = str(value).strip().lower().replace("_", "-")
    return folded or None


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


def normalize_max_blend_levels(value: Any) -> int:
    """Return a validated Laplacian pyramid depth, or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"stitching.max_blend_levels must be an integer; got {value!r}")
    try:
        # int() raises OverflowError, not ValueError, for an infinity - which a
        # YAML `.inf` produces - and truncates a float, which hides a typo.
        levels = int(value)
        if isinstance(value, float) and levels != value:
            raise ValueError
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"stitching.max_blend_levels must be an integer; got {value!r}") from exc
    # The native stitchers take an int; reject here rather than after the
    # stitching lock and the artifact rewrite.
    if not -(2**31) <= levels < 2**31:
        raise ValueError(f"stitching.max_blend_levels is out of range; got {value!r}")
    return levels


def normalize_feather_fraction(value: Any) -> float:
    """Return a validated alpha crossfade width as a fraction of the narrowest camera.

    Messages name the offending value, so an unresolved ``GLOBAL.*`` reference from
    a config root that predates the key is diagnosable from the error alone.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"stitching.blend_feather_fraction must be a number; got {value!r}")
    try:
        fraction = float(value)
    except (ValueError, OverflowError) as exc:
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

    def __post_init__(self) -> None:
        # Mirrors hm-cupano's BlendSettings::Validate, and coerces every field, so
        # a caller that builds this directly cannot get past validation with a
        # float depth or leak a TypeError out of a comparison.
        object.__setattr__(self, "mode", normalize_blend_mode(self.mode))
        object.__setattr__(
            self, "feather_fraction", normalize_feather_fraction(self.feather_fraction)
        )
        object.__setattr__(self, "max_levels", normalize_max_blend_levels(self.max_levels))
        if self.max_levels < 0:
            raise ValueError(
                f"stitching.max_blend_levels must not be negative; got {self.max_levels!r}"
            )
        # Naming laplacian with no levels is a caller mistake, not a request for a
        # hard seam: callers spell that `mode="gpu-hard-seam"`, and
        # resolve_blend_settings turns a non-positive count into the default first.
        if self.mode == "laplacian" and self.max_levels < 1:
            raise ValueError(
                f"Laplacian blending needs at least one pyramid level; got {self.max_levels!r}"
            )

    @property
    def levels(self) -> int:
        """Pyramid level count in the historical encoding, where 0 means hard seam.

        hm-cupano interprets a bare level count exactly this way, and alpha mode
        carries its width in ``feather_fraction`` instead, so 0 is right there too.
        """
        return self.max_levels if self.mode == "laplacian" else 0

    def _require_renderable(self, blender: str, renderable: tuple, other: tuple, hint: str):
        if self.mode in renderable:
            return self
        message = (
            f"Stitching blend mode {self.mode!r} cannot be rendered by the {blender} blender; "
            f"choose one of: {', '.join(renderable)}"
        )
        # Only suggest the other path when it can actually run this mode;
        # `multiblend` is in neither set and pointing at either is a dead end.
        if self.mode in other:
            message = f"{message}, or {hint}"
        raise ValueError(message)

    def require_gpu_mode(self) -> "BlendSettings":
        """Raise when the CUDA panorama stitchers cannot render this mode."""
        return self._require_renderable(
            "GPU", GPU_BLEND_MODES, PYTHON_BLEND_MODES, "stitch with --python-blender"
        )

    def require_python_mode(self) -> "BlendSettings":
        """Raise when the Python blender cannot render this mode.

        Table-driven on purpose: a mode added to BLEND_MODES but to neither
        renderable set is refused by both paths rather than falling through to
        whichever branch happens not to match its name.
        """
        return self._require_renderable(
            "Python", PYTHON_BLEND_MODES, GPU_BLEND_MODES, "drop --python-blender"
        )


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
    mode = config_blend_mode(mode)
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
    levels = DEFAULT_BLEND_LEVELS if levels is None else normalize_max_blend_levels(levels)
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
    "PYTHON_BLEND_MODES",
    "BlendSettings",
    "config_blend_mode",
    "normalize_blend_mode",
    "normalize_feather_fraction",
    "normalize_max_blend_levels",
    "resolve_blend_settings",
]
