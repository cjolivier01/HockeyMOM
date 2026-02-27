"""Utility subpackage: GPU helpers, image ops, progress UI, etc."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .mean_tracker import MeanTracker

__all__ = [
    "MeanTracker",
]


def __getattr__(name: str) -> Any:
    if name == "MeanTracker":
        from .mean_tracker import MeanTracker

        return MeanTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
