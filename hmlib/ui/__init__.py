"""Lightweight UI helpers for image and video visualization.

Expose :func:`show_image` and :class:`Shower` for use in CLIs and notebooks.
"""

from .show import show_image
from .shower import Shower

__all__ = [
    "show_image",
    "Shower",
]
