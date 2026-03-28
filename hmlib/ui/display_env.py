"""Helpers for deciding whether OpenCV GUI imports are safe."""

from __future__ import annotations

import ctypes
import ctypes.util
import os

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def has_usable_x11_display(display_name: str) -> bool:
    library_name = ctypes.util.find_library("X11")
    if not library_name:
        return False
    try:
        x11 = ctypes.cdll.LoadLibrary(library_name)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int
        handle = x11.XOpenDisplay(display_name.encode("utf-8"))
        if not handle:
            return False
        x11.XCloseDisplay(handle)
        return True
    except OSError:
        return False


def has_local_display_env() -> bool:
    if env_flag("HM_FORCE_HEADLESS_PREVIEW"):
        return False
    if os.name == "nt":
        return True
    display_name = os.environ.get("DISPLAY", "").strip()
    if not display_name:
        return False
    return has_usable_x11_display(display_name)


def sanitize_display_env_for_cv2() -> None:
    if has_local_display_env():
        return
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
