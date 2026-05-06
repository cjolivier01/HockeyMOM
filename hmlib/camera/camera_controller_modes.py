from __future__ import annotations

from typing import Optional


def normalize_play_tracker_camera_controller(
    controller: Optional[str], camera_model: Optional[str]
) -> str:
    normalized = str(controller or "rule")
    if normalized == "drivegpt":
        if not camera_model:
            raise RuntimeError(
                "rink.camera.controller='drivegpt' requires rink.camera.camera_model so "
                "PlayTracker only disables the C++ backend when learned camera boxes are available"
            )
        return "gpt"
    return normalized
